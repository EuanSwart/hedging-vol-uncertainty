"""unit tests for the building blocks: pricing, simulation, the bsb solver, posterior, networks."""
import math

import pytest
import torch

from hedging.baselines import bs_delta_path, posterior_mean_delta, posterior_mean_sigma
from hedging.bsb import bsb_delta_path, bsb_price
from hedging.config import Calibration, N_STEPS, S0, T
from hedging.losses import cvar, shortfall
from hedging.networks import GRUHedger, MLPHedger, build_features
from hedging.payoffs import bs_delta, bs_price, payoff
from hedging.simulate import gbm_from_noise, make_generator, sample_prior_sigma, simulate_prior_paths

# a fixed calibration so the tests do not depend on run_calibrate.py
CALIB = Calibration(band=(0.07, 0.30), a0=2.22, b0=0.0278, sigma_ref=0.132)


def test_bs_call_matches_known_value():
    # at the money, sigma 0.2, one year: price is about 0.0797 * S
    p = bs_price("call", torch.tensor(100.0, dtype=torch.float64), 0.2, 1.0)
    assert abs(float(p) - 7.9656) < 1e-3


def test_bs_delta_is_derivative_of_price():
    S = torch.tensor([95.0, 100.0, 108.0], dtype=torch.float64, requires_grad=True)
    bs_price("butterfly", S, 0.2, 0.25).sum().backward()
    assert torch.allclose(S.grad, bs_delta("butterfly", S.detach(), 0.2, 0.25), atol=1e-6)


def test_butterfly_payoff_is_a_tent():
    S = torch.tensor([80.0, 90.0, 100.0, 110.0, 120.0])
    assert payoff("butterfly", S).tolist() == [0.0, 0.0, 10.0, 0.0, 0.0]


def test_paths_are_a_martingale_with_the_right_volatility():
    gen = make_generator(0)
    S, sigma = simulate_prior_paths(200_000, gen, CALIB)
    assert abs(float(S[:, -1].mean()) - S0) < 0.1
    realised = torch.log(S[:, 1:] / S[:, :-1]).pow(2).sum(1) / T          # annualised variance per path
    assert abs(float((realised / sigma ** 2).mean()) - 1.0) < 0.01


def test_prior_samples_stay_inside_the_band():
    s = sample_prior_sigma(50_000, make_generator(1), CALIB)
    assert float(s.min()) >= 0.07 - 1e-6 and float(s.max()) <= 0.30 + 1e-6


def test_bsb_call_equals_black_scholes_at_the_top_of_the_band():
    price = bsb_price("call", CALIB.band)
    bs_hi = float(bs_price("call", torch.tensor(S0, dtype=torch.float64), CALIB.band[1], T))
    assert abs(price - bs_hi) / bs_hi < 0.01


def test_bsb_butterfly_reproduces_the_report_numbers():
    # report, section 5.2: band [0.10, 0.30] gives 6.67 against 6.10 for the best constant volatility
    price = bsb_price("butterfly", (0.10, 0.30))
    best_constant = max(float(bs_price("butterfly", torch.tensor(S0, dtype=torch.float64), s, T))
                        for s in torch.linspace(0.10, 0.30, 41).tolist())
    assert 6.6 < price < 6.75
    assert 6.05 < best_constant < 6.15 and price > best_constant


def test_posterior_mean_moves_towards_realised_volatility():
    gen = make_generator(2)
    for sigma_true in (0.08, 0.25):
        z = torch.randn(2000, N_STEPS, generator=gen)
        S = gbm_from_noise(z, sigma_true)
        est = posterior_mean_sigma(S, CALIB)
        assert abs(float(est[:, -1].mean()) - sigma_true) < abs(float(est[:, 0].mean()) - sigma_true)


def test_posterior_mean_at_date_zero_is_the_prior_mean():
    S = gbm_from_noise(torch.randn(4, N_STEPS), 0.15)
    est = posterior_mean_sigma(S, CALIB)[:, 0]
    assert torch.allclose(est, est[0].expand_as(est))                       # same for every path, no data yet
    assert CALIB.band[0] < float(est[0]) < CALIB.band[1]


@pytest.mark.parametrize("make", [lambda: GRUHedger("butterfly", 0.132), lambda: MLPHedger("butterfly", 0.132)])
def test_network_starts_at_black_scholes_and_has_no_look_ahead(make):
    torch.manual_seed(0)
    model = make()
    S = gbm_from_noise(torch.randn(8, N_STEPS), 0.2)
    assert torch.allclose(model(S), bs_delta_path("butterfly", S, 0.132), atol=1e-6)   # zero-initialised head
    for p in model.head.parameters():                                       # make the correction non-trivial
        torch.nn.init.normal_(p, std=0.5)
    k = 30
    S2 = S.clone()
    S2[:, k + 1:] *= 1.3                                                    # change only the future
    assert torch.allclose(model(S)[:, :k + 1], model(S2)[:, :k + 1], atol=1e-6)


def test_features_use_the_latest_return_only_from_date_one():
    S = gbm_from_noise(torch.randn(3, N_STEPS), 0.2)
    f = build_features(S, "butterfly", 0.132)
    assert f.shape == (3, N_STEPS, 3) and torch.all(f[:, 0, 2] == 0)


def test_replication_is_nearly_exact_at_the_true_volatility():
    # with the true sigma the black-scholes hedge leaves a small spread; at a wrong sigma the spread is larger
    S = gbm_from_noise(torch.randn(20_000, N_STEPS, generator=make_generator(3)), 0.2)
    right = shortfall("butterfly", S, bs_delta_path("butterfly", S, 0.2)).std()
    wrong = shortfall("butterfly", S, bs_delta_path("butterfly", S, 0.1)).std()
    assert right < wrong
    assert abs(float(shortfall("butterfly", S, bs_delta_path("butterfly", S, 0.2)).mean())
               - float(payoff("butterfly", S[:, -1]).mean())) < 0.05       # zero-mean gains


def test_bsb_hedge_has_no_shortfall_above_its_price_for_constant_volatility():
    # capital = bsb price, any constant sigma in the band: the shortfall tail should be about zero (daily trading)
    S = gbm_from_noise(torch.randn(20_000, N_STEPS, generator=make_generator(4)), 0.20)
    x = shortfall("butterfly", S, bsb_delta_path("butterfly", S, CALIB.band))
    assert cvar(x) < bsb_price("butterfly", CALIB.band) + 0.5


def test_uniform_prior_has_a_heavier_upper_tail_and_stays_in_the_band():
    gen = make_generator(6)
    uni = sample_prior_sigma(100_000, gen, CALIB, prior="uniform")
    cal = sample_prior_sigma(100_000, gen, CALIB, prior="calibrated")
    assert float(uni.min()) >= 0.07 and float(uni.max()) <= 0.30
    assert float((uni > 0.25).float().mean()) > 5 * float((cal > 0.25).float().mean())
