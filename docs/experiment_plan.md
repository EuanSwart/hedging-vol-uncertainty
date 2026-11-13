# Experiment plan v4 (laptop scale, with real data)

Minimal revision of `experiment_plan_v3.md` (kept unchanged) after the review questions of 2026-11-13. The experiment, the hedgers and the compute budget stay as in v3. Section 1 lists exactly what changed and why. Section 3 lists every simplification and admits its cost.

## 1. What changed from v3, and why

| # | Change | Reason |
|---|---|---|
| 1 | **Real SPY data added** as an out-of-sample test (Section 6), and the volatility band and prior are **calibrated to SPY** (Section 4) instead of chosen by hand | The experiment must be empirically justified. |
| 2 | **Capital and shortfall defined explicitly** (Section 5): zero starting cash, metric = CVaR95 of the shortfall = the *required premium* in dollars | v3 left this implicit (the review's "critical" point). |
| 3 | **"Bayes plug-in" renamed "posterior-mean hedge"**; it is a reference, not a Bayes-optimal benchmark; "distance to Bayes" removed | It ignores posterior spread and the value of future learning. The exact Bayes-optimal hedge is not computed. |
| 4 | **MLP receives the same instantaneous inputs as the GRU**: (log-moneyness, time to maturity, latest return r_k) | Otherwise the MLP vs GRU difference mixes the latest return with memory. (The old report text said memory was the only difference, but the v1 MLP lacked r_k.) |
| 5 | **BSB is stated as solving a different (more cautious) problem** and is also tested on real data, where volatility actually varies inside the quarter | Fairness to BSB (the review's point on rectangular vs fixed volatility). |
| 6 | **Go/no-go noise check before any training** (Step 3) | Make sure the effect can be measured at all. |

Everything else (hedgers A to D, 12 trainings, about 2 CPU-hours, call as control) is unchanged.

## 2. The question

A bank has sold an option and does not know the volatility of the underlying. One answer is to guard against the worst volatility in an uncertainty band (the Black–Scholes–Barenblatt hedge, Section 5.2 of the report). Another is to learn the volatility from the price path (the Bayes-adaptive idea of Section 3, which a recurrent network trained across a prior learns to approximate, Section 4). **When does learning beat guarding, and does the result survive on real SPY data?**

This is the gap named in Section 5.4 of the report: no reviewed paper compares a learning hedger with the robust hedge for an unknown volatility.

## 3. Simplifications, admitted

1. **Volatility is fixed within each simulated quarter.** This is an idealisation, not a description of markets. It is used because it is the hidden-constant case of Section 3, it makes the posterior depend only on (k, realised variance), and it isolates the question "does inferring the volatility from the path help?". Evidence on how wrong it is (SPY, 63 quarters, Section 4): the realised volatility of the first and second half of a quarter correlate at 0.58, and the median relative change between halves is about 22%. So volatility is persistent but not constant. The real-data test (Section 6) is how the report checks that conclusions survive this.
2. **No exact Bayes-optimal hedge.** The posterior-mean hedge (Black–Scholes delta at the posterior-mean volatility) is a certainty-equivalent rule. The GRU may beat it, and that would not be a contradiction. The exact benchmark (a dynamic programme, or the quadratic-loss formula on the state (k, S, realised variance)) is future work.
3. **No stochastic-volatility simulator and no adversarial "worst-case path" world.** Real SPY paths play the role of a world that follows none of the hedgers' assumptions. The cost: no controlled test of the world BSB is designed for.
4. **The metric is CVaR95 of the uncentred shortfall, not mean-squared error.** Centring would remove the premium term that the worst-case comparison and the call control depend on. MSE-type numbers (standard deviation of the shortfall) are reported as a secondary check, and are the main real-data metric because 31 to 93 windows are too few for a 5% tail.
5. **BSB is scored by a criterion it was not designed for.** It guarantees no loss at its own price against every volatility path in the band, not minimal tail risk. We report that guaranteed price next to its measured required premium, so its conservatism is visible and not hidden inside a loss.
6. **No transaction costs, no dividends modelled, no drift correction.** The report's shortfall identity (eq. shortfall-identity) shows drift does not enter the continuous-time hedging error, so real-data drift (about 13% a year in the sample) is left in; discrete-time effects are assumed second-order and checked once in Step 5.
7. **Real-data test has low power**: SPY 2018 to 2025 gives 31 non-overlapping quarters or 93 overlapping windows (stride 21 days). It is a consistency check with confidence intervals, not a proof.

## 4. Empirical and literature justification of each design choice

| Choice | Evidence |
|---|---|
| Unknown volatility is what drives hedging error | El Karoui et al. (1998), eq. shortfall-identity of the report: error depends on realised minus assumed variance times Gamma |
| Band and prior from SPY | Realised volatility of 63-day windows, 2010–2017 (the calibration half): 5th, 25th, 50th, 75th, 95th percentiles 0.073, 0.096, 0.127, 0.144, 0.254, mean 0.136. **Band [0.07, 0.30]** (covers 90% of calibration quarters, 97% of 2018–2025 quarters). **Prior:** inverse-gamma on σ² fitted to the 31 calibration-half quarters, truncated to the band (replaces the arbitrary IG(3, 0.10)) |
| Learning from the path is worthwhile | Volatility clusters and is persistent (Cont, 2001, already cited); on SPY, first-half vs second-half quarterly volatility correlation 0.58 |
| Limits of that claim | Correlation of log realised volatility between *consecutive* quarters is only 0.26 (62 pairs), so the hedger starts each window from the prior and does not carry volatility across windows |
| CVaR training loss, pathwise gradients | Buehler et al. (2019) |
| Squared-error numbers as secondary metric | Used as the main loss by Lütkebohmert et al. (2022) and Abi Jaber and Gérard (2025) |
| Recurrent hedger, memoryless baseline | Duan et al. (2016) for RL²; Tobin et al. (2017) for domain randomisation |
| Real-data test on an index | Precedent: Lütkebohmert et al. test on S&P 500 constituents through March 2020 |
| Market-practice baseline: delta hedge at implied volatility | First classical answer in Section 5.2; VIX (30-day implied) at the start of each window, a stated approximation for a 63-day horizon |

The statistics above come from a quick script on `experiment/v1/data/spy_vix.csv` (SPY and VIX daily closes, 2010-01-04 to 2025-12-31). `calibrate.py` in the new code will reproduce them, and the band and prior are fixed from the calibration half only (2010–2017), so the 2018–2025 test half is never used for tuning.

## 5. Setup and definitions

- **Market (simulation):** geometric Brownian motion, S0 = 100, zero drift and rate, T = 0.25, n = 63 daily rebalancing dates. Volatility σ drawn once per path from the calibrated prior on [0.07, 0.30].
- **Payoffs:** butterfly 90/100/110 (main), call K = 100 (control: the report predicts that for a convex payoff the worst-case gap between BSB and a learner vanishes).
- **Shortfall and capital (explicit):** a hedge is a sequence Δ_k, a function of information up to t_k only (price S_k, latest return r_k = log(S_k/S_{k-1}), and for recurrent nets earlier returns). The seller starts with **zero cash**, self-financing, no costs:

  X = g(S_n) − Σ_{k=0}^{n-1} Δ_k (S_{k+1} − S_k).

  **Required premium** = CVaR95(X) = min_w { w + E[(X − w)⁺]/0.05 }, in dollars: the cash that makes the tail risk of the seller zero (the risk-measure price of Buehler et al.). Every hedger is scored from the same zero start, so no hedger gets more capital. With zero drift, E[Σ Δ_k ΔS_k] = 0 for every hedge, so any common starting capital shifts all hedgers equally and rankings do not depend on it. This replaces v3's W0 = Black–Scholes price at 0.20, which had the same ranking but no interpretable level.
- **BSB's own price** u^B(0, S0) (finite-difference table, recomputed for the new band) is reported as a separate reference: it is the capital at which BSB guarantees no loss in continuous time. At any true σ in the band, BSB's required premium should not exceed it (up to discretisation); the difference is the premium BSB over-charges for its guarantee.
- **Losses reported:** (i) *prior loss*: CVaR95 on paths with σ drawn from the prior; (ii) *worst-case loss*: the largest CVaR95 over a grid of constant σ in the band (9 points); (iii) secondary: standard deviation of X.

## 6. Hedgers and tests

| # | Hedger | Type | Sees at step k | Compute |
|---|---|---|---|---|
| A | **BSB hedge**, delta of the uncertain-volatility PDE on the band | traditional, numerical, robust, no learning | k, S_k | seconds |
| B | **GRU meta-hedger**, deep hedging trained across the prior (CVaR95, pathwise) | deep / meta-RL | k, S_k, r_k, hidden state | about 8 min per run |
| C | **Posterior-mean hedge**: Black–Scholes delta at the posterior-mean volatility (closed-form inverse-gamma update) | traditional, adaptive, certainty-equivalent, **not Bayes-optimal** | k, S_k, realised variance | seconds |
| D | **MLP**, same training and prior as B, **inputs (log-moneyness, time to maturity, r_k)** | deep, memoryless ablation | k, S_k, r_k | about 8 min per run |

References (free): Black–Scholes delta at the prior-mean volatility (simulation) or at VIX/100 at the window start (real data); oracle delta at the true σ (simulation only). Parameter counts of B and D are reported; if the MLP has far more parameters than the GRU, shrink it to 2×32.

**Test 1, simulation (controlled).** Fixed-σ paths, common random numbers for all hedgers, 100k prior paths and 9 σ grid points with 50k paths each. Networks: 3 seeds.

**Test 2, real SPY data (out of sample, no retraining).** Windows of 63 consecutive daily closes from 2018–2025, stride 21 days (93 windows, overlapping), each rebased to S0 = 100 so the strikes are 90/100/110. Hedgers A to D as trained in Test 1 (band and prior from 2010–2017), plus the VIX-delta baseline. Metrics: standard deviation and mean of X, and CVaR95 for completeness (low power, flagged). Uncertainty by a moving-block bootstrap (blocks of 3 windows, about one non-overlapping quarter), and paired differences against BSB. Also reported: the share of test windows whose realised volatility falls outside the band, and the same table for the top third of windows by realised volatility (stress). Exploratory: no directional claim beyond the hypotheses below.

## 7. Hypotheses (rules fixed before running)

- **H1 (simulation, learning pays when Gamma changes sign).** On the butterfly the GRU's worst-case and prior CVaR95 are below BSB's by more than two seed standard errors. On the call the worst-case difference is within two standard errors of zero.
- **H2 (memory and learning).** The GRU is no worse than the MLP (memory helps), and no worse than the posterior-mean hedge by more than two standard errors. If the GRU beats the posterior-mean hedge, we report it as evidence that the certainty-equivalent rule is not optimal. Optional probe: linear regression of the GRU hidden state on the posterior-mean volatility, scored by held-out R² against an untrained GRU.
- **H3 (real data, exploratory).** On SPY 2018–2025 the adaptive hedgers (B, C) have a lower standard deviation of X than BSB and the VIX-delta baseline, or else we report where and why not (quarters outside the band, the 2020 crash). A null result is reported as such.

Honest-failure clause: v1 quick-scale runs had the GRU worse than BSB in the worst case, probably from undertraining. A negative H1 after proper training is reported as evidence against H1.

## 8. Compute (unchanged from v3, measured on this laptop on 2026-11-13)

Benchmark of the v1 code, one training step: GRU16 batch 4096 0.19 s on one thread, batch 8192 0.37 s; MLP64 batch 8192 0.36 s (machine busy, 2.3 GB RAM free).

| Block | Runs | CPU time |
|---|---|---|
| GRU16 × {butterfly, call} × 3 seeds (batch 4096, 2500 steps) | 6 | about 0.8 h |
| MLP × {butterfly, call} × 3 seeds | 6 | about 0.8 h |
| Calibration, BSB table for the new band, simulation tests | | about 15 min |
| Real-data evaluation (93 windows × 4 hedgers) | | seconds |
| **Total** | **12 trainings** | **about 2 h CPU, about 1 h wall with 2 workers** |

Fallback if the pilot shows the butterfly loss still falling: batch 8192, 4000 steps for the butterfly GRU only (about 25 min per run). Use at most 2 or 3 single-thread workers and close other applications (free RAM was the limit in v1). No GPU.

## 9. Code

Reuse `experiment/v1/` (copy to `experiment/v3/`, then prune), as in the v3 plan, with these changes:
- `calibrate.py` (new): quarterly realised volatility from `spy_vix.csv` on 2010–2017, band percentiles, inverse-gamma fit; writes `config` values.
- `baselines.py`: posterior-mean hedge as the delta *at* the posterior-mean volatility (v1 averaged deltas); `bs_delta` accepts sigma of shape (N, n); BSB table recomputed for the new band.
- `hedgers.py`: MLP features (log-moneyness, τ, r_k).
- `train.py`: shortfall with zero starting cash (W0 = 0).
- `real_data.py` (new): windows, rebasing, VIX baseline, block bootstrap.
- `probe.py`: fix the v1 bug (regressing on k within fixed k). Drop GRU64, StatMLP, regime switching, HMM, HSG, LFP.
- `pfhedge` only as a one-off Black–Scholes cross-check in a separate environment (it cannot host a per-path volatility or a recurrent hedger).

## 10. Displays for Section 6 of the report

1. **Diagram:** the hedgers on one path with what each sees (replaces the hierarchy figure and shortens the setup).
2. **Figure:** CVaR95 against the true σ, two panels (butterfly, call), one line per hedger; max of each line = worst case.
3. **Figure:** hedge ratio over time on a low-volatility and a high-volatility butterfly path for BSB, GRU, oracle (shows the GRU adapting).
4. **Table (simulation):** prior and worst-case required premium per hedger and payoff, mean ± s.e., with BSB's guaranteed price alongside.
5. **Table or figure (real data):** standard deviation of X per hedger on SPY 2018–2025, with bootstrap intervals and the high-volatility subset.

## 11. Steps

1. Copy v1, prune, make the code changes above, run unit tests (half a day).
2. `calibrate.py`: fix the band and prior from 2010–2017; recompute the BSB table; cross-check Black–Scholes against pfhedge.
3. **Go/no-go:** evaluate the free hedgers (A, C, references) on 200k+ common-random-number paths with bootstrap intervals. If the butterfly gap between A and C is not resolvable, stop and rethink before training anything.
4. Pilot (1 seed, butterfly, GRU and MLP) to set steps and batch.
5. Core runs (12 trainings), simulation tests, real-data test; one sensitivity check of the drift and discrete-trading assumption.
6. Write Section 6 in the next report version (v2.4).

## 12. Report changes this implies (for v2.4, not done)

- The new band [0.07, 0.30] changes the BSB numbers and `fig:uvm` in Section 5.2 (6.67 vs 6.10 were for [0.10, 0.30]), `tab:exp-params`, and the predictions table.
- Replace "Bayes proxy" by "posterior-mean hedge" everywhere; correct the sentence that memory is the only difference between MLP and GRU (true only after change 4).
- Add the real-data paragraph, the capital definition, and a short admitted-simplifications paragraph to Section 6, with about half a page extra.
- Option to avoid the ripple: keep the band [0.10, 0.30] and admit that 25% of the 2010–2025 quarters lie below 0.105. I recommend recalibration because the user asked for empirical justification.

## 13. Risks

| Risk | Handling |
|---|---|
| Effect smaller than noise (in v1 quick runs the posterior-mean rule beat BSB by only about 4% in the butterfly worst case) | go/no-go step; common random numbers; paired bootstrap; more test paths |
| GRU does not beat BSB | longer pilot; report both losses; report as evidence against H1 if it persists |
| Real data: too few windows, crash windows outside the band | block bootstrap, report share outside the band, treat as exploratory |
| Posterior-mean hedge below or above the GRU is not informative about Bayes-optimality | stated in Section 3; no claim about Bayes-optimality |
| RAM pressure | 2 to 3 workers, close other applications |

## 14. References needed by Section 6

`buehler2019deep`, `avellaneda1995pricing`, `duan2016rl2`, `tobin2017domain`, `lutkebohmert2022robust` (real-data precedent), `elkaroui1998robustness`, `cont2001empirical` (persistence, already in the bibliography), `pfhedge` (cross-check). SPY and VIX series: Yahoo Finance via `yfinance`, as in v1's `download_data.py`.
