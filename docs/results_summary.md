# results summary (scale `main`: batch 4096, 2500 steps, seeds 0 to 2)

all numbers are in `results/main/` (`sim_results.json`, `real_results.json`, `probe_results.json`, `tables.md`, figures,
console logs). metric: required premium = cvar95 of the shortfall from zero starting cash, in dollars. prior loss uses
100k paths with sigma from the prior; worst case is the largest cvar95 over 9 constant sigma in [0.07, 0.30], 50k paths each.
"+/-" is the standard error over 3 seeds. test-sampling error (paired bootstrap over paths) is about 0.01 to 0.02 for the prior loss.

## simulation, butterfly (main case; bsb guaranteed price 7.568)

| hedger | prior | worst case |
|---|---|---|
| bsb | 7.484 | 8.169 |
| posterior mean | 7.308 | 8.001 |
| gru (mean of 3 seeds) | 7.302 | 8.049 |
| mlp (mean of 3 seeds) | 7.357 | 8.280 |
| oracle | 7.254 | 7.874 |

- value of learning (bsb minus gru): +0.182 +/- 0.010 prior (2.4%), +0.120 +/- 0.009 worst case (1.5%). h1 for the butterfly holds under the preset rule.
- gru versus posterior-mean hedge: -0.005 +/- 0.010 prior (indistinguishable), +0.048 +/- 0.009 worst case (gru slightly worse). h2 holds on the prior loss.
- value of memory (mlp minus gru): +0.055 +/- 0.008 prior; +0.231 +/- 0.133 worst case (one mlp seed is much worse, so the worst-case figure is noisy).
- probe: the gru hidden state predicts the posterior-mean volatility with held-out r^2 of 0.74 to 0.80 against 0.50 to 0.63 for an untrained gru (margin 0.15, 0.30, 0.16 for the three seeds; mean 0.20, at the preset threshold).

## simulation, call (control; bsb guaranteed price 5.979)

- prior loss: gru 5.251, mlp 5.247, posterior mean 5.288, bsb 5.379, oracle 5.249; learned hedgers beat bsb by 0.127 +/- 0.002. memory adds nothing (mlp minus gru: -0.005 +/- 0.002).
- worst case: bsb 7.454, oracle 7.456, posterior mean 7.755, **gru 8.70, mlp 8.44** (all at sigma = 0.30).
- the preset rule "gru and bsb within 2 standard errors of zero in the worst case" **fails**, in the direction opposite to a learning gain: the networks are worse than bsb at the top of the band. nobody beats bsb there (the oracle only ties it), as the theory predicts, but the networks give up protection at sigma near 0.30 because they were trained on the prior loss, and the calibrated prior puts little mass there (probability of sigma above 0.25 is 2.7%, above 0.28 is 0.8%). this is the prior-versus-worst-case trade-off, not a coding error; training on a prior that is heavier in the tail (or on a worst-case criterion) would change it.
- probe on the call: trained gru r^2 0.40 to 0.49 is not above the untrained control (0.50 to 0.63): the gru does not need to encode sigma for the call.

## real spy data, 2018-2025 (93 overlapping windows, no retraining; 5% of windows outside the band)

standard deviation of the shortfall on butterfly windows (95% moving-block bootstrap interval of the difference to bsb):

| hedger | std | minus bsb |
|---|---|---|
| bsb | 2.281 | 0 |
| bs at vix/100 | 2.056 | -0.225 [-0.544, +0.127] |
| bs at prior mean | 2.171 | -0.109 [-0.360, +0.172] |
| mlp | 2.220 | -0.061 [-0.303, +0.227] |
| gru | 2.111 | -0.170 [-0.656, +0.366] |
| posterior mean | 1.750 | **-0.531 [-0.978, -0.026]** |

- h3 is only partly supported. the posterior-mean hedge beats bsb with an interval that excludes zero (barely); the gru has the right sign but its interval includes zero. the posterior-mean hedge also beats the gru out of sample (1.75 against 2.11; top-third volatility windows 2.43 against 3.18), so the trained network is less robust to real returns than the simple rule.
- with 93 overlapping windows this test is low-powered; call windows show no separation at all (all intervals contain zero).

## cautions

- effect sizes on the butterfly are small (1.5% to 2.4% of the premium); the oracle bounds the possible gain in the worst case at about 3.6%.
- the worst-case sigma on the butterfly is the bottom of the band (0.07) for bsb, the posterior-mean hedge and the gru; the bsb measured premium there (8.17) exceeds its guaranteed price (7.57) because of daily trading.
- the posterior-mean hedge and the oracle are not cvar-optimal in discrete time, so learned hedgers can come out below them.
- the checks above were reviewed after the run; no inconsistency pointing to an implementation error was found (martingale, no look-ahead, bsb price, oracle ordering are covered by unit tests or by the go/no-go step).

## addendum: where the gru stops, and the heavier-tailed prior for the call

- profile over true sigma (butterfly): the gru is within 0.2 of the oracle for sigma <= 0.10 but 0.8 above it at sigma = 0.185 and 3.0 above it at sigma = 0.30, where it behaves almost like bsb; the posterior-mean hedge stays within 0.3 of the oracle throughout. the prior puts 9% of its mass above sigma = 0.20 and 2.7% above 0.25. "the gru matches the posterior-mean hedge" therefore holds for the prior loss only. the scalar worst case hides this because the butterfly's worst volatility is the bottom of the band (0.07) for most hedgers.
- call, networks retrained on a uniform prior over the band (22% of mass above sigma = 0.25), 3 seeds, `run_prior_ablation.py`:

| hedger | calibrated-prior loss | uniform-prior loss | worst case |
|---|---|---|---|
| bsb | 5.38 | 6.35 | 7.45 |
| oracle | 5.25 | 6.38 | 7.46 |
| gru, calibrated prior | 5.251 +/- 0.002 | 6.775 +/- 0.011 | 8.696 +/- 0.030 |
| gru, uniform prior | 5.433 +/- 0.014 | 6.382 +/- 0.008 | 7.747 +/- 0.010 |
| mlp, calibrated prior | 5.247 +/- 0.001 | 6.687 +/- 0.005 | 8.445 +/- 0.012 |
| mlp, uniform prior | 5.368 +/- 0.001 | 6.373 +/- 0.004 | 7.582 +/- 0.009 |

  the call's worst-case failure was a training-distribution effect: with the heavier tail the worst case drops by about 10% (to within 0.1 to 0.3 of the bsb floor, which nothing beats) at a cost of 2 to 3% on the calibrated prior. the mlp is as good as the gru here, so memory still buys nothing on the call.
