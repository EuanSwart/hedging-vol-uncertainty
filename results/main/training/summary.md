# training-curve summary

smoothing window 5 logs (= 250 steps). start = mean of first 3 logs, end = mean of last 5 logs of the smoothed curve.

| net | payoff | params | start | end | drop % | steps to 50% | 90% | 99% | seconds |
|---|---|---|---|---|---|---|---|---|---|
| mlp | call | 1217 | 5.465 | 5.242 | 4.1 | 150 | 300 | 350 | 294 |
| gru | call | 1025 | 5.555 | 5.251 | 5.5 | 217 | 350 | 433 | 711 |
| mlp | butterfly | 1217 | 7.732 | 7.357 | 4.9 | 267 | 600 | 1450 | 272 |
| gru | butterfly | 1025 | 7.752 | 7.311 | 5.7 | 283 | 800 | 1200 | 530 |

## final value against the black-scholes reference (last 10 logs, per seed)

| net | payoff | final CVaR per seed | seed mean | BS ref (last batch, seed mean) | % vs BS |
|---|---|---|---|---|---|
| mlp | call | 5.226, 5.271, 5.242 | 5.246 | 5.719 | -8.3 |
| gru | call | 5.235, 5.275, 5.255 | 5.255 | 5.719 | -8.1 |
| mlp | butterfly | 7.366, 7.353, 7.339 | 7.353 | 7.834 | -6.1 |
| gru | butterfly | 7.313, 7.287, 7.307 | 7.302 | 7.834 | -6.8 |

## paired GRU - MLP (same seed, same noise stream)

| payoff | within-run noise sd (gru) | sd of paired difference | final diff per seed (last 10 logs) | mean |
|---|---|---|---|---|
| call | 0.112 | 0.034 | +0.009, +0.004, +0.013 | +0.009 |
| butterfly | 0.130 | 0.036 | -0.053, -0.066, -0.033 | -0.051 |
