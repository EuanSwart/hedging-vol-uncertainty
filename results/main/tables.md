### simulation, butterfly (bsb guaranteed price 7.568)

| hedger | prior cvar95 | worst-case cvar95 | worst sigma |
|---|---|---|---|
| bs_ref | 7.808 | 9.225 | 0.300 |
| bsb | 7.484 | 8.169 | 0.070 |
| posterior_mean | 7.308 | 8.001 | 0.070 |
| oracle | 7.254 | 7.874 | 0.070 |
| gru_s0 | 7.319 | 8.065 | 0.070 |
| gru_s1 | 7.285 | 8.033 | 0.070 |
| gru_s2 | 7.302 | 8.049 | 0.070 |
| mlp_s0 | 7.377 | 8.527 | 0.300 |
| mlp_s1 | 7.352 | 8.263 | 0.300 |
| mlp_s2 | 7.342 | 8.049 | 0.070 |

| difference (cvar95) | mean | s.e. over seeds |
|---|---|---|
| value_of_learning_prior_cvar | +0.182 | 0.010 |
| gru_minus_posterior_mean_prior_cvar | -0.005 | 0.010 |
| value_of_memory_prior_cvar | +0.055 | 0.008 |
| value_of_learning_worst_cvar | +0.120 | 0.009 |
| gru_minus_posterior_mean_worst_cvar | +0.048 | 0.009 |
| value_of_memory_worst_cvar | +0.231 | 0.133 |

### simulation, call (bsb guaranteed price 5.979)

| hedger | prior cvar95 | worst-case cvar95 | worst sigma |
|---|---|---|---|
| bs_ref | 5.695 | 10.442 | 0.300 |
| bsb | 5.379 | 7.454 | 0.300 |
| posterior_mean | 5.288 | 7.755 | 0.300 |
| oracle | 5.249 | 7.456 | 0.300 |
| gru_s0 | 5.253 | 8.667 | 0.300 |
| gru_s1 | 5.248 | 8.666 | 0.300 |
| gru_s2 | 5.254 | 8.756 | 0.300 |
| mlp_s0 | 5.249 | 8.466 | 0.300 |
| mlp_s1 | 5.246 | 8.423 | 0.300 |
| mlp_s2 | 5.244 | 8.445 | 0.300 |

| difference (cvar95) | mean | s.e. over seeds |
|---|---|---|
| value_of_learning_prior_cvar | +0.127 | 0.002 |
| gru_minus_posterior_mean_prior_cvar | -0.036 | 0.002 |
| value_of_memory_prior_cvar | -0.005 | 0.002 |
| value_of_learning_worst_cvar | -1.243 | 0.030 |
| gru_minus_posterior_mean_worst_cvar | +0.941 | 0.030 |
| value_of_memory_worst_cvar | -0.252 | 0.032 |

### real data, butterfly (93 windows, 5% outside the band)

| hedger | std | 95% ci | std minus bsb | 95% ci | top-third std |
|---|---|---|---|---|---|
| bs_ref | 2.171 | [1.299, 3.018] | -0.109 | [-0.360, +0.172] | 3.567 |
| bs_vix | 2.056 | [1.316, 2.800] | -0.225 | [-0.544, +0.127] | 3.070 |
| bsb | 2.281 | [1.346, 3.268] | +0.000 | [+0.000, +0.000] | 3.688 |
| posterior_mean | 1.750 | [1.212, 2.364] | -0.531 | [-0.978, -0.026] | 2.427 |
| gru | 2.111 | [1.388, 2.745] | -0.170 | [-0.656, +0.366] | 3.184 |
| mlp | 2.220 | [1.420, 3.057] | -0.061 | [-0.303, +0.227] | 3.440 |

### real data, call (93 windows, 5% outside the band)

| hedger | std | 95% ci | std minus bsb | 95% ci | top-third std |
|---|---|---|---|---|---|
| bs_ref | 1.267 | [0.888, 1.607] | -0.078 | [-0.420, +0.224] | 1.388 |
| bs_vix | 1.297 | [0.839, 1.658] | -0.048 | [-0.260, +0.124] | 1.457 |
| bsb | 1.345 | [0.914, 1.723] | +0.000 | [+0.000, +0.000] | 1.451 |
| posterior_mean | 1.384 | [0.900, 1.785] | +0.039 | [-0.099, +0.175] | 1.455 |
| gru | 1.188 | [0.813, 1.504] | -0.157 | [-0.403, +0.065] | 1.186 |
| mlp | 1.289 | [0.839, 1.664] | -0.057 | [-0.166, +0.067] | 1.369 |
