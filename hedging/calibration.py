"""fit the volatility band and the prior to spy realised volatility of the calibration half (2010-2017)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from hedging.config import CALIBRATION_END, N_STEPS, SPY_CSV


def window_realised_vol(log_ret: np.ndarray, n: int = N_STEPS, stride: int | None = None) -> np.ndarray:
    """annualised realised volatility of consecutive n-day windows (non-overlapping by default)."""
    stride = stride or n
    starts = range(0, len(log_ret) - n + 1, stride)
    return np.array([math.sqrt(252.0 * np.mean(log_ret[i:i + n] ** 2)) for i in starts])


def band_from_quantiles(vol: np.ndarray) -> tuple:
    """5th percentile rounded down to 0.01 and 95th percentile rounded up to 0.05."""
    lo = math.floor(np.quantile(vol, 0.05) * 100) / 100
    hi = math.ceil(np.quantile(vol, 0.95) / 0.05) * 0.05
    return round(lo, 2), round(hi, 2)


def fit_inverse_gamma(vol: np.ndarray) -> tuple:
    """maximum-likelihood inverse gamma on the realised variance (location fixed at zero)."""
    a, _, b = stats.invgamma.fit(vol ** 2, floc=0)
    return float(a), float(b)


def prior_mean_sigma(a: float, b: float, band: tuple, n_grid: int = 2001) -> float:
    """mean of sigma under the inverse-gamma prior truncated to the band, by quadrature."""
    sigma = np.linspace(band[0], band[1], n_grid)
    density = stats.invgamma.pdf(sigma ** 2, a, scale=b) * 2 * sigma
    return float((sigma * density).sum() / density.sum())


def diagnostics(log_ret: np.ndarray, n: int = N_STEPS) -> dict:
    """the descriptive numbers quoted to justify the setting (persistence of volatility)."""
    vol = window_realised_vol(log_ret, n)
    half = n // 2
    first = np.array([math.sqrt(252 * np.mean(log_ret[i:i + half] ** 2)) for i in range(0, len(log_ret) - n + 1, n)])
    second = np.array([math.sqrt(252 * np.mean(log_ret[i + half:i + n] ** 2)) for i in range(0, len(log_ret) - n + 1, n)])
    return dict(
        n_quarters=len(vol),
        quantiles_5_25_50_75_95=[float(q) for q in np.quantile(vol, [0.05, 0.25, 0.5, 0.75, 0.95])],
        corr_log_vol_consecutive_quarters=float(np.corrcoef(np.log(vol[:-1]), np.log(vol[1:]))[0, 1]),
        corr_log_vol_first_vs_second_half=float(np.corrcoef(np.log(first), np.log(second))[0, 1]),
        median_abs_log_ratio_halves=float(np.median(np.abs(np.log(second / first)))),
    )


def calibrate(csv=SPY_CSV) -> dict:
    df = pd.read_csv(csv, parse_dates=["date"]).dropna().reset_index(drop=True)
    log_ret = np.diff(np.log(df["spy"].to_numpy(float)))
    ret_dates = df["date"].iloc[1:].reset_index(drop=True)
    cal = log_ret[(ret_dates <= CALIBRATION_END).to_numpy()]         # calibration half only
    vol = window_realised_vol(cal)
    band = band_from_quantiles(vol)
    a0, b0 = fit_inverse_gamma(vol)
    return dict(
        band=list(band), a0=a0, b0=b0, sigma_ref=prior_mean_sigma(a0, b0, band),
        calibration_period=f"{df['date'].iloc[0].date()} to {CALIBRATION_END}",
        diagnostics_calibration_half=diagnostics(cal),
        diagnostics_full_sample=diagnostics(log_ret),
    )
