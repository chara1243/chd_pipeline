"""Standardized Precipitation Index (SPI-1 / SPI-3) with gamma fitting.

Sensitivity-analysis variant of the 30-day percentile dryness definition
(concept note section 6.2). SPI is computed per calendar day using the same
+/-15-day moving climatological window as the main definition, so results are
directly comparable.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gammadist
from scipy.stats import norm

from core.events import dayofyear_calendar


def _gamma_params(x: np.ndarray) -> tuple:
    """Method-of-moments gamma fit for positive values (Thom 1958).

    A = ln(mean) - mean(ln x)  >= 0 satisfies  A ~ 1/(2a) + 1/(12a^2),
    giving the positive root  a = (3 + sqrt(9 + 12A)) / (12A).
    """
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 20:
        return np.nan, np.nan
    m = x.mean()
    lm = np.log(x).mean()
    A = np.log(m) - lm
    if A <= 1e-12:
        return np.nan, np.nan
    alpha = (3 + np.sqrt(9 + 12 * A)) / (12 * A)
    beta = m / alpha
    return alpha, beta


def spi_series(values: np.ndarray, dates: np.ndarray,
               window: int = 15) -> np.ndarray:
    """SPI for a daily series (typically rolling 30-day sums).

    For each calendar day (doy 1..365, leap days folded), a gamma distribution
    is fitted to the base-period values within +/-window days; the CDF is
    evaluated and mapped to the standard normal. Zero-precipitation days use
    the mixed-distribution adjustment of Chen et al. style SPI (P0 handling).
    """
    doy = dayofyear_calendar(dates)
    out = np.full(values.shape, np.nan)
    for d in range(1, 366):
        idxs = [(d - 1 + k) % 365 for k in range(-window, window + 1)]
        sel = np.isin(doy, [i + 1 for i in idxs])
        pool = values[sel]
        p0 = np.mean(pool <= 0) if np.isfinite(pool).any() else 0.0
        alpha, beta = _gamma_params(pool)
        if not (np.isfinite(alpha) and np.isfinite(beta) and beta > 0):
            continue
        x = values.copy()
        x[x <= 0] = np.nan
        # CDF of gamma at the (positive) values, mixed adjustment for zeros
        cdf = gammadist.cdf(x, a=alpha, scale=beta) * (1 - p0)
        res = norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
        # dry days (value <= 0) mapped to the lower half of the dry-prob mass
        z = (values <= 0) & np.isfinite(values)
        res[z] = norm.ppf(np.clip(p0 / 2, 1e-6, 0.5))
        out[doy == d] = res[doy == d]
    return out
