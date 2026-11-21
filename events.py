"""Compound hot-dry event detection on daily climate cubes (numpy/xarray).

This module is the scientific heart of the pipeline. It operates on ordinary
numpy arrays so it can be unit-tested offline and run identically in Colab
or locally. Grid layout of every daily array: (time, lat, lon).

Design decisions (documented in the manuscript Methods):
* Heat day:  Tmax(d) > P90 of the same calendar day +/-15 d, 1991-2020.
* Dry day:   P30(d) < P20 of P30 on the same calendar day +/-15 d, 1991-2020.
* Episode:   >= MIN_EPISODE_DAYS consecutive compound days; assigned to the
             year in which it starts (cross-year episodes counted once, in the
             start year; documented rather than apportioned).
* Severity:  Se = sum over episode days of max(0, ZT) + max(0, -ZP), with
             ZT/ZP standardised against the same 1991-2020 climatology.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# class codes
NORMAL, HOT_ONLY, DRY_ONLY, COMPOUND = 0, 1, 2, 3

CLASS_NAMES = {
    NORMAL: "normal",
    HOT_ONLY: "heat only",
    DRY_ONLY: "dry only",
    COMPOUND: "compound hot-dry",
}


# ---------------------------------------------------------------------------
# Climatology helpers
# ---------------------------------------------------------------------------
def dayofyear_calendar(dates: np.ndarray) -> np.ndarray:
    """Calendar day-of-year for a sequence of numpy datetime64 days.

    Leap days (Feb 29) are folded onto Feb 28: doy 366 -> 60. All non-leap
    days keep their standard doy. This gives a stable 365-day climatology.
    """
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    days = (dates - dates.astype("datetime64[M]")).astype(int) + 1
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    doy = cum[months - 1] + days
    # fold Feb 29 (month==2, day==29) onto the Feb 28 slot (doy 59)
    leap = (months == 2) & (days == 29)
    doy[leap] = 59
    return doy


def calendar_thresholds(values: np.ndarray, dates: np.ndarray,
                        percentile: float, window: int = 15) -> np.ndarray:
    """Calendar-day percentile thresholds with a +/-window-day moving window.

    values : (time,) or (time, ny, nx) daily series for the BASE period only.
    Returns thresholds shaped (365,) or (365, ny, nx).
    """
    doy = dayofyear_calendar(dates)
    per_doy = [values[doy == d] for d in range(1, 366)]
    out = []
    for d in range(1, 366):
        # +/- window with wrap-around at the year edges
        idxs = [(d - 1 + k) % 365 for k in range(-window, window + 1)]
        pool = np.concatenate([per_doy[i] for i in idxs], axis=0)
        out.append(np.nanpercentile(pool, percentile, axis=0))
    return np.stack(out)


def daily_threshold_series(dates: np.ndarray,
                           thresholds: np.ndarray) -> np.ndarray:
    """Expand (365, ...) calendar thresholds onto the actual date vector."""
    doy = dayofyear_calendar(dates)
    return thresholds[doy - 1]


def rolling_sum(values: np.ndarray, days: int = 30,
                min_valid_frac: float = 0.9) -> np.ndarray:
    """Rolling `days`-day precipitation sum along axis 0 (continuous time axis,
    so windows naturally cross year boundaries).

    The first `days-1` positions of the series have incomplete windows and are
    returned as NaN (this is how the first 29 days of 1981 are handled). Windows
    containing more than (1-min_valid_frac) missing days are also NaN, so
    isolated missing observations are tolerated but data gaps are not silently
    filled with zero.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)

    def _cum(a):
        if a.ndim == 1:
            c = np.concatenate([[0.0], np.cumsum(a)])
            return c
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)], 0)
        return c

    def _win_sum(a):
        c = _cum(a)
        out = np.full(a.shape, np.nan)
        n = a.shape[0]
        # full-window sums: out[i] = cum[i+1] - cum[i-days+1], i >= days-1
        if n >= days:
            if a.ndim == 1:
                out[days - 1:] = c[days:] - c[:n - days + 1]
            else:
                out[days - 1:] = c[days:] - c[:n - days + 1]
        return out

    s = _win_sum(filled)
    nv = _win_sum(valid)
    n_req = days * min_valid_frac
    s[nv < n_req] = np.nan
    return s


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_days(tmax: np.ndarray, p30: np.ndarray,
                  tmax_thr: np.ndarray, p30_thr: np.ndarray) -> np.ndarray:
    """Four-way classification per day/pixel -> codes 0..3."""
    hot = tmax > tmax_thr
    dry = p30 < p30_thr
    valid = np.isfinite(tmax) & np.isfinite(p30) & np.isfinite(tmax_thr) \
        & np.isfinite(p30_thr)
    cls = np.where(hot & dry, COMPOUND,
                   np.where(hot, HOT_ONLY, np.where(dry, DRY_ONLY, NORMAL)))
    cls = np.where(valid, cls, np.nan)
    return cls


# ---------------------------------------------------------------------------
# Episode detection (run-length)
# ---------------------------------------------------------------------------
def episodes_1d(compound_flag: np.ndarray, min_days: int = 3):
    """Run-length episode detection for one pixel-year.

    compound_flag : 1-D boolean (a year of days, may cross year boundary via
        the caller). Returns list of (start_idx, length) for runs >= min_days.
    """
    eps = []
    run = 0
    start = None
    n = len(compound_flag)
    for i, v in enumerate(compound_flag):
        if v:
            if run == 0:
                start = i
            run += 1
        if (not v) or (i == n - 1):
            if run >= min_days:
                eps.append((start, run))
            run = 0
    return eps


def annual_metrics(compound_flag: np.ndarray, zt: np.ndarray = None,
                   zp: np.ndarray = None, min_days: int = 3) -> Dict:
    """Per pixel-year summary metrics from a boolean compound-day flag.

    compound_flag : (365/366,) bool
    zt, zp        : standardised anomalies of Tmax and P30 (same length)

    Returns dict: n_episodes, episode_days, mean_duration, max_duration,
    severity, n_compound_days. Days belonging to <min_days runs are NOT
    counted as event days (distinguishes episodes from isolated compound
    days, which are reported separately through n_compound_days).
    """
    eps = episodes_1d(compound_flag, min_days)
    n_eps = len(eps)
    e_days = sum(L for _, L in eps)
    dur = [L for _, L in eps]
    sev = 0.0
    if zt is not None and zp is not None:
        for s, L in eps:
            a = zt[s:s + L]
            b = zp[s:s + L]
            sev += float(np.nansum(np.maximum(0, a)) +
                         np.nansum(np.maximum(0, -b)))
    return {
        "n_episodes": n_eps,
        "episode_days": e_days,
        "n_compound_days": int(np.nansum(compound_flag)),
        "mean_duration": float(np.mean(dur)) if dur else 0.0,
        "max_duration": int(max(dur)) if dur else 0,
        "severity": sev,
    }


# ---------------------------------------------------------------------------
# Seasonal tagging
# ---------------------------------------------------------------------------
def season_of_month(month: int, seasons: Dict[str, list]) -> str:
    for name, months in seasons.items():
        if month in months:
            return name
    return "unknown"


def seasonal_counts(compound_flag: np.ndarray, months: np.ndarray,
                    seasons: Dict[str, list]) -> Dict[str, int]:
    """Compound-day counts per season for one pixel-year."""
    out = {k: 0 for k in seasons}
    for name, ms in seasons.items():
        m = np.isin(months, ms)
        out[name] = int(np.nansum(compound_flag[m]))
    return out


# ---------------------------------------------------------------------------
# Convenience: full pipeline for one pixel (also used in unit tests)
# ---------------------------------------------------------------------------
def pixel_pipeline(tmax: np.ndarray, precip: np.ndarray, dates: np.ndarray,
                   years: Tuple[int, int], base_years: Tuple[int, int],
                   heat_pct: float = 90, dry_pct: float = 20,
                   window: int = 15, min_days: int = 3) -> Dict[int, Dict]:
    """End-to-end event detection for a single pixel time series.

    Returns {year: annual_metrics} plus classification array on the result
    dict under key "classes". Base-period thresholds use calendar-day
    percentiles; severity uses standardised anomalies of the FULL series
    (standardisation parameters must come from the base period for research
    use -- see core.spi / build_climatology for the cube-level version).
    """
    base = (dates >= np.datetime64(f"{base_years[0]}-01-01")) & \
           (dates <= np.datetime64(f"{base_years[-1]}-12-31"))
    t_thr = calendar_thresholds(tmax[base], dates[base], heat_pct, window)
    p30 = rolling_sum(precip, 30)
    p_thr = calendar_thresholds(p30[base], dates[base], dry_pct, window)

    t_thr_s = daily_threshold_series(dates, t_thr)
    p_thr_s = daily_threshold_series(dates, p_thr)
    cls = classify_days(tmax, p30, t_thr_s, p_thr_s)
    comp = cls == COMPOUND

    # base-period standardisation parameters (documented, not fitted here)
    sd_t = np.nanstd(tmax[base])
    mean_t = np.nanmean(tmax[base])
    sd_p = np.nanstd(p30[base])
    mean_p = np.nanmean(p30[base])
    zt = (tmax - mean_t) / sd_t
    zp = (p30 - mean_p) / sd_p

    years_arr = dates.astype("datetime64[Y]").astype(int) + 1970
    out = {}
    for y in range(years[0], years[-1] + 1):
        m = years_arr == y
        if not m.any():
            continue
        # Episode assigned to its STARTING year: the year segment is extended
        # by (min_days - 1) days into the next year so that episodes starting
        # on 29-31 December are detected once, in the start year, and are not
        # double-counted in the following year (which classifies those days
        # only through its own segment that no longer contains the start).
        lo, hi = np.where(m)[0][[0, -1]]
        hi = min(hi + min_days - 1, len(dates) - 1)
        seg = comp[lo:hi + 1]
        met = annual_metrics(seg, zt[lo:hi + 1], zp[lo:hi + 1], min_days)
        months = dates[lo:hi + 1].astype("datetime64[M]").astype(int) % 12 + 1
        met["seasonal"] = seasonal_counts(seg, months, {
            "pre_monsoon": [3, 4, 5], "monsoon": [6, 7, 8, 9],
            "post_monsoon": [10, 11], "winter": [12, 1, 2]})
        out[y] = met
    out["classes"] = cls
    return out
