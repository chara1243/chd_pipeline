"""Local cube-level event detection driver (Phase B/C/D of the concept note).

Operates on annual daily cubes (time, ny, nx) with calendar-day climatologies.
Everything is vectorized: run-length statistics across the whole grid are
computed with forward/backward run counters instead of per-pixel Python
loops, so 45 years x 3,910 pixels runs in minutes.

Metrics per pixel-year (concept note 6.4):
  n_episodes, episode_days, mean_duration, max_duration, severity,
  n_compound_days, seasonal compound-day counts.
Cross-year policy: each year's segment is extended by (min_days - 1) days
into the next year; episodes belong to their starting year and are never
double-counted (unit-tested in tests/test_events.py).
"""
from __future__ import annotations

import numpy as np

from core.events import (COMPOUND, NORMAL, classify_days, rolling_sum)


def forward_backward_run_lengths(c: np.ndarray) -> tuple:
    """c: (T,) or (T, ny, nx) bool. Returns (L_back, L_fwd): number of
    consecutive True days ending at / starting from each day."""
    T = c.shape[0]
    L_back = np.zeros(c.shape, dtype=np.int16)
    L_fwd = np.zeros(c.shape, dtype=np.int16)
    # backward: running count of consecutive Trues ending at day t
    prev = np.zeros(c.shape[1:], dtype=np.int16)
    for t in range(T):
        prev = np.where(c[t], prev + 1, 0).astype(np.int16)
        L_back[t] = prev
    prev = np.zeros(c.shape[1:], dtype=np.int16)
    for t in range(T - 1, -1, -1):
        prev = np.where(c[t], prev + 1, 0).astype(np.int16)
        L_fwd[t] = prev
    return L_back, L_fwd


def year_metrics_grid(comp: np.ndarray, zt: np.ndarray, zp: np.ndarray,
                      min_days: int = 3) -> dict:
    """Vectorized annual metrics for one year across the whole grid.

    comp : (T, ny, nx) bool compound-day flag for the (extended) year segment
    zt, zp : standardized anomalies, same shape
    Returns dict of 2-D (ny, nx) metric arrays.
    """
    L_back, L_fwd = forward_backward_run_lengths(comp.astype(bool))
    run_len = (L_back + L_fwd - 1).astype(np.int16)
    is_event_day = comp & (run_len >= min_days)
    # episode starts: first day of a run that qualifies
    is_start = comp & (L_back == 1) & (L_fwd >= min_days)
    T, ny, nx = comp.shape

    n_episodes = is_start.sum(axis=0).astype(np.int16)
    # total episode days = sum over starts of that run's length
    episode_days = np.where(is_start, run_len, 0).sum(axis=0).astype(np.int32)
    # duration stats
    dur_at_start = np.where(is_start, run_len, 0)
    max_duration = np.where(n_episodes > 0, dur_at_start.max(axis=0),
                            0).astype(np.int16)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_duration = np.where(n_episodes > 0,
                                 episode_days / np.maximum(n_episodes, 1),
                                 0.0)
    n_compound_days = comp.sum(axis=0).astype(np.int16)
    # severity: contributions only on qualifying event days
    contrib = (np.maximum(0, zt) + np.maximum(0, -zp))
    severity = np.where(is_event_day, contrib, 0.0).sum(axis=0)
    return {
        "n_episodes": n_episodes,
        "episode_days": episode_days,
        "n_compound_days": n_compound_days,
        "mean_duration": mean_duration.astype(np.float32),
        "max_duration": max_duration,
        "severity": severity.astype(np.float32),
    }


def standardization_climatology(cube: np.ndarray, dates: np.ndarray,
                                base_mask: np.ndarray,
                                window: int = 15) -> dict:
    """Per-calendar-day mean/sd for severity standardization (ZT, ZP)."""
    from core.events import calendar_thresholds, dayofyear_calendar
    doy = dayofyear_calendar(dates)
    mus, sds = [], []
    for d in range(1, 366):
        idxs = [(d - 1 + k) % 365 for k in range(-window, window + 1)]
        sel = np.isin(doy, [i + 1 for i in idxs]) & base_mask
        pool = cube[sel]
        mus.append(np.nanmean(pool, axis=0))
        sds.append(np.nanstd(pool, axis=0))
    return {"mean": np.stack(mus), "sd": np.stack(sds)}


def expand_calendar_fields(field_365: np.ndarray, dates: np.ndarray) -> np.ndarray:
    from core.events import dayofyear_calendar
    doy = dayofyear_calendar(dates)
    return field_365[doy - 1]


def detect_for_years(tmax: np.ndarray, precip: np.ndarray, dates: np.ndarray,
                     thr_t: np.ndarray, thr_p: np.ndarray,
                     clim_t: dict, clim_p: dict,
                     years: list, min_days: int = 3) -> dict:
    """Full detection pipeline on prepared cubes.

    thr_t/thr_p : (365, ny, nx) calendar-day thresholds
    clim_t/clim_p : standardization climatologies from standardization_climatology
    Returns {year: metrics dict of 2-D arrays} plus the classification cube.
    """
    p30 = rolling_sum(precip, days=30)
    thr_t_s = expand_calendar_fields(thr_t, dates)
    thr_p_s = expand_calendar_fields(thr_p, dates)
    cls = classify_days(tmax, p30, thr_t_s, thr_p_s)
    comp = cls == COMPOUND

    mu_t_s = expand_calendar_fields(clim_t["mean"], dates)
    sd_t_s = expand_calendar_fields(clim_t["sd"], dates)
    mu_p_s = expand_calendar_fields(clim_p["mean"], dates)
    sd_p_s = expand_calendar_fields(clim_p["sd"], dates)
    zt = (tmax - mu_t_s) / sd_t_s
    zp = (p30 - mu_p_s) / sd_p_s

    years_arr = dates.astype("datetime64[Y]").astype(int) + 1970
    out = {}
    for y in years:
        idx = np.where(years_arr == y)[0]
        if len(idx) == 0:
            continue
        # extend segment into next year for cross-year episodes
        ext = idx[0], min(idx[-1] + min_days - 1, len(dates) - 1)
        lo, hi = np.where(years_arr == y)[0][0], ext[1]
        seg = np.unique(np.concatenate([idx, np.arange(idx[-1] + 1, hi + 1)]))
        out[y] = year_metrics_grid(comp[seg], zt[seg], zp[seg], min_days)
    out["classes"] = cls
    out["p30"] = p30
    return out
