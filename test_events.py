"""Synthetic validation of compound-event detection (concept note Phase C).

Each test constructs a daily series whose qualifying episodes are KNOWN and
verifies the detector against hand-computed ground truth:
  * 2 consecutive compound days      -> NOT a qualifying event
  * 3 consecutive compound days      -> exactly 1 event
  * 10 consecutive compound days     -> 1 event (not 8 or 10)
  * separated episodes               -> counted separately
  * cross-year episode               -> assigned to the STARTING year, once
  * mean / max duration              -> hand-computed values
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.events import (annual_metrics, calendar_thresholds,
                         classify_days, dayofyear_calendar, episodes_1d,
                         pixel_pipeline, rolling_sum)
from core.trends import (sens_slope, yue_pilon_prewhiten)


def _dates_for_year(y: int):
    return np.arange(np.datetime64(f"{y}-01-01"),
                     np.datetime64(f"{y + 1}-01-01"))


def _flag_from_runs(n: int, runs: list, offset: int = 0):
    f = np.zeros(n, dtype=bool)
    for start, length in runs:
        f[offset + start: offset + start + length] = True
    return f


# --------------------------------------------------------------------------
def test_two_consecutive_do_not_qualify():
    f = _flag_from_runs(365, [(10, 2)])
    m = annual_metrics(f)
    assert m["n_episodes"] == 0
    assert m["episode_days"] == 0
    assert m["n_compound_days"] == 2


def test_three_consecutive_qualify_as_one():
    f = _flag_from_runs(365, [(10, 3)])
    m = annual_metrics(f, zt=np.ones(365), zp=-np.ones(365))
    assert m["n_episodes"] == 1
    assert m["episode_days"] == 3
    assert m["mean_duration"] == 3.0
    assert m["max_duration"] == 3
    # severity: 3 days * (max(0, ZT=1) + max(0, -ZP=1)) = 6
    assert m["severity"] == pytest.approx(6.0)


def test_ten_consecutive_is_one_event():
    f = _flag_from_runs(365, [(100, 10)])
    m = annual_metrics(f, zt=np.full(365, 0.5), zp=np.full(365, -1.0))
    assert m["n_episodes"] == 1
    assert m["max_duration"] == 10
    # severity: 10 * (0.5 + 1.0) = 15
    assert m["severity"] == pytest.approx(15.0)


def test_separated_episodes_counted_separately():
    f = _flag_from_runs(365, [(5, 4), (20, 3), (60, 7)])
    m = annual_metrics(f, zt=np.zeros(365), zp=np.zeros(365))
    assert m["n_episodes"] == 3
    assert m["episode_days"] == 14
    assert m["mean_duration"] == pytest.approx(14 / 3)
    assert m["max_duration"] == 7


def test_episode_ending_at_series_end():
    f = _flag_from_runs(365, [(362, 3)])
    m = annual_metrics(f)
    assert m["n_episodes"] == 1


def test_doy_calendar_folds_leap_day():
    d = np.array([np.datetime64("2020-02-28"), np.datetime64("2020-02-29"),
                  np.datetime64("2020-03-01")])
    doy = dayofyear_calendar(d)
    assert doy[0] == doy[1] == 59   # Feb 29 folded onto Feb 28
    assert doy[2] == 60


def test_rolling_sum_crosses_year_boundary():
    # continuous series across 2 years: window at Jan 5 must include Dec
    dates = np.concatenate([_dates_for_year(2010), _dates_for_year(2011)])
    precip = np.ones(len(dates))
    p30 = rolling_sum(precip, 30)
    i_jan5 = np.where(dates == np.datetime64("2011-01-05"))[0][0]
    assert p30[i_jan5] == pytest.approx(30.0)   # 26 days of Dec + 4 days Jan
    # lead-in incomplete -> NaN for the first 29 days
    assert np.all(np.isnan(p30[:29]))
    assert np.isfinite(p30[29])


def test_rolling_sum_missing_days_masked():
    v = np.ones(90)
    v[40:44] = np.nan                     # 4 consecutive missing days
    out = rolling_sum(v, 30)              # min_valid_frac 0.9 -> >=27 valid
    assert out[39] == pytest.approx(30.0)   # last complete pre-gap window
    assert out[40] == pytest.approx(29.0)   # 1 missing tolerated
    assert out[42] == pytest.approx(27.0)   # 3 missing = exact 90% floor
    assert np.isnan(out[43])                # 4 missing -> below floor -> NaN
    assert np.isnan(out[69])
    assert out[74] == pytest.approx(30.0)   # complete again after the gap


def test_classification_categories():
    t = np.array([30.0, 35.0, 30.0, 35.0, 30.0])
    p = np.array([50.0, 50.0, 50.0, 5.0, 5.0])
    tt = np.full(5, 32.0)
    pt = np.full(5, 10.0)
    cls = classify_days(t, p, tt, pt)
    assert cls[0] == 0   # normal
    assert cls[1] == 1   # heat only
    assert cls[2] == 0
    assert cls[3] == 3   # compound
    assert cls[4] == 2   # dry only


def _synthetic_pixel(dates, hot_days, dry_days):
    """Build tmax/precip so that the given (month, day) dates are hot/dry.

    Noise sigma is small (0.2) so base-period percentiles are tight and the
    planted pattern is deterministic."""
    rng = np.random.default_rng(42)
    tmax = np.full(len(dates), 20.0) + rng.normal(0, 0.2, len(dates))
    precip = rng.gamma(1.0, 2.0, len(dates))
    hotset = set(np.datetime64(f"2018-{m:02d}-{d:02d}")
                 for m, d in hot_days)
    dryset = set(np.datetime64(f"2018-{m:02d}-{d:02d}") for m, d in dry_days)
    for i, d in enumerate(dates):
        if d in hotset:
            tmax[i] += 15
        if d in dryset:
            precip[i] = 0.0
    return tmax, precip


def test_pixel_pipeline_known_pattern():
    dates = _dates_for_year(2018)
    # all of March + April 2018 bone dry -> P30 ~ 0 through the dry spell;
    # heat planted on Apr 10-12 (+ May 1-2, a 2-day run that must NOT count)
    tmax, precip = _synthetic_pixel(
        dates,
        hot_days=[(4, 10), (4, 11), (4, 12), (5, 1), (5, 2)],
        dry_days=[(3, d) for d in range(1, 32)] +
                 [(4, d) for d in range(1, 31)],
    )
    base_dates = np.concatenate([_dates_for_year(y) for y in
                                 range(1991, 2021)])
    bt, bp = _synthetic_pixel(base_dates, [], [])
    all_dates = np.concatenate([base_dates, dates])
    t = np.concatenate([bt, tmax])
    p = np.concatenate([bp, precip])
    res = pixel_pipeline(t, p, all_dates, (2018, 2018), (1991, 2020),
                         min_days=3)
    m = res[2018]
    # qualifying: exactly the Apr 10-12 run (May 1-2 is 2 days -> excluded)
    assert m["n_episodes"] == 1
    assert m["max_duration"] == 3
    assert m["episode_days"] == 3
    assert m["mean_duration"] == pytest.approx(3.0)


def test_cross_year_episode_assigned_to_start_year():
    # continuous 2017-2018 series; compound run Dec 30 - Jan 2 (4 days)
    dates = np.concatenate([_dates_for_year(2017), _dates_for_year(2018)])
    comp_days = [np.datetime64("2017-12-30"), np.datetime64("2017-12-31"),
                 np.datetime64("2018-01-01"), np.datetime64("2018-01-02")]
    flag = np.isin(dates, comp_days)
    # (a) the episode spans the boundary as ONE run of 4
    eps = episodes_1d(flag, 3)
    assert len(eps) == 1
    assert eps[0][0] == 363          # 2017-12-30 (0-based day index)
    assert eps[0][1] == 4
    # (b) start-year assignment: the caller extends the year segment by
    # min_days-1 days into the next year (implemented in pixel_pipeline);
    # simulate that extended segment for 2017:
    lo = 0
    hi = 365 + 2                     # 2017 days + 2-day extension
    m2017 = annual_metrics(flag[lo:hi], zt=flag[lo:hi].astype(float),
                           zp=-flag[lo:hi].astype(float))
    assert m2017["n_episodes"] == 1
    assert m2017["episode_days"] == 4
    # (c) severity of the boundary-crossing episode counted once
    assert m2017["severity"] == pytest.approx(4 * 2.0)


def test_prewhitening_preserves_true_trend():
    rng = np.random.default_rng(11)
    t = np.arange(120, dtype=float)
    x = 0.5 * t + 0.9 * np.zeros(120)
    for i in range(1, 120):          # AR(1) noise around the true trend
        x[i] = 0.5 * i + 0.8 * (x[i - 1] - 0.5 * (i - 1)) + rng.normal(0, 1)
    pw = yue_pilon_prewhiten(x)
    r = sens_slope(pw)
    assert np.isfinite(r["slope"])
    assert abs(r["slope"] - 0.5) < 0.25


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
