"""Input-cube validators: run before any event analysis (Phase A discipline).

Each check returns (ok: bool, message: str) and the runner prints a compact
report. These checks catch the silent failures that most often corrupt
downstream statistics: unit errors (K vs degC), fill values, reversed axes,
date gaps, and misaligned grids.
"""
from __future__ import annotations

import numpy as np


def check_dates_complete(dates: np.ndarray, year: int) -> tuple:
    """Every calendar day of `year` present exactly once (leap-aware)."""
    start = np.datetime64(f"{year}-01-01")
    end = np.datetime64(f"{year + 1}-01-01")
    expected = np.arange(start, end)
    d = np.unique(dates)
    missing = np.setdiff1d(expected, d).astype("datetime64[D]")
    ok = len(missing) == 0 and len(d) == len(expected)
    return ok, f"{year}: {len(d)} days (expected {len(expected)}); " \
               f"missing {len(missing)}"


def check_units_tmax_celsius(tmax_c: np.ndarray) -> tuple:
    """Plausible Celsius range for Nepal: -50 < mean-of-finite < 60."""
    fin = tmax_c[np.isfinite(tmax_c)]
    if fin.size == 0:
        return False, "tmax all NaN"
    lo, hi = np.percentile(fin, [0.1, 99.9])
    mean = fin.mean()
    ok = (mean > -50) and (mean < 60) and (hi < 70) and (lo > -60)
    return ok, (f"tmax mean {mean:.1f} C, p0.1 {lo:.1f}, p99.9 {hi:.1f} "
                f"{'[OK degC]' if ok else '[CHECK: still Kelvin?]'}")


def check_precip_mm(precip: np.ndarray) -> tuple:
    fin = precip[np.isfinite(precip)]
    if fin.size == 0:
        return False, "precip all NaN"
    mx = np.nanpercentile(fin, 99.9)
    neg = float((fin < -0.01).mean())
    ok = (mx < 800) and (neg < 0.001)
    return ok, f"precip p99.9 {mx:.1f} mm, negative fraction {neg:.5f}"


def check_grid_alignment(lat: np.ndarray, lon: np.ndarray,
                         scale_deg: float = 0.1) -> tuple:
    """Regular ascending grid at ~scale_deg spacing."""
    ok_lat = np.all(np.diff(lat) > 0)
    ok_lon = np.all(np.diff(lon) > 0)
    dlat = np.median(np.diff(lat))
    dlon = np.median(np.diff(lon))
    ok = ok_lat and ok_lon and abs(dlat - scale_deg) < scale_deg * 0.25 \
        and abs(dlon - scale_deg) < scale_deg * 0.25
    return ok, f"dlat {dlat:.3f} dlon {dlon:.3f} deg (target ~{scale_deg})"


def check_nan_fraction(cube: np.ndarray, name: str, max_frac: float = 0.2):
    frac = float(np.mean(~np.isfinite(cube)))
    return frac <= max_frac, f"{name}: NaN fraction {frac:.3f}"


def run_all_checks(ctx: dict) -> bool:
    """ctx keys: dates, tmax_c, precip, lat, lon, year. Prints report."""
    checks = [
        check_dates_complete(ctx["dates"], ctx["year"]),
        check_units_tmax_celsius(ctx["tmax_c"]),
        check_precip_mm(ctx["precip"]),
        check_grid_alignment(ctx["lat"], ctx["lon"]),
        check_nan_fraction(ctx["tmax_c"], "tmax_c"),
        check_nan_fraction(ctx["precip"], "precip"),
    ]
    all_ok = True
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        all_ok &= ok
    return all_ok
