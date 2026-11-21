# ============================================================================
# CELL 5 -- Read cubes, VALIDATE them, build the 1991-2020 climatology (Phase B)
# ============================================================================
# Replaces the simplified monthly thresholds of the pilot with pixel-specific
# calendar-day percentiles over a +/-15-day moving window (concept note 6.1-
# 6.2, config.HEAT_PERCENTILE=90, DRY_PERCENTILE=20, CLIM_WINDOW_DAYS=15).
# Leap days are folded onto Feb 28 so the climatology has exactly 365 slots.
# Every cube passes the io_validators gate BEFORE analysis (Phase A rule).
# [NOT executed in this sandbox: run in Colab after the exports finish.]
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, "/content/chd_pipeline")   # where this repo is cloned
from core import io_validators
from core.events import (calendar_thresholds, dayofyear_calendar,
                         rolling_sum)
from local_analysis.run_events import standardization_climatology

import config as C

DATA = Path("/content/chd_data")             # chd_tmax_*.tif, chd_pr_*.tif
YEARS = list(range(1981, 2026))
BASE = (1991, 2020)


def read_year(var, year, data_dir=DATA):
    with rasterio.open(data_dir / f"chd_{var}_{year}.tif") as src:
        a = src.read()                        # (366, ny, nx) or (365, ...)
        prof = src.profile
        dates = np.array([np.datetime64(f"{year}-01-01") + np.timedelta64(i, "D")
                          for i in range(a.shape[0])])
        # band names are YYYYMMDD; assert they match the assumed order
        names = src.descriptions
        expected = [f"{year}{m:02d}{d:02d}" for m, d in
                    ((1, 1), (7, 1), (12, 31))]
        got = [names[0][:8], names[len(names) // 2][:8], names[-1][:8]]
        assert got == expected, f"band order mismatch {year} {var}: {got}"
    return a.astype(np.float32), dates, prof


# ---- read all years ----------------------------------------------------------
tmax_cube, dates_cube = [], []
for y in YEARS:
    a, d, _ = read_year("tmax", y)
    tmax_cube.append(a); dates_cube.append(d)
tmax = np.concatenate(tmax_cube); dates = np.concatenate(dates_cube)

# 1981 lead-in: CHIRPS starts 1981-01-01, so the first 29 days have no full
# 30-day window -> rolling_sum returns NaN there (documented exclusion).
pr_cube, pr_dates = [], []
for y in YEARS:
    a, d, _ = read_year("pr", y)
    pr_cube.append(a); pr_dates.append(d)
precip = np.concatenate(pr_cube)

ny, nx = tmax.shape[1:]
lat = np.arange(30.475, 26.275, -0.101)[:ny]   # top->bottom, ~0.1 deg
lon = np.arange(79.975, 88.475, 0.101)[:nx]
print("Cube shape:", tmax.shape, precip.shape)

# ---- validators (hard gate) --------------------------------------------------
ctx = {"dates": dates, "year": 1991, "tmax_c": tmax, "precip": precip,
       "lat": lat, "lon": lon}
ok = io_validators.run_all_checks(ctx)
assert ok, "Cube validation FAILED -- fix inputs before analysis."

# ---- calendar-day thresholds (the Phase B upgrade) --------------------------
base_mask = ((dates >= np.datetime64(f"{BASE[0]}-01-01")) &
             (dates <= np.datetime64(f"{BASE[1]}-12-31")))

thr_t = calendar_thresholds(tmax[base_mask], dates[base_mask],
                            C.HEAT_PERCENTILE, C.CLIM_WINDOW_DAYS)  # (365,H,W)
p30 = rolling_sum(precip, C.ROLLING_DAYS)
thr_p = calendar_thresholds(p30[base_mask & np.isfinite(p30)],
                            dates[base_mask & np.isfinite(p30)],
                            C.DRY_PERCENTILE, C.CLIM_WINDOW_DAYS)

clim_t = standardization_climatology(tmax, dates, base_mask,
                                     C.CLIM_WINDOW_DAYS)   # ZT fields
clim_p = standardization_climatology(p30, dates, base_mask,
                                     C.CLIM_WINDOW_DAYS)   # ZP fields

np.savez_compressed("/content/chd_outputs/climatology.npz",
                    thr_t=thr_t.astype(np.float32),
                    thr_p=thr_p.astype(np.float32),
                    clim_t_mean=clim_t["mean"].astype(np.float32),
                    clim_t_sd=clim_t["sd"].astype(np.float32),
                    clim_p_mean=clim_p["mean"].astype(np.float32),
                    clim_p_sd=clim_p["sd"].astype(np.float32),
                    lat=lat, lon=lon)
print("Climatology saved: thresholds (365,%d,%d) + Z-standardization fields"
      % thr_t.shape[1:])

# Quick climatological plausibility print (NOT results -- just QA):
july = 196                                                    # ~15 Jul
print("Sample threshold, ~15 July, grid centre:",
      float(thr_t[july, ny // 2, nx // 2]), "degC (expect ~24-32)",
      float(thr_p[july, ny // 2, nx // 2]), "mm/30d (expect monsoon-high)")
