# ============================================================================
# CELL 10 -- NEX-GDDP-CMIP6 future projections (Phase H)
# ============================================================================
# Access layout VERIFIED against the public S3 bucket (2026-11):
#   s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/{scenario}/{realization}/
#       {variable}/{var}_day_{model}_{scenario}_{ri}_gn_{year}[_vX.X].nc
#   scenarios: historical, ssp126, ssp245, ssp370, ssp585; variables incl.
#   tasmax, pr (plus hurs etc.). File variants: base, _v1.1, _v2.0.
# Units: tasmax K -> degC; pr kg m-2 s-1 -> mm/day (x86400).
# Bias adjustment: Quantile Delta Mapping (Cannon et al. 2015) against the
#   1991-2020 observed climatology on the SAME grid (quantiles computed per
#   calendar-day window). CHD detection then reuses the SAME observed
#   1991-2020 thresholds (thr_t, thr_p) applied to bias-adjusted series --
#   the consistent-threshold requirement of the concept note (Phase H).
# [NOT executed in this sandbox: run in Colab -- needs ~50 GB working disk
#  for the full ensemble; start with ONE model for the pilot.]
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr

sys.path.insert(0, "/content/chd_pipeline")
import config as C
from core.events import calendar_thresholds, classify_days, rolling_sum

fs = s3fs.S3FileSystem(anon=True)
ROOT = f"{C.GDDP_S3}/{C.GDDP_ROOT}"


def gddp_files(model, scenario, var_gddp, years):
    """Enumerate year files for one model/scenario; tolerate version suffixes
    (base -> _v2.0 -> _v1.1 priority, verified naming)."""
    out = []
    ri = C.GDDP_REALIZATION
    for y in years:
        base = f"{C.GDDP_ROOT}/{model}/{scenario}/{ri}/{var_gddp}/" \
               f"{var_gddp}_day_{model}_{scenario}_{ri}_gn_{y}"
        cands = [base + ".nc", base + "_v2.0.nc", base + "_v1.1.nc"]
        hit = next((f"s3://{C.GDDP_S3.split('//')[1]}/{c}"
                    for c in cands if fs.exists(f"{C.GDDP_ROOT}/{model}/{scenario}/{ri}/{var_gddp}/" + c.split('/')[-1])), None)
        if hit is None:
            print("MISSING:", model, scenario, var_gddp, y)
        else:
            out.append(hit)
    return out


def load_subset(model, scenario, var_gddp, years, bbox):
    """Open the year files and subset to the Nepal bbox (lazy, then load)."""
    files = gddp_files(model, scenario, var_gddp, years)
    ds = xr.open_mfdataset(files, combine="by_coords", engine="h5netcdf",
                           chunks={"time": 366})
    sub = ds[var_gddp].sel(lat=slice(bbox[3], bbox[1]),
                           lon=slice(bbox[0], bbox[2])).load()
    if var_gddp == "tasmax":
        sub = sub - 273.15
    elif var_gddp == "pr":
        sub = sub * 86400.0
    return sub


def quantile_delta_mapping(obs_train, mod_train, mod_future):
    """1-D QDM per grid cell over the full daily distribution.

    obs_train, mod_train : (Tb, ...) base-period observed / modeled arrays
    mod_future : (Tf, ...) future modeled series
    Corrects model bias and preserves the model's quantile change signal.
    """
    Tf = mod_future.shape[0]
    qs = np.linspace(0.005, 0.995, 199)
    qo = np.nanquantile(obs_train, qs, axis=0)
    qmh = np.nanquantile(mod_train, qs, axis=0)
    qmf = np.nanquantile(mod_future, qs, axis=0)
    # delta change factor per quantile (additive for temperature)
    delta = qmf - qmh
    # bias correction of the future series quantile by quantile
    out = np.empty_like(mod_future, dtype=np.float32)
    # map each future day's value through the model's historical CDF, then
    # add the corresponding delta. Vectorized with searchsorted per cell.
    flat = mod_future.reshape(Tf, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        pos = np.searchsorted(qmh[:, c], col)
        pos = np.clip(pos, 0, len(qs) - 1)
        out[:, c] = col + delta[pos, c]
    return out.reshape(mod_future.shape)


def projected_chd_metrics(model, scenario, years, thr_t, thr_p,
                          window_days=30):
    """Bias-adjusted CHD metrics for a future period, using the OBSERVED
    1991-2020 thresholds (consistent baseline across past and future)."""
    hist_y = list(range(1991, 2021))
    obs = np.load("/content/chd_outputs/climatology.npz")
    # observed base period arrays from the cubes (loaded in Cell 5 scope)
    obs_t = tmax[base_mask]; obs_p = p30[base_mask & np.isfinite(p30)]

    fut_t = load_subset(model, scenario, "tasmax", years, C.BBOX).values
    fut_p = load_subset(model, scenario, "pr", years, C.BBOX).values
    fut_p30 = rolling_sum(fut_p, window_days)

    mod_hist_t = load_subset(model, "historical", "tasmax",
                             hist_y, C.BBOX).values
    mod_hist_p30 = rolling_sum(
        load_subset(model, "historical", "pr", hist_y, C.BBOX).values,
        window_days)

    adj_t = quantile_delta_mapping(obs_t, mod_hist_t, fut_t)
    adj_p = quantile_delta_mapping(obs_p, mod_hist_p30, fut_p30)

    fut_dates = np.arange(np.datetime64(f"{years[0]}-01-01"),
                          np.datetime64(f"{years[-1] + 1}-01-01"))
    from local_analysis.run_events import detect_for_years
    res = detect_for_years(adj_t, adj_p, fut_dates,
                           thr_t, thr_p,
                           {"mean": obs["clim_t_mean"],
                            "sd": obs["clim_t_sd"]},
                           {"mean": obs["clim_p_mean"],
                            "sd": obs["clim_p_sd"]},
                           years, min_days=C.MIN_EPISODE_DAYS)
    return {y: {k: float(np.nanmean(v)) for k, v in res[y].items()}
            for y in years}
