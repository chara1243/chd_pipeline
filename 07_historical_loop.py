# ============================================================================
# CELL 7 -- HISTORICAL LOOP 1981-2025 + district aggregation (Phases D/E prep)
# ============================================================================
# Runs the same validated detector for every year with checkpointing (one NPZ
# per year so a crash never loses work), then aggregates to districts with
# exact per-pixel area weights (cos(latitude)) -- NOT naive means, because
# equal-angle pixels shrink with latitude.
# [NOT executed in this sandbox: run in Colab after Cell 5.]
# ============================================================================

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, "/content/chd_pipeline")
import config as C
from local_analysis.run_events import detect_for_years

OUT = Path("/content/chd_outputs")
clim = np.load(OUT / "climatology.npz")
lat = clim["lat"]; lon = clim["lon"]
years = list(range(1981, 2026))

# exact area weights per grid row (fraction of total)
w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
w /= w.sum()

# districts shapefile (uploaded District.shp -> uploaded to Colab, or the
# GeoJSON packaged in this repo under data/)
gdf = gpd.read_file("/content/District.shp")     # 77 districts, EPSG:4326

def district_summary(year, met, gdf, lat, lon, w):
    """Aggregate annual metric grids to district means (area-weighted).

    Uses rasterio.features.geometry_mask per district on the shared grid.
    """
    from rasterio.features import geometry_mask
    transform = rasterio.transform.from_bounds(
        lon.min() - 0.05, lat.min() - 0.05, lon.max() + 0.05,
        lat.max() + 0.05, len(lon), len(lat))
    rows = []
    for _, r in gdf.iterrows():
        mask = geometry_mask([r.geometry], out_shape=(len(lat), len(lon)),
                             transform=transform, invert=True)
        wm = w * mask
        wm /= wm.sum()
        rows.append({
            "district": r["DISTRICT"], "province": r["PR_NAME"], "year": year,
            "n_episodes": float((met["n_episodes"] * wm).sum()),
            "episode_days": float((met["episode_days"] * wm).sum()),
            "compound_days": float((met["n_compound_days"] * wm).sum()),
            "mean_duration": float((met["mean_duration"] * wm).sum()),
            "max_duration": float((met["max_duration"] * wm).sum()),
            "severity": float((met["severity"] * wm).sum()),
        })
    return rows

import rasterio
from rasterio.transform import from_bounds as _fb

all_rows = []
for y in years:
    ckpt = OUT / f"metrics_{y}.npz"
    if ckpt.exists():
        met = dict(np.load(ckpt))
    else:
        idx = np.isin(dates, np.arange(np.datetime64(f"{y}-01-01"),
                                       np.datetime64(f"{y + 1}-01-01"),
                                       dtype="datetime64[D]"))
        res = detect_for_years(
            tmax[idx], precip[idx], dates[idx],
            clim["thr_t"], clim["thr_p"],
            {"mean": clim["clim_t_mean"], "sd": clim["clim_t_sd"]},
            {"mean": clim["clim_p_mean"], "sd": clim["clim_p_sd"]},
            [y], min_days=C.MIN_EPISODE_DAYS)
        met = res[y]
        np.savez_compressed(ckpt, **met)
    all_rows += district_summary(y, met, gdf, lat, lon, w)
    print(y, "done")

df = pd.DataFrame(all_rows)
df.to_csv(OUT / "district_annual_chd_1981_2025.csv", index=False)

# province rollup
prov = (df.groupby(["province", "year"])
        [["n_episodes", "episode_days", "severity"]].mean().reset_index())
prov.to_csv(OUT / "province_annual_chd_1981_2025.csv", index=False)

# national series (area-weighted mean of district values)
nat = (df.groupby("year")
       [["n_episodes", "episode_days", "compound_days", "mean_duration",
         "max_duration", "severity"]].mean().reset_index())
nat.to_csv(OUT / "national_annual_chd_1981_2025.csv", index=False)
print("Wrote district/province/national CSVs to", OUT)
