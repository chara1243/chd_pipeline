# ============================================================================
# CELL 4 -- Export daily climate CUBES 1981-2025 (the architecture change)
# ============================================================================
# WHY CUBES INSTEAD OF PER-DAY GEE CLOSURES
#   The original draft computed rolling sums and percentile closures inside
#   GEE for every day of 45 years -- a graph far beyond EE's limits. Instead
#   we export each year's daily Tmax/precipitation as ONE small multi-band
#   GeoTIFF (~6 MB: 366 bands x 46 rows x 85 cols), then run thresholds,
#   classification and episode detection locally (Cells 5-7). This is exact
#   (no client-side approximation), reproducible, and quota-friendly.
#
# GRID: EPSG:4326, 11132 m (~0.1 deg) = ERA5-Land native. CHIRPS (0.05 deg)
#   is aggregated with mean reduceResolution (conservative, area-true).
#   Never interpret the finer CHIRPS resolution as the analysis resolution.
# [NOT executed in this sandbox: run in Colab; exports go to your Drive.]
# ============================================================================

import time

import ee

ee.Initialize(project="mygeepr")

nepdist = ee.FeatureCollection("projects/mygeepr/assets/Nepaldist")
nepal = nepdist.geometry()

chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(
    "temperature_2m_max")

SCALE = 11132
CRS = "EPSG:4326"
YEARS = range(1981, 2026)          # 1981-2025 inclusive

# Nepal bbox with small padding (grid cells must cover edge districts)
bbox = ee.Geometry.Rectangle([79.9, 26.3, 88.3, 30.5], proj=CRS, geodesic=False)


def year_cube(imgcol, year, band, is_chirps):
    """One multi-band GeoTIFF per year: bands named YYYYMMDD."""
    ic = (imgcol.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .filterBounds(bbox))
    if is_chirps:
        # aggregate 0.05-deg CHIRPS onto the 11.1-km grid (area-true mean)
        ic = ic.map(lambda im: im.reduceResolution(
            reducer=ee.Reducer.mean(), grid={"crs": CRS, "scale": SCALE})
            .copyProperties(im, ["system:time_start"]))
    else:
        ic = ic.map(lambda im: im.subtract(273.15)   # Kelvin -> degC
                    .copyProperties(im, ["system:time_start"]))
    img = ic.toBands().renameBands(
        lambda b: ee.String(b).replace("-", "").slice(0, 8))  # YYYYMMDD
    return img


tasks = []
for y in YEARS:
    for var, imgcol, band, is_chirps in (
            ("tmax", era5, "temperature_2m_max", False),
            ("pr", chirps, "precipitation", True)):
        img = year_cube(imgcol, y, band, is_chirps)
        t = ee.batch.Export.image.toDrive(
            image=img.clip(bbox),
            description=f"CHD_{var}_{y}",
            folder="nepal_chd_cubes",
            fileNamePrefix=f"chd_{var}_{y}",
            region=bbox, crs=CRS, scale=SCALE,
            maxPixels=1e13, formatOptions={"cloudOptimized": True})
        t.start()
        tasks.append((t.id, f"chd_{var}_{y}"))

print(f"Started {len(tasks)} export tasks (2 vars x 45 years).")
time.sleep(20)

# ---- task monitor: re-run this snippet until every task is COMPLETED --------
import collections

status = collections.Counter()
for tid, name in tasks:
    s = ee.data.getTaskStatus(tid)[0].get("state", "?")
    status[s] += 1
print(dict(status))
# Expected total runtime: ~1-2 h for all 90 tasks on a free account.
# After completion, download the folder nepal_chd_cubes to /content/chd_data
# (or mount Drive and read directly -- Cell 5 accepts either).
