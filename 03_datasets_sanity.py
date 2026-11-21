# ============================================================================
# CELL 3 -- Load CHIRPS + ERA5-Land and PROVE the data are real (Phase A)
# ============================================================================
# Verifies collection IDs, band names, units, coverage, and produces national
# sanity statistics for the pilot year before any threshold logic runs.
# [NOT executed in this sandbox: run in Colab after Cells 1-2.]
# ============================================================================

import calendar

import ee

ee.Initialize(project="mygeepr")

chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
          .select("precipitation"))                 # mm/day, 0.05 deg, >=1981
era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .select("temperature_2m_max"))              # KELVIN, ~11.1 km, >=1950

nepdist = ee.FeatureCollection("projects/mygeepr/assets/Nepaldist")
nepal = nepdist.geometry()

Y = 2024                                           # pilot year
start, end = f"{Y}-01-01", f"{Y}-12-31"

chirps_y = chirps.filterDate(start, end)
era5_y = (era5.filterDate(start, end)
          .map(lambda im: im.subtract(273.15)       # K -> degC
               .copyProperties(im, ["system:time_start"])))

# ---- coverage checks --------------------------------------------------------
n_chirps = chirps_y.size().getInfo()                # expect 366 (leap year)
n_era5 = era5_y.size().getInfo()
print(f"{Y}: CHIRPS images = {n_chirps}, ERA5-Land images = {n_era5}")

# ---- national sanity statistics (reduce over Nepal) -------------------------
mean_pr = (chirps_y.mean().reduceRegion(
    reducer=ee.Reducer.mean(), geometry=nepal, scale=11132,
    maxPixels=1e13).get("precipitation").getInfo())
tmax_stats = (era5_y.mean().reduceRegion(
    reducer=ee.Reducer.mean(), geometry=nepal, scale=11132,
    maxPixels=1e13).get("temperature_2m_max").getInfo())
print(f"National mean daily precip {Y}: {mean_pr:.2f} mm/day")
print(f"National mean daily Tmax {Y}: {tmax_stats:.1f} degC")

# Interpretation: precip ~3-6 mm/day over Nepal in a normal year (monsoon
# dominated); Tmax means ~15-25 degC. Wildly different values (e.g. ~290 for
# temperature) mean the Kelvin->Celsius conversion was skipped.
