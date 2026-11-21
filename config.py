"""Central configuration for the Nepal Compound Hot-Dry (CHD) pipeline.

All paths, periods, dataset IDs and scientific thresholds live here so that
every downstream module reads one consistent set of values.
Edit this file only; never hard-code parameters in analysis scripts.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Google Earth Engine assets (user-verified)
# ----------------------------------------------------------------------------
GEE_PROJECT = "mygeepr"                       # confirm during ee.Authenticate()
NEPAL_DISTRICTS_ASSET = "projects/mygeepr/assets/Nepaldist"
# NOTE: the uploaded District.shp has 77 features, fields PROVINCE/PR_NAME/
# DISTRICT, and no .prj file: coordinates are geographic degrees (EPSG:4326).
# Upload it to GEE with `crs: EPSG:4326` explicitly.

# ----------------------------------------------------------------------------
# Study area / grid
# ----------------------------------------------------------------------------
# Fixed analysis grid: ~0.1 deg (~11.1 km), matching ERA5-Land native resolution.
# CHIRPS (0.05 deg) is resampled conservatively (mean) onto this grid; never
# interpret the finer CHIRPS resolution as the resolution of the analysis.
GRID_CRS = "EPSG:4326"
GRID_SCALE = 11132.0        # metres (ERA5-Land native)
BBOX = (79.9, 26.3, 88.3, 30.5)   # min_lon, min_lat, max_lon, max_lat (pad 0.1)
DATA_DIR = Path("/content/chd_data")          # Colab working data dir
OUT_DIR = Path("/content/chd_outputs")        # Colab outputs (CSV/GeoTIFF/PNG)

# ----------------------------------------------------------------------------
# Periods
# ----------------------------------------------------------------------------
HIST_PERIOD = (1981, 2025)      # historical climate analysis
BASE_PERIOD = (1991, 2020)      # reference climatology for all thresholds
VEG_PERIOD = (2001, 2025)       # MODIS vegetation assessment
PROJ_PERIODS = {
    "near": (2021, 2040),
    "mid": (2041, 2060),
    "late": (2081, 2100),
}
SCENARIOS = ("ssp245", "ssp585")
PILOT_YEAR = 2024

# ----------------------------------------------------------------------------
# GEE dataset IDs (verified against the EE catalogue, 2026-11)
# ----------------------------------------------------------------------------
CHIRPS_ID = "UCSB-CHG/CHIRPS/DAILY"             # band: precipitation [mm/d]
ERA5_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"          # band: temperature_2m_max [K]
ERA5_TMAX_BAND = "temperature_2m_max"
CHIRPS_BAND = "precipitation"
MOD13Q1_ID = "MODIS/061/MOD13Q1"                # NDVI/EVI scale 1e-4, SummaryQA
MOD11A2_ID = "MODIS/061/MOD11A2"                # LST_Day_1km scale 0.02 [K]
WORLDCOVER_ID = "ESA/WorldCover/v200"           # band Map; cropland == 40

# ----------------------------------------------------------------------------
# Scientific thresholds (concept note section 6)
# ----------------------------------------------------------------------------
HEAT_PERCENTILE = 90      # P90 of daily Tmax, calendar-day, +/-15-day window
DRY_PERCENTILE = 20       # P20 of rolling 30-day precipitation
CLIM_WINDOW_DAYS = 15     # moving calendar window half-width
ROLLING_DAYS = 30         # precipitation accumulation window
MIN_EPISODE_DAYS = 3      # qualifying compound event length
N_BASESAMPLE_DAYS = 20    # per calendar-day window sample size floor (diagnostic)

# Severity: Se = sum(max(0, ZT) + max(0, -ZP)) over episode days, where
# ZT / ZP are standardised vs the BASE_PERIOD climatology (same calendar day).

# Seasons (Nepal, hydro-meteorological convention)
SEASONS = {
    "pre_monsoon": [3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "post_monsoon": [10, 11],
    "winter": [12, 1, 2],
}

# Crop calendar focus months (district medians; refine per-district later):
# rice transplant/monsoon Jun-Sep; maize Mar-Aug; wheat Nov-Apr.
CROP_SEASONS = {
    "rice": [6, 7, 8, 9, 10],
    "maize": [3, 4, 5, 6, 7, 8],
    "wheat": [11, 12, 1, 2, 3, 4],
}

# ----------------------------------------------------------------------------
# CMIP6 / NEX-GDDP-CMIP6 (access layout verified from the public S3 bucket)
# ----------------------------------------------------------------------------
# s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/{scenario}/{realization}/
#     {variable}/{var}_day_{model}_{scenario}_{ri}_gn_{year}[_v1.1|_v2.0].nc
GDDP_S3 = "https://nex-gddp-cmip6.s3.amazonaws.com"
GDDP_ROOT = "NEX-GDDP-CMIP6"
GDDP_VARS = {"tmax": "tasmax", "precip": "pr"}
# Default ensemble: 5 widely used models with full 1950-2100 coverage; the
# discovery function checks availability at run time and reports any gaps.
GDDP_MODELS = ("ACCESS-CM2", "GFDL-ESM4", "MPI-ESM1-2-HR", "MRI-ESM2-0",
               "NorESM2-LM")
GDDP_REALIZATION = "r1i1p1f1"

# Historical reference for bias adjustment: BASE_PERIOD observations on the
# same grid; model baseline uses GDDP historical 1991-2020.
