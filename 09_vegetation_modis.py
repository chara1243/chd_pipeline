# ============================================================================
# CELL 9 -- MODIS NDVI/EVI vegetation response (Phase F)
# ============================================================================
# MOD13Q1 (Collection 6.1): NDVI/EVI scale 1e-4; quality filter SummaryQA<=1
# (bits 0-1: 0=good, 1=marginal; excludes cloud/snow fills). District-monthly
# means are exported as a GEE table in year chunks (the audited two-stage
# pattern), then matched to CHD exposure months in Python.
# [NOT executed in this sandbox: run in Colab.]
# ============================================================================

import ee

ee.Initialize(project="mygeepr")
nepdist = ee.FeatureCollection("projects/mygeepr/assets/Nepaldist")

mod13 = (ee.ImageCollection("MODIS/061/MOD13Q1")
         .select(["NDVI", "EVI", "SummaryQA"]))


def qa_mask(im):
    qa = im.select("SummaryQA")
    ok = qa.bitwiseAnd(3).lte(1)          # good + marginal quality
    vi = im.select(["NDVI", "EVI"]).multiply(0.0001)
    return vi.updateMask(ok).copyProperties(im, ["system:time_start"])


mod13q = mod13.map(qa_mask)


def year_table(year):
    ic = mod13q.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    fc = ic.map(lambda im: im.reduceRegions(
        collection=nepdist,
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.stdDev(), sharedInputs=True),
        scale=250)).flatten() \
        .select(["DISTRICT", "PR_NAME", "mean", "mean_1", "system:index"])
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=f"CHD_NDVI_{year}",
        folder="nepal_chd_cubes", fileFormat="CSV",
        selectors=["DISTRICT", "PR_NAME", "mean", "mean_1", "system:index"])
    task.start()
    return task.id


ids = [year_table(y) for y in range(2001, 2026)]
print("Started", len(ids), "NDVI table exports (2001-2025).")
# Each yearly table has 23 16-day composites x 77 districts ~ 1.8k rows.
# After completion, merge in pandas; composite month = from system:index
# (e.g. 2024-01-01 -> month 1). Then compute per-district growing-season
# anomalies (z vs 2001-2020) and composite with the CHD classes from Cells 6-7:
# class means (normal / heat-only / dry-only / compound) and lag responses
# (composite of the NEXT 16-day period for delayed response).
