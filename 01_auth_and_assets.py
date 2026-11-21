# ============================================================================
# CELL 1 -- Authenticate, set the Cloud project, and VERIFY the district asset
# ============================================================================
# What it does (concept note Phase A):
#   1. Authenticates Google Earth Engine from Colab.
#   2. Initializes with the project `mygeepr` (change if the console shows
#      a different enabled project).
#   3. Prints the feature count, property names and one sample feature of the
#      Nepal district asset -- we NEVER assume 77 districts without checking.
#
# [NOT executed in this sandbox: requires your Google account. Run in Colab.]
# ============================================================================

import ee

ee.Authenticate()                       # opens the OAuth flow in Colab
ee.Initialize(project="mygeepr")        # confirm this is the right project

nepdist = ee.FeatureCollection("projects/mygeepr/assets/Nepaldist")
nepal = nepdist.geometry()

# --- asset verification -----------------------------------------------------
n_features = nepdist.size().getInfo()
first = nepdist.first().getInfo()
props = first["properties"]
print("District features:", n_features)
print("Property names:", sorted(props.keys()))
print("Sample district:", props.get("DISTRICT", props.get("District")),
      "| province:", props.get("PR_NAME", props.get("Province")))
print("Geometry type:", first["geometry"]["type"])
print("Province names:", sorted(
    nepdist.aggregate_array("PR_NAME").distinct().getInfo()))
# Expected (matches the uploaded District.shp): 77 features; fields
# OBJECTID, PROVINCE, PR_NAME, DISTRICT, geometry; polygon/multipolygon.
# The uploaded shapefile carries no .prj: coordinates are EPSG:4326 degrees.
# If the count differs from 77, STOP and reconcile the asset first.
