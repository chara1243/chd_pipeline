# ============================================================================
# CELL 2 (THE FIX) -- Nepal districts map WITHOUT the blocked OSM basemap
# ============================================================================
# Diagnosis of the 403 error
# ---------------------------
# The 403 came from OpenStreetMap's tile servers (folium's default
# tiles="OpenStreetMap"), which blocks automated/Colab traffic by usage
# policy. It is NOT an Earth Engine problem. Three things fix it:
#   (a) create the folium map with tiles=None  -> no third-party tiles at all;
#   (b) render EE layers through ee.Image.getMapId() so Google serves the
#       EE tiles (authorization rides on your ee.Initialize session);
#   (c) verify tiles are REALLY being produced by requesting one tile URL
#       with `requests` and printing the HTTP status (200 = working).
# If (c) returns 403/404, the problem is EE-side (auth/project) -- see the
# diagnostics at the bottom. A matplotlib fallback (2b) guarantees you can
# always see the boundaries, independent of any browser/leaflet behaviour.
#
# [NOT executed in this sandbox: run in Colab after Cell 1.]
# ============================================================================

import requests

import ee
import folium
from IPython.display import display

nepdist = ee.FeatureCollection("projects/mygeepr/assets/Nepaldist")
nepal = nepdist.geometry()


def add_ee_layer(map_object, image, vis_params, name, show=True):
    """Add an EE Image to a folium map using EE's own tile service."""
    map_id = ee.Image(image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=show,
    ).add_to(map_object)


# ---- (a) map with NO basemap tiles -----------------------------------------
m = folium.Map(location=[28.3, 84.1], zoom_start=7, tiles=None,
               control_scale=True)

# district boundaries painted as a raster layer (fill + outline)
district_image = nepdist.style(color="222222", fillColor="E8F1E8", width=1.2)
add_ee_layer(m, district_image, {}, "Nepal Districts")

# ---- (c) PROVE the tile layer is real --------------------------------------
map_id = ee.Image(district_image).getMapId({})
probe = requests.get(map_id["tile_fetcher"].url_format +
                     "/8/199/120.png", timeout=30)   # z/x/y covering Nepal
print("EE tile probe HTTP status:", probe.status_code,
      "(200 = tiles are being generated; 403/404 = EE-side problem)")

m.fit_bounds([[26.3, 80.0], [30.5, 88.3]])
folium.LayerControl(collapsed=False).add_to(m)
display(m)

# ----------------------------------------------------------------------------
# CELL 2b -- Fallback static map (matplotlib): works even if folium/leaflet
# misbehaves in the browser. Downloads district polygons once as GeoJSON.
# ----------------------------------------------------------------------------
import io

import geopandas as gpd
import matplotlib.pyplot as plt

fc = nepdist.getInfo()                     # 77 features, small payload
gdf = gpd.GeoDataFrame.from_features(fc["features"])
fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=True)
gdf.plot(ax=ax, facecolor="#E8F1E8", edgecolor="#222222", linewidth=0.6)
ax.set_title("Nepal districts (EE asset verification)", fontsize=11)
ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
plt.show()
print("Fallback map rendered from", len(gdf), "districts")
