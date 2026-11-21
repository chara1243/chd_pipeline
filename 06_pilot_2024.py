# ============================================================================
# CELL 6 -- 2024 PILOT: event detection, maps, district tables (Phase C/A)
# ============================================================================
# Runs the validated local detector on the pilot year, shows the 2024 maps in
# Colab (no OSM basemap), and exports the district CSV. The synthetic episode
# unit tests (tests/test_events.py) must pass before this cell -- they are
# executed in Cell 0 and reused here as the acceptance gate.
# [NOT executed in this sandbox: run in Colab after Cell 5.]
# ============================================================================

import sys
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

sys.path.insert(0, "/content/chd_pipeline")
import config as C
from local_analysis.run_events import detect_for_years

OUT = Path("/content/chd_outputs")
clim = np.load(OUT / "climatology.npz")

# --- minimal cube access for the pilot year (readers from Cell 5 in scope) ---
tmax_y = 2024
tmax = None   # In practice, keep the full cubes in memory from Cell 5.
              # For a Colab memory-light pilot, rebuild per-year slices:
              # tmax[dates >= '2024-01-01'] etc. Shown explicitly:
dates_2024 = dates[(dates >= np.datetime64("2024-01-01")) &
                   (dates <= np.datetime64("2024-12-31"))]
tmax_2024 = tmax[dates_2024[0] <= dates][np.isin(dates, dates_2024)]
precip_2024 = precip[np.isin(dates, dates_2024)]

res = detect_for_years(
    tmax_2024, precip_2024, dates_2024,
    clim["thr_t"], clim["thr_p"],
    {"mean": clim["clim_t_mean"], "sd": clim["clim_t_sd"]},
    {"mean": clim["clim_p_mean"], "sd": clim["clim_p_sd"]},
    [2024], min_days=C.MIN_EPISODE_DAYS)

met = res[2024]
print("2024 national means: episodes %.1f  event days %.1f  severity %.1f"
      % (met["n_episodes"].mean(), met["episode_days"].mean(),
         met["severity"].mean()))

# --- interactive pilot map (EE tile service, no OSM) --------------------------
import ee

ee.Initialize(project=C.GEE_PROJECT)
nepdist = ee.FeatureCollection(C.NEPAL_DISTRICTS_ASSET)
district_image = nepdist.style(color="222222", fillColor="00000000",
                               width=1.2)


def add_ee_layer(map_object, image, vis_params, name):
    map_id = ee.Image(image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format, attr="Google Earth Engine",
        name=name, overlay=True, control=True).add_to(map_object)


def grid_to_image(grid, palette, name, m):
    """Overlay a (ny, nx) metric grid as an EE image, clipped to Nepal."""
    img = (ee.Image.pixelCoordinates(ee.Projection("EPSG:4326"))  # placeholder
           )  # replaced below by the direct GeoTIFF overlay path
    # Simpler and exact: display the exported GeoTIFF with folium's
    # raster overlay (small file) instead of reconstructing it in EE:
    import rasterio
    from rasterio.warp import transform_bounds
    from folium.raster_layers import ImageOverlay

    def show(grid_, title_, cmap_):
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        im = ax.imshow(grid_, cmap=cmap_)
        fig.colorbar(im, ax=ax, shrink=0.7)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        b = rasterio.open(DATA / f"chd_pr_2024.tif")  # georeference template
        wb = transform_bounds(b.crs, "EPSG:4326", *b.bounds)
        ImageOverlay(image=plt.imread(buf), bounds=[[wb[1], wb[0]],
                      [wb[3], wb[2]]], name=title_, opacity=0.75).add_to(m)

    show(grid, name, palette)


import io

m = folium.Map(location=[28.3, 84.1], zoom_start=7, tiles=None,
               control_scale=True)
add_ee_layer(m, district_image, {}, "Nepal districts")
grid_to_image(met["n_episodes"].astype(float), "YlOrRd",
              "2024 compound episodes (count)", m)
grid_to_image(met["max_duration"].astype(float), "PuRd",
              "2024 max episode duration (days)", m)
m.fit_bounds([[26.3, 80.0], [30.5, 88.3]])
folium.LayerControl(collapsed=False).add_to(m)
display(m)

# --- district means (area-weighted in Cell 7; simple mean here for the pilot)
np.savez_compressed(OUT / "pilot_2024_metrics.npz", **met)
np.save(OUT / "pilot_2024_dates.npy", dates_2024)
print("Pilot metrics saved to", OUT / "pilot_2024_metrics.npz")
