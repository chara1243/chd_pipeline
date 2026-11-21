# Remaining inputs you must supply (nothing here can be invented)

| # | Input | Where it goes | Status |
|---|---|---|---|
| 1 | Google account + Earth Engine access to `projects/mygeepr/assets/Nepaldist` | Cells 1–10 | user-side |
| 2 | Confirm the EE Cloud project id during `ee.Authenticate()` | Cell 1 | user-side |
| 3 | District-level crop statistics (production + harvested area) for rice/maize/wheat, ~1990/2000–2024: MoALD "Statistical Information on Nepalese Agriculture" yearbooks (annual PDFs) + district yearbook tables | `core/panel.py` / Phase G notebook cell | **you must download** |
| 4 | Crop calendars per district (sowing/harvest months; national medians are pre-configured in `config.CROP_SEASONS`) | Phase G | refine |
| 5 | Optional: official ecological-belt (Terai/Hill/Mountain) shapefile for stratified reporting | Figure 1 / stratification | optional |
| 6 | Optional: irrigation share per district (control variable for the yield panel) | Phase G | optional |

Deliberately excluded: any fabricated crop data, results, significance
statements. Everything in `results/` will be produced only by running the
pipeline.
