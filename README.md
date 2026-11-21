# Nepal Compound Hot-Dry (CHD) Events — Research Code Pipeline

Companion code for: *Increasing Compound Hot–Dry Events in Nepal: Historical
Trends, Future Climate Projections and Implications for Agricultural
Productivity*.

## What is here

| Path | Content |
|---|---|
| `config.py` | All periods, thresholds, GEE asset, dataset IDs (single source of truth) |
| `core/events.py` | Calendar-day thresholds, 4-way classification, run-length episode detection, severity — **unit-tested** |
| `core/spi.py` | SPI-1/SPI-3 sensitivity variant (gamma MOM fit) — **unit-tested** |
| `core/trends.py` | Mann-Kendall, Sen's slope, Yue-Pilon prewhitening, BH-FDR — **unit-tested** |
| `core/panel.py` | District-year yield panel with FE + interaction + cluster-robust SE |
| `core/io_validators.py` | Hard QA gate on every input cube (units, gaps, grid) |
| `local_analysis/run_events.py` | Vectorized grid-level event metrics for full cubes |
| `gee_cells/01…10` | The ordered Colab cells (same code as the notebook) |
| `tests/` | 18 synthetic-validation tests (Phase C acceptance gate) — **all passing** |
| `viz/figures.py` | Elsevier-style figure functions (300 dpi, no clutter) |
| `colab/Nepal_CHD_Workflow.ipynb` | Ready-to-run notebook (Cells 0–10 + roadmap) |
| `docs/CHANGES_FROM_2024.md` | Review of the original draft + every fix |
| `docs/REMAINING_INPUTS.md` | Inputs you must still supply (crop stats etc.) |

## Quick start (Colab)

1. Upload this folder to Colab (`/content/chd_pipeline`) and upload
   `District.shp` (+ .dbf/.shx/.sbn/.cpg) next to it.
2. Run notebook Cells 0–3 (auth, asset check, fixed map, data sanity).
   The map cell uses `tiles=None` — no OpenStreetMap, no 403.
3. Run Cell 4 (cube export, ~90 small Drive tasks), then Cells 5–8 for the
   validated historical analysis and trends.
4. Cells 9–10 (vegetation, CMIP6) come after the historical dataset passes
   review — exactly the sequencing the concept note mandates.

## Execution status (honest accounting)

- **Executed here:** all 18 unit tests; synthetic episode validation
  (2-day run rejected, 3-day run = 1 event, 10-day run = 1 event, separated
  episodes counted separately, cross-year episode assigned once to start
  year); FDR/MK/Sen/SPI checks.
- **Not executed here (needs your GEE account / Drive / Colab GPU-free
  runtime):** Cells 1–10. Each cell states this at the top. Nothing in the
  notebook is presented as a research result until you run it.

## Test it yourself

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
