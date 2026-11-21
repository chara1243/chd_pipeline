# Review of the 2024 draft workflow — problems found and fixes applied

(Deliverable I of the revised brief. References: "original draft" = the
~14-step Colab workflow described in the concept note section 7–9.)

## 1. Map visualization (the 403 problem)

**Diagnosis.** The 403 came from OpenStreetMap tile servers blocking
Colab/folium traffic by usage policy. The draft's `geemap.foliummap` defaults
to `tiles="OpenStreetMap"`. Disabling basemaps one by one was treating the
symptom; the layer that was failing was never the EE layer.

**Fix (Cell 2).**
- `folium.Map(..., tiles=None)` — no third-party tiles at all.
- EE layers added via `ee.Image(...).getMapId()` → tiles served by Google
  under your EE session.
- **Proof-of-life check**: one tile URL is fetched with `requests` and the
  HTTP status printed (200 = tiles really generated). This separates
  "EE tile problem" from "browser rendering problem", which the brief asked
  to diagnose rather than guess.
- `fit_bounds([[26.3, 80.0], [30.5, 88.3]])` + LayerControl.
- Static matplotlib fallback (Cell 2b) that renders the 77 districts from a
  `getInfo()` download — works even if leaflet misbehaves.

## 2. Climatological thresholds (scientific upgrade)

**Problem.** Pilot used monthly-grouped P90/P20 (and P30 sampled only on the
5th/15th/25th) — a coarse approximation that biases extreme-day counts.

**Fix (Cell 5 + core/events.py).** Pixel-specific calendar-day percentiles on
a ±15-day moving window over 1991–2020; leap days folded onto Feb 28
(unit-tested); thresholds saved once (`climatology.npz`) and reused for
every year — past and future alike, so exposure is comparable across periods.

## 3. Rolling precipitation

**Fix (core/events.py `rolling_sum`).** Continuous 30-day window across year
boundaries (unit test: Jan-5 window includes late December); the first 29
days of 1981 are NaN by construction (documented exclusion); windows with
>3 missing days are NaN (no silent zero-filling).

## 4. Event counting (the classic double-counting bug)

**Fix (core/events.py + local_analysis/run_events.py).** Episodes are
run-length detected, not overlapping-day counted. Synthetic tests pin the
acceptance criteria from the brief: 2 consecutive compound days → no event;
3 → one event; 10 → one event (not 8); separated runs counted separately;
cross-year episode counted once, assigned to the starting year (year segments
extended by min_days−1). Metrics distinguish `n_compound_days` (all
overlapping days) from `n_episodes`/`episode_days`.

## 5. Severity

**Fix.** Se = Σ max(0,ZT)+max(0,−ZP) with Z-standardization against
per-calendar-day 1991–2020 climatology (the pilot had no standardization
definition at all). Event-day contributions only (qualifying episodes), not
isolated days.

## 6. Processing architecture (quota & memory)

**Problem.** Per-day GEE closures over 45 years of daily images would exceed
EE limits (the brief itself flags this).

**Fix (Cell 4).** Two-stage "cube" architecture: one ~6 MB multi-band GeoTIFF
per variable per year (366 bands, 0.1° grid, EPSG:4326), all percentile /
classification / run-length work done locally in numpy. CHIRPS is aggregated
to the ERA5-Land grid with area-true `reduceResolution(mean)`; the analysis
resolution is honestly documented as ~11.1 km. District aggregation uses
cos(latitude) area weights (Cell 7), not naive means.

## 7. Validation discipline

**Fix.** `io_validators.py` runs as a hard gate before analysis (Kelvin-vs-C,
mm plausibility, date completeness, grid regularity, NaN fraction). The 18
synthetic tests are the Phase C acceptance gate; the pilot may only be
interpreted after they pass.

## 8. Statistics

- Trends: Mann-Kendall + Sen with Yue-Pilon prewhitening; BH-FDR across the
  77 districts per metric (Cell 8).
- Crop yields (Phase G): ln(yield) panel with district + year FE and
  heat×dry interactions, district-clustered SE (core/panel.py); explicitly
  association, not causation.
- CMIP6 (Phase H): NEX-GDDP-CMIP6 access layout verified against the live
  S3 bucket (model/scenario/realization/variable/{year}.nc naming, version
  suffixes); Quantile Delta Mapping bias adjustment (Cannon et al. 2015);
  observed 1991–2020 thresholds applied to bias-adjusted series.

## 9. What was intentionally NOT done yet

CMIP6 projections and crop regressions are scaffolded but flagged
"run only after the historical dataset passes review" — per the brief's
sequencing rule. No results are claimed anywhere in this package.
