# ============================================================================
# CELL 8 -- TREND ANALYSIS (Phase E): Mann-Kendall + Sen + FDR
# ============================================================================
# District-level trends of every CHD metric (1981-2025) with Yue-Pilon
# prewhitening option, field-significance control by Benjamini-Hochberg FDR,
# and the national annual series test. Uses core.trends (unit-tested).
# [NOT executed in this sandbox: run in Colab after Cell 7.]
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content/chd_pipeline")
from core.trends import fdr_bh, trend_with_prewhitening

OUT = Path("/content/chd_outputs")
df = pd.read_csv(OUT / "district_annual_chd_1981_2025.csv")

METRICS = ["n_episodes", "episode_days", "compound_days", "mean_duration",
           "max_duration", "severity"]

# ---- national trends ---------------------------------------------------------
nat = pd.read_csv(OUT / "national_annual_chd_1981_2025.csv")
rows = []
for m in METRICS:
    r = trend_with_prewhitening(nat[m].values, prewhiten=True)
    rows.append({"metric": m, "sen_slope_per_decade": r["slope"] * 10,
                 "ci_lo": r["ci_lo"] * 10, "ci_hi": r["ci_hi"] * 10,
                 "p": r["p"], "tau": r["tau"]})
print(pd.DataFrame(rows).to_string(index=False))

# ---- district trends + FDR ---------------------------------------------------
out_rows = []
for (dist, prov), g in df.groupby(["district", "province"]):
    g = g.sort_values("year")
    for m in METRICS:
        r = trend_with_prewhitening(g[m].values, prewhiten=True)
        out_rows.append({"district": dist, "province": prov, "metric": m,
                         "slope_decade": r["slope"] * 10, "p": r["p"],
                         "tau": r["tau"]})
tr = pd.DataFrame(out_rows)

# FDR per metric across the 77 districts
tr["p_fdr"] = np.nan
for m, g in tr.groupby("metric"):
    tr.loc[g.index, "p_fdr"] = g["p"].values  # placeholder for mapping
sig_masks = {}
for m, g in tr.groupby("metric"):
    sig_masks[m] = fdr_bh(g["p"].values, alpha=0.05)
    tr.loc[g.index, "fdr_sig"] = sig_masks[m]

tr.to_csv(OUT / "district_trends_chd.csv", index=False)
print("\nSignificant (FDR 5%) district trends per metric:")
print(tr.groupby("metric")["fdr_sig"].sum().to_string())
