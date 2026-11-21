"""District-year panel regression linking CHD exposure to crop yields.

Model (concept note Phase G):
  ln(Y_dt) = a + b1*HDT_dt + b2*HeatDays_dt + b3*DryDays_dt
             + b4*(HDT*Heat) + ... + district FE + year FE + e_dt
with district-clustered standard errors. Estimated with statsmodels OLS on
dummy-encoded fixed effects (no linearmodels dependency required).

Exposure variables are z-scored or expressed as shares of growing-season
days; the joint heat-dry interaction tests whether the compound association
exceeds the sum of individual associations. This is an observational
association, NOT causal evidence (stated in the manuscript).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_panel(district_csv: pd.DataFrame) -> pd.DataFrame:
    """Standardise a district-year exposure table.

    Expected columns (per row):
      district, year, rice_prod_t, rice_area_ha, maize_prod_t, maize_area_ha,
      wheat_prod_t, wheat_area_ha,
      gs_compound_days, gs_heat_days, gs_dry_days  (growing-season totals for
      the crop's calendar; build in local_analysis/run_events.py)
    Adds ln_yield columns and anomaly columns (district-mean removed).
    """
    df = district_csv.copy()
    for crop in ("rice", "maize", "wheat"):
        prod = df[f"{crop}_prod_t"]
        area = df[f"{crop}_area_ha"]
        df[f"{crop}_yield"] = np.where(area > 0, prod / area, np.nan)
        df[f"ln_{crop}_yield"] = np.log(df[f"{crop}_yield"])
    # exposure shares of growing-season days
    for c in ("gs_compound_days", "gs_heat_days", "gs_dry_days"):
        if c in df.columns:
            df[c.replace("days", "share")] = df[c] / 180.0  # ~180-d season
    return df


def fit_fe_interaction(df: pd.DataFrame, crop: str = "rice",
                       exposure: str = "gs_compound_share",
                       heat: str = "gs_heat_share",
                       dry: str = "gs_dry_share") -> dict:
    """OLS with district + year dummies, interaction term, cluster-robust SE.

    Returns the statsmodels result and a tidy dict of key coefficients.
    Requires >= 3 districts and >= 3 years to identify FE.
    """
    import statsmodels.api as sm
    y = df[f"ln_{crop}_yield"]
    X = pd.DataFrame(index=df.index)
    X[exposure] = df[exposure]
    X[heat] = df[heat]
    X[dry] = df[dry]
    X["compound_x_heat"] = df[exposure] * df[heat]
    X["compound_x_dry"] = df[exposure] * df[dry]
    # fixed effects as dummies (drop_first to avoid collinearity)
    D = pd.get_dummies(df["district"].astype(str), prefix="d", drop_first=True)
    Yr = pd.get_dummies(df["year"].astype(str), prefix="y", drop_first=True)
    X = pd.concat([X.astype(float), D.astype(float), Yr.astype(float)], axis=1)
    ok = y.notna() & X.notna().all(axis=1)
    model = sm.OLS(y[ok], X[ok])
    groups = df.loc[ok, "district"]
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    keys = [exposure, heat, dry, "compound_x_heat", "compound_x_dry"]
    tidy = {k: {"coef": float(res.params[k]),
                "se": float(res.bse[k]),
                "p": float(res.pvalues[k])} for k in keys if k in res.params}
    return {"result": res, "tidy": tidy, "n": int(ok.sum()),
            "r2": float(res.rsquared)}
