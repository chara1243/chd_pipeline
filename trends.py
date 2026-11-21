"""Trend analysis: Mann-Kendall test, Sen's slope, prewhitening, FDR.

Self-contained implementations (no pymannkendall dependency) so the module
runs anywhere. Includes the Yue-Pilon (2002) prewhitening option for
autocorrelation-robust trend testing and the Benjamini-Hochberg (1995)
false-discovery-rate correction for spatial field significance.
"""
from __future__ import annotations

import numpy as np


def mann_kendall(x: np.ndarray) -> dict:
    """Two-sided Mann-Kendall test with continuity and tie correction.

    x : 1-D array (NaNs removed). Returns S, var(S), z, p, tau.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return {"S": np.nan, "z": np.nan, "p": np.nan, "tau": np.nan,
                "n": n}
    # S statistic
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))
    # tie-corrected variance
    uniq, counts = np.unique(x, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return {"S": float(s), "z": float(z), "p": float(p), "tau": float(tau),
            "n": n}


def sens_slope(x: np.ndarray) -> dict:
    """Sen's slope with 95% CI (Theil-Sen). Returns slope, intercept, lo, hi,
    p (from MK test)."""
    x = np.asarray(x, dtype=float)
    idx = np.where(np.isfinite(x))[0]
    if len(idx) < 4:
        return {"slope": np.nan, "intercept": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan}
    y = x[idx]
    t = idx.astype(float)
    try:
        from scipy.stats import theilslopes
        slope, intercept, lo, hi = theilslopes(y, t, 0.95)
        mk = mann_kendall(x)
        return {"slope": float(slope), "intercept": float(intercept),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "p": mk["p"]}
    except Exception:
        return {"slope": np.nan, "intercept": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": mann_kendall(x)["p"]}


def yue_pilon_prewhiten(x: np.ndarray) -> np.ndarray:
    """Trend-free pre-whitening (Yue et al. 2002): estimate monotonic Sen
    trend, de-trend, remove AR(1) of the residuals, re-impose the trend."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 8:
        return x.copy()
    t = np.arange(len(x), dtype=float)
    tr = sens_slope(x)
    if not np.isfinite(tr["slope"]):
        return x.copy()
    resid = x - tr["slope"] * t
    r = resid[np.isfinite(resid)]
    if len(r) < 3:
        return x.copy()
    r0, r1 = r[:-1], r[1:]
    var0 = np.var(r0)
    rho = 1.0 if var0 == 0 else float(np.cov(r0, r1)[0, 1] / var0)
    rho = float(np.clip(rho, 0, 0.99))   # only positive lag-1 matters here
    pw = x.copy()
    # prewhiten the residuals, keep the trend term
    base = tr["slope"] * t
    res_all = resid
    for i in range(1, len(x)):
        if np.isfinite(res_all[i]) and np.isfinite(res_all[i - 1]):
            pw[i] = res_all[i] - rho * res_all[i - 1] + base[i]
    return pw


def trend_with_prewhitening(x: np.ndarray, prewhiten: bool = False) -> dict:
    """MK + Sen, optionally on Yue-Pilon prewhitened series."""
    if prewhiten:
        x = yue_pilon_prewhiten(x)
    res = sens_slope(x)
    res.update(mann_kendall(x))
    res["prewhitened"] = bool(prewhiten)
    return res


def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR: boolean mask of significant grid cells
    for a spatial field of p-values. Reject H0 for all ranks k = 1..K where
    K is the largest k with p(k) <= k*alpha/n."""
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    ok = np.isfinite(flat)
    out = np.zeros(flat.shape, dtype=bool)
    ps = flat[ok]
    n = len(ps)
    if n == 0:
        return out.reshape(p.shape)
    order = np.argsort(ps)
    thresh = (np.arange(n) + 1) * alpha / n
    passed = ps[order] <= thresh
    idx = np.where(passed)[0]
    if idx.size:
        out[ok] = False
        keep = np.zeros(n, dtype=bool)
        keep[order[:idx[-1] + 1]] = True
        out[ok] = keep
    return out.reshape(p.shape)
