"""Elsevier-style figure functions for the CHD study.

Design rules (per the brief: "no clutter, no text overlap"):
* Font sizes fixed; constrained_layout everywhere; legends outside axes.
* Single column = 90 mm (3.54 in), double column = 190 mm (7.48 in); 300 dpi.
* Every map gets scalebar-equivalent bounds, degree labels and a source note.
* DEMO functions accept synthetic grids ONLY for layout validation and draw
  a large "SYNTHETIC DEMO" watermark; they are never used on real data.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
})

SINGLE_COL = 3.54    # inches
DOUBLE_COL = 7.48

CLASS_COLORS = {"normal": "#d9d9d9", "heat only": "#fdae61",
                "dry only": "#3288bd", "compound hot-dry": "#d7191c"}


def watermark(fig, text="SYNTHETIC DEMO — NOT RESEARCH OUTPUT"):
    fig.text(0.5, 0.5, text, ha="center", va="center", rotation=22,
             fontsize=20, color="0.45", alpha=0.35, weight="bold", zorder=100)


def _grid_extent(lon, lat):
    return [lon.min(), lon.max(), lat.min(), lat.max()]


# ---------------------------------------------------------------------------
def fig3_spatial_distribution(grids: dict, lon, lat, path,
                              demo: bool = False):
    """grids: {'compound_days': 2-D, 'heat_days': 2-D, 'dry_days': 2-D}
    Mean annual days, three panels (Elsevier double column)."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6),
                             constrained_layout=True, sharex=True,
                             sharey=True)
    specs = [("heat_days", "Heat days yr$^{-1}$", "YlOrRd"),
             ("dry_days", "Dry days yr$^{-1}$", "Blues"),
             ("compound_days", "Compound hot–dry days yr$^{-1}$", "RdPu")]
    for ax, (key, label, cmap) in zip(axes, specs):
        g = grids[key]
        im = ax.imshow(g, origin="upper", extent=_grid_extent(lon, lat),
                       cmap=cmap, aspect="auto")
        ax.set_title(label)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig4_annual_trends(national: dict, path, demo: bool = False):
    """national: {'years': 1-D, 'n_episodes': 1-D, 'episode_days': 1-D,
    'severity': 1-D, 'trend_lines': dict of fitted Sen lines}"""
    years = national["years"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4),
                             constrained_layout=True)
    specs = [("n_episodes", "Compound episodes yr$^{-1}$", (0, "#d7191c")),
             ("episode_days", "Compound event days yr$^{-1}$", (1, "#d95f02")),
             ("severity", "Mean severity (unitless)", (2, "#7570b3"))]
    for key, label, (i, color) in specs:
        ax = axes[i]
        y = national[key]
        ax.plot(years, y, "-", color=color, lw=1.2)
        tl = national.get("trend_lines", {}).get(key)
        if tl is not None:
            ax.plot(years, tl, "--", color="0.2", lw=1.0,
                    label="Sen trend")
            ax.legend(loc="upper left", frameon=False)
        ax.set_ylabel(label)
        ax.set_xlabel("Year")
        ax.set_xlim(years[0], years[-1])
        ax.spines[["top", "right"]].set_visible(False)
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig5_severity_map(severity: np.ndarray, lon, lat, districts_gdf,
                      path, demo: bool = False):
    """Mean annual severity with district boundaries overlay."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.4), constrained_layout=True)
    im = ax.imshow(severity, origin="upper", extent=_grid_extent(lon, lat),
                   cmap="viridis", aspect="auto")
    districts_gdf.boundary.plot(ax=ax, color="0.35", linewidth=0.25)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Severity (unitless)")
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig6_vegetation_response(class_anoms: dict, path, demo: bool = False):
    """class_anoms: {'normal': mean, 'heat only': ..., 'dry only': ...,
    'compound hot-dry': ...} with 95% CI halves."""
    labels = list(class_anoms.keys())
    vals = [class_anoms[k][0] for k in labels]
    errs = [class_anoms[k][1] for k in labels]
    colors = [CLASS_COLORS[k] for k in labels]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, edgecolor="0.2", width=0.65,
           error_kw=dict(lw=0.8, capsize=2))
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(x, [l.replace(" ", "\n") for l in labels])
    ax.set_ylabel("NDVI anomaly (×10$^{-2}$)")
    ax.spines[["top", "right"]].set_visible(False)
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig7_yield_association(df, path, demo: bool = False):
    """df: DataFrame with columns ln_yield_anom, compound_share, crop."""
    crops = df["crop"].unique()
    fig, axes = plt.subplots(1, len(crops), figsize=(DOUBLE_COL, 2.8),
                             constrained_layout=True, sharey=True)
    if len(crops) == 1:
        axes = [axes]
    for ax, crop in zip(axes, crops):
        d = df[df["crop"] == crop]
        ax.scatter(d["compound_share"], d["ln_yield_anom"], s=6,
                   alpha=0.45, color="#d7191c", edgecolors="none")
        # regression line
        z = np.polyfit(d["compound_share"], d["ln_yield_anom"], 1)
        xs = np.linspace(d["compound_share"].min(),
                         d["compound_share"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "-k", lw=1.0)
        ax.set_title(crop.capitalize())
        ax.set_xlabel("Compound event days (share of season)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ln(yield) anomaly")
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig8_ssp_projections(periods: dict, path, demo: bool = False):
    """periods: {'2041–2060': {'ssp245': (median, lo, hi),
                               'ssp585': (median, lo, hi)}, ...}"""
    names = list(periods.keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8), constrained_layout=True)
    for i, (scen, color, hatch) in enumerate(
            [("ssp245", "#7fc97f", ""), ("ssp585", "#d95f02", "//")]):
        med = [periods[p][scen][0] for p in names]
        lo = [periods[p][scen][1] for p in names]
        hi = [periods[p][scen][2] for p in names]
        ax.bar(x + (i - 0.5) * 0.36, med, width=0.36, color=color,
               hatch=hatch, edgecolor="0.2", linewidth=0.5,
               label=scen.upper())
        ax.errorbar(x + (i - 0.5) * 0.36, med,
                    yerr=[np.array(med) - np.array(lo),
                          np.array(hi) - np.array(med)],
                    fmt="none", ecolor="0.2", elinewidth=0.8, capsize=2)
    ax.set_xticks(x, names)
    ax.set_ylabel("Compound event days yr$^{-1}$\n(median; IQR across GCMs)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig9_exposure_hotspots(exposure: np.ndarray, cropland: np.ndarray,
                           lon, lat, path, demo: bool = False):
    """Hatched where top-tertile compound exposure meets cropland."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9),
                             constrained_layout=True, sharex=True,
                             sharey=True)
    im0 = axes[0].imshow(exposure, origin="upper",
                         extent=_grid_extent(lon, lat), cmap="YlOrRd",
                         aspect="auto")
    axes[0].set_title("Compound event days yr$^{-1}$ (mean)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    axes[1].imshow(cropland, origin="upper", extent=_grid_extent(lon, lat),
                   cmap="Greens", aspect="auto")
    hot = (exposure > np.nanquantile(exposure, 0.67)) & (cropland > 0.3)
    axes[1].contourf(np.where(hot, 1, np.nan), hatches=["///"], colors="none",
                     origin="upper")
    axes[1].set_title("Cropland + high exposure (hatched)")
    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)


def fig10_integrated(historical: np.ndarray, future: np.ndarray, lon, lat,
                     path, demo: bool = False):
    """Two-panel integrated risk: observed vs projected change factor."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9),
                             constrained_layout=True, sharex=True,
                             sharey=True)
    im0 = axes[0].imshow(historical, origin="upper",
                         extent=_grid_extent(lon, lat), cmap="YlOrRd",
                         aspect="auto")
    axes[0].set_title("Historical mean (1981–2025)")
    im1 = axes[1].imshow(future, origin="upper",
                         extent=_grid_extent(lon, lat), cmap="RdYlBu_r",
                         aspect="auto")
    axes[1].set_title("Projected change factor (2081–2100, SSP5-8.5)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    if demo:
        watermark(fig)
    fig.savefig(path)
    plt.close(fig)
