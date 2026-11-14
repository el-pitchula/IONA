
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# IONA | SCIENTIFIC FIGURES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "docs" / "figuras"

OUTPUT.mkdir(parents=True, exist_ok=True)

DPI = 300

plt.rcParams.update({
    "figure.figsize": (11, 7),
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "DejaVu Sans"
})


def save_figure(fig, filename):
    path = OUTPUT / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def grid_edges(values):
    values = np.asarray(values, dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2

    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])

    return np.concatenate([[first], midpoints, [last]])


def load_spatial_grid(path, column):
    df = pd.read_csv(path)

    grid = (
        df.pivot(
            index="latitude",
            columns="longitude",
            values=column
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if grid.isna().any().any():
        raise ValueError(
            f"Missing values found in spatial variable: {column}"
        )

    return grid


def spatial_axes(ax, lon, lat):
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(lon.min(), lon.max(), 6))
    ax.set_yticks(np.linspace(lat.min(), lat.max(), 6))


# ============================================================
# 1. ANNUAL PRECIPITATION
# ============================================================

annual_path = DATA / "exploratory" / "annual_precipitation.csv"

annual = pd.read_csv(annual_path)

years = annual["year"].to_numpy()
precipitation = annual["annual_precipitation_mm"].to_numpy()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    years,
    precipitation,
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="Precipitação anual"
)

ax.set_title(
    "IONA | Annual Regional Precipitation (2001–2024)"
)

ax.set_xlabel("Year")
ax.set_ylabel("Annual precipitation (mm)")

ax.set_xticks(years[::2])
ax.tick_params(axis="x", rotation=45)

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

save_figure(fig, "01_annual_precipitation.png")


# ============================================================
# 2. MONTHLY CLIMATOLOGY
# ============================================================

climatology_path = (
    DATA / "exploratory" / "monthly_climatology.csv"
)

climatology = pd.read_csv(climatology_path)

months = climatology["month"].to_numpy()
monthly_mean = climatology[
    "mean_precipitation_mm"
].to_numpy()

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    months,
    monthly_mean,
    width=0.7,
    edgecolor="black",
    linewidth=0.5
)

ax.set_title(
    "IONA | Monthly Precipitation Climatology (2001–2024)"
)

ax.set_xlabel("Month")
ax.set_ylabel("Mean precipitation (mm/month)")

ax.set_xticks(months)
ax.set_xticklabels(month_names)

ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

for bar, value in zip(bars, monthly_mean):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

fig.tight_layout()

save_figure(fig, "02_monthly_climatology.png")


# ============================================================
# 3. SPATIAL SEN SLOPE MAP
# ============================================================

trend_path = DATA / "trends" / "cell_trends.csv"

trend_grid = load_spatial_grid(
    trend_path,
    "sen_slope"
)

lon = trend_grid.columns.to_numpy(dtype=float)
lat = trend_grid.index.to_numpy(dtype=float)

lon_edges = grid_edges(lon)
lat_edges = grid_edges(lat)

values = trend_grid.to_numpy()

fig, ax = plt.subplots(figsize=(10, 8))

mesh = ax.pcolormesh(
    lon_edges,
    lat_edges,
    values,
    cmap="RdBu_r",
    shading="flat",
    vmin=-18,
    vmax=18
)

colorbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
colorbar.set_label(
    "Sen's slope (mm/year of annual precipitation per year)"
)

ax.set_title(
    "IONA | Spatial Distribution of Sen's Slope"
)

spatial_axes(ax, lon, lat)

fig.tight_layout()

save_figure(fig, "03_spatial_sen_trends.png")


# ============================================================
# 4. LOCAL MORAN / LISA MAP
# ============================================================

lisa_path = DATA / "trends" / "local_spatial_clusters.csv"

lisa_grid = load_spatial_grid(
    lisa_path,
    "cluster"
)

categories = [
    "not_significant",
    "high-high",
    "low-low",
    "high-low",
    "low-high"
]

category_codes = {
    category: index
    for index, category in enumerate(categories)
}

lisa_values = lisa_grid.replace(category_codes).to_numpy()

lisa_cmap = ListedColormap([
    "#D9D9D9",  # Not significant
    "#D73027",  # High-High
    "#4575B4",  # Low-Low
    "#FDAE61",  # High-Low
    "#ABD9E9"   # Low-High
])

lisa_norm = BoundaryNorm(
    np.arange(-0.5, len(categories) + 0.5, 1),
    lisa_cmap.N
)

fig, ax = plt.subplots(figsize=(10, 8))

mesh = ax.pcolormesh(
    lon_edges,
    lat_edges,
    lisa_values,
    cmap=lisa_cmap,
    norm=lisa_norm,
    shading="flat"
)

colorbar = fig.colorbar(
    mesh,
    ax=ax,
    ticks=np.arange(len(categories)),
    shrink=0.85
)

colorbar.ax.set_yticklabels([
    "Not significant",
    "High-High",
    "Low-Low",
    "High-Low",
    "Low-High"
])

ax.set_title(
    "IONA | Local Spatial Autocorrelation (Queen)"
)

spatial_axes(ax, lon, lat)

fig.tight_layout()

save_figure(fig, "04_lisa_clusters_queen.png")


# ============================================================
# 5. QUEEN VS ROOK ROBUSTNESS
# ============================================================

comparison_path = (
    DATA / "trends" / "robustness"
    / "queen_rook_cell_comparison.csv"
)

comparison = pd.read_csv(comparison_path)

queen_grid = (
    comparison.pivot(
        index="latitude",
        columns="longitude",
        values="queen_cluster"
    )
    .sort_index()
    .sort_index(axis=1)
)

rook_grid = (
    comparison.pivot(
        index="latitude",
        columns="longitude",
        values="rook_cluster"
    )
    .sort_index()
    .sort_index(axis=1)
)

if not queen_grid.index.equals(rook_grid.index):
    raise ValueError("Queen and Rook latitude grids differ.")

if not queen_grid.columns.equals(rook_grid.columns):
    raise ValueError("Queen and Rook longitude grids differ.")

queen_values = queen_grid.replace(category_codes).to_numpy()
rook_values = rook_grid.replace(category_codes).to_numpy()

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7),
    sharex=True,
    sharey=True
)

for ax, values, title in [
    (axes[0], queen_values, "Queen contiguity"),
    (axes[1], rook_values, "Rook contiguity")
]:
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        values,
        cmap=lisa_cmap,
        norm=lisa_norm,
        shading="flat"
    )

    ax.set_title(title)
    spatial_axes(ax, lon, lat)

colorbar = fig.colorbar(
    mesh,
    ax=axes,
    ticks=np.arange(len(categories)),
    shrink=0.8
)

colorbar.ax.set_yticklabels([
    "Not significant",
    "High-High",
    "Low-Low",
    "High-Low",
    "Low-High"
])

fig.suptitle(
    "IONA | Spatial Robustness: Queen vs Rook",
    fontsize=16
)

fig.subplots_adjust(
    top=0.88,
    wspace=0.15
)

save_figure(fig, "05_queen_vs_rook.png")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("IONA | SCIENTIFIC FIGURES COMPLETE")
print("=" * 60)

print("Output directory:", OUTPUT)
print("\nGenerated figures:")

for filename in sorted(OUTPUT.glob("*.png")):
    print("-", filename.name)

print("\nTotal figures:", len(list(OUTPUT.glob("*.png"))))
