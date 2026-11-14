
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# IONA | SPATIAL TREND MAP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TREND_FILE = ROOT / "data" / "processed" / "trends" / "cell_trends.csv"
COORD_FILE = ROOT / "data" / "processed" / "coordinates.npz"

OUTPUT_DIR = ROOT / "data" / "processed" / "trends"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "spatial_sen_trends.png"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("IONA | SPATIAL TREND VISUALIZATION")
print("=" * 60)

trends = pd.read_csv(TREND_FILE)
coordinates = np.load(COORD_FILE)

longitudes = coordinates["longitude"]
latitudes = coordinates["latitude"]

print(f"Trend records: {len(trends)}")
print(f"Longitude points: {len(longitudes)}")
print(f"Latitude points: {len(latitudes)}")


# ============================================================
# 2. BUILD SPATIAL MATRIX
# ============================================================

# Rows = latitude
# Columns = longitude

spatial = trends.pivot(
    index="latitude",
    columns="longitude",
    values="sen_slope"
)

spatial = spatial.reindex(
    index=latitudes,
    columns=longitudes
)

Z = spatial.to_numpy()

X, Y = np.meshgrid(longitudes, latitudes)

if Z.shape != (len(latitudes), len(longitudes)):
    raise ValueError("Spatial matrix dimensions do not match coordinates.")

if np.isnan(Z).any():
    raise ValueError("Missing values detected in spatial trend matrix.")

print(f"Spatial matrix: {Z.shape}")


# ============================================================
# 3. COLOR SCALE
# ============================================================

max_abs = np.max(np.abs(Z))

norm = TwoSlopeNorm(
    vmin=-max_abs,
    vcenter=0,
    vmax=max_abs
)


# ============================================================
# 4. IDENTIFY POSITIVE CELLS
# ============================================================

positive = trends[trends["sen_slope"] > 0]

print(f"Positive cells: {len(positive)}")
print(f"Negative cells: {(trends['sen_slope'] < 0).sum()}")


# ============================================================
# 5. CREATE MAP
# ============================================================

fig, ax = plt.subplots(figsize=(11, 9))

mesh = ax.pcolormesh(
    X,
    Y,
    Z,
    cmap="RdBu_r",
    norm=norm,
    shading="auto",
    edgecolors="none"
)

# Highlight positive estimated slopes.
# These markers do NOT indicate statistical significance.

ax.scatter(
    positive["longitude"],
    positive["latitude"],
    s=35,
    facecolors="none",
    edgecolors="black",
    linewidths=1.0,
    label="Positive estimated slope (not FDR-significant)",
    zorder=3
)


# ============================================================
# 6. MAP FORMATTING
# ============================================================

colorbar = fig.colorbar(
    mesh,
    ax=ax,
    orientation="vertical",
    pad=0.035,
    shrink=0.85
)

colorbar.set_label(
    "Sen's slope (mm/year per calendar year)",
    fontsize=10
)

ax.set_title(
    "IONA | Spatial Distribution of Precipitation Trends",
    fontsize=14,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Longitude (°)", fontsize=11)
ax.set_ylabel("Latitude (°)", fontsize=11)

ax.set_xlim(longitudes.min() - 0.05, longitudes.max() + 0.05)
ax.set_ylim(latitudes.min() - 0.05, latitudes.max() + 0.05)

ax.set_xticks(np.arange(-48.5, -45.9, 0.25))
ax.set_yticks(np.arange(-16.5, -13.9, 0.25))

ax.tick_params(axis="both", labelsize=8)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

# Approximate geographic aspect correction.
mean_latitude = np.mean(latitudes)
ax.set_aspect(1 / np.cos(np.deg2rad(mean_latitude)))

ax.legend(
    loc="upper right",
    fontsize=8,
    framealpha=0.9
)


# ============================================================
# 7. SCIENTIFIC ANNOTATION
# ============================================================

fig.text(
    0.5,
    0.025,
    "Pilot bounding box | 625 grid cells | 2001–2024\n"
    "No cell passed the Benjamini-Hochberg FDR threshold (q < 0.05). "
    "Slopes are estimates, not confirmed trends.",
    ha="center",
    fontsize=8,
    color="dimgray"
)

plt.tight_layout(rect=[0, 0.07, 1, 1])


# ============================================================
# 8. EXPORT
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print("\nMAP EXPORTED SUCCESSFULLY")
print(OUTPUT_FILE)

print("\nSlope range:")
print(f"Minimum: {Z.min():.4f}")
print(f"Maximum: {Z.max():.4f}")

plt.show()
