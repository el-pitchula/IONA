
from pathlib import Path
import json

import numpy as np
import pandas as pd

from libpysal.weights import lat2W
from esda.moran import Moran_Local


# ============================================================
# IONA | LOCAL INDICATORS OF SPATIAL ASSOCIATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "trends" / "cell_trends.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "trends"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "local_spatial_clusters.csv"
OUTPUT_JSON = OUTPUT_DIR / "local_spatial_summary.json"

PERMUTATIONS = 9999
RANDOM_SEED = 42
ALPHA = 0.05


# ============================================================
# 1. LOAD AND ORGANIZE DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

longitudes = np.sort(df["longitude"].unique())
latitudes = np.sort(df["latitude"].unique())

spatial = (
    df.pivot(
        index="latitude",
        columns="longitude",
        values="sen_slope"
    )
    .reindex(index=latitudes, columns=longitudes)
)

values = spatial.to_numpy().flatten()

if np.isnan(values).any():
    raise ValueError("Missing values in spatial grid.")

n_rows, n_cols = spatial.shape

print("=" * 60)
print("IONA | LOCAL SPATIAL CLUSTER ANALYSIS")
print("=" * 60)

print("Grid:", spatial.shape)
print("Cells:", len(values))


# ============================================================
# 2. QUEEN CONTIGUITY WEIGHTS
# ============================================================

weights = lat2W(
    n_rows,
    n_cols,
    rook=False
)

# Binary weights to maintain consistency
# with the global Queen-contiguity analysis.

weights.transform = "B"

print("Spatial weights: Queen contiguity")
print("Permutations:", PERMUTATIONS)


# ============================================================
# 3. LOCAL MORAN'S I
# ============================================================

print("\nCalculating Local Moran's I...")

lisa = Moran_Local(
    values,
    weights,
    transformation="B",
    permutations=PERMUTATIONS,
    seed=RANDOM_SEED,
    n_jobs=1
)

local_I = lisa.Is
p_values = lisa.p_sim
quadrants = lisa.q


# ============================================================
# 4. BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(p):

    p = np.asarray(p)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / np.arange(1, n + 1)

    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q = np.empty(n)
    q[order] = adjusted

    return q


q_values = benjamini_hochberg(p_values)

significant = q_values < ALPHA


# ============================================================
# 5. CLASSIFY SPATIAL CLUSTERS
# ============================================================

quadrant_names = {
    1: "high-high",
    2: "low-high",
    3: "low-low",
    4: "high-low"
}

clusters = []

for i in range(len(values)):

    if significant[i]:
        clusters.append(quadrant_names[int(quadrants[i])])
    else:
        clusters.append("not_significant")


# ============================================================
# 6. BUILD OUTPUT TABLE
# ============================================================

result = pd.DataFrame({
    "longitude": np.tile(longitudes, n_rows),
    "latitude": np.repeat(latitudes, n_cols),
    "sen_slope": values,
    "local_moran_i": local_I,
    "p_value": p_values,
    "q_value": q_values,
    "quadrant": quadrants,
    "cluster": clusters,
    "significant_fdr": significant
})

result.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# 7. SUMMARY
# ============================================================

counts = result["cluster"].value_counts().to_dict()

summary = {
    "method": "Local Moran's I (LISA)",
    "weights": "Queen contiguity, binary",
    "permutations": PERMUTATIONS,
    "random_seed": RANDOM_SEED,
    "fdr_method": "Benjamini-Hochberg",
    "alpha": ALPHA,
    "cells": len(result),
    "significant_cells": int(significant.sum()),
    "cluster_counts": {
        name: int(counts.get(name, 0))
        for name in [
            "high-high",
            "low-low",
            "high-low",
            "low-high",
            "not_significant"
        ]
    }
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)


# ============================================================
# 8. REPORT
# ============================================================

print("\n" + "=" * 60)
print("IONA | LOCAL SPATIAL RESULTS")
print("=" * 60)

print("Significant cells after FDR:", significant.sum())

for name, count in summary["cluster_counts"].items():
    print(f"{name}: {count}")

print("\nOUTPUT FILES")
print(OUTPUT_CSV)
print(OUTPUT_JSON)

print("\nAnalysis completed.")
