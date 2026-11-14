
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# IONA | GLOBAL MORAN'S I
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "trends" / "cell_trends.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "trends"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "spatial_autocorrelation.json"

PERMUTATIONS = 9999
RANDOM_SEED = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

longitudes = np.sort(df["longitude"].unique())
latitudes = np.sort(df["latitude"].unique())

grid = (
    df.pivot(
        index="latitude",
        columns="longitude",
        values="sen_slope"
    )
    .reindex(index=latitudes, columns=longitudes)
    .to_numpy()
)

if np.isnan(grid).any():
    raise ValueError("Missing values in spatial grid.")

n_rows, n_cols = grid.shape

print("=" * 60)
print("IONA | GLOBAL SPATIAL AUTOCORRELATION")
print("=" * 60)

print("Grid:", grid.shape)
print("Cells:", grid.size)


# ============================================================
# 2. BUILD QUEEN CONTIGUITY WEIGHTS
# ============================================================

# Each cell is connected to its 8 immediate neighbors.
# No self-connections.

source = []
target = []

for r in range(n_rows):
    for c in range(n_cols):

        i = r * n_cols + c

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:

                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                if 0 <= nr < n_rows and 0 <= nc < n_cols:

                    j = nr * n_cols + nc

                    source.append(i)
                    target.append(j)

source = np.array(source)
target = np.array(target)

S0 = len(source)

print("Neighbor relationships:", S0)


# ============================================================
# 3. MORAN'S I FUNCTION
# ============================================================

def morans_i(values):

    z = values - np.mean(values)

    denominator = np.sum(z ** 2)

    numerator = np.sum(
        z[source] * z[target]
    )

    return (len(values) / S0) * (
        numerator / denominator
    )


# ============================================================
# 4. OBSERVED MORAN'S I
# ============================================================

values = grid.flatten()

observed_I = morans_i(values)

expected_I = -1 / (len(values) - 1)


# ============================================================
# 5. PERMUTATION TEST
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

permuted_I = np.empty(PERMUTATIONS)

print("\nRunning permutation test...")

for k in range(PERMUTATIONS):

    shuffled = rng.permutation(values)

    permuted_I[k] = morans_i(shuffled)

    if (k + 1) % 1000 == 0:
        print(f"Permutations: {k + 1}/{PERMUTATIONS}")


# One-sided test for positive spatial autocorrelation.

p_positive = (
    np.sum(permuted_I >= observed_I) + 1
) / (PERMUTATIONS + 1)

# Two-sided descriptive permutation test.

p_two_sided = (
    np.sum(
        np.abs(permuted_I - expected_I)
        >= np.abs(observed_I - expected_I)
    ) + 1
) / (PERMUTATIONS + 1)


# ============================================================
# 6. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("MORAN'S I RESULTS")
print("=" * 60)

print(f"Observed Moran's I: {observed_I:.6f}")
print(f"Expected Moran's I: {expected_I:.6f}")
print(f"Positive-tail permutation p-value: {p_positive:.6f}")
print(f"Two-sided permutation p-value: {p_two_sided:.6f}")

if p_positive < 0.05 and observed_I > expected_I:
    interpretation = "Positive spatial autocorrelation detected"
elif p_two_sided < 0.05:
    interpretation = "Spatial autocorrelation detected"
else:
    interpretation = "No significant global spatial autocorrelation detected"

print("\nInterpretation:", interpretation)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

import json

results = {
    "method": "Global Moran's I",
    "weights": "Queen contiguity (8 neighbors)",
    "permutations": PERMUTATIONS,
    "seed": RANDOM_SEED,
    "observed_moran_i": float(observed_I),
    "expected_moran_i": float(expected_I),
    "p_positive": float(p_positive),
    "p_two_sided": float(p_two_sided),
    "interpretation": interpretation,
    "variable": "Sen's slope",
    "cells": int(len(values))
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\nResults saved to:")
print(OUTPUT_FILE)
