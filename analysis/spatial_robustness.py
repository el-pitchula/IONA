
from pathlib import Path
import json

import numpy as np
import pandas as pd
from libpysal.weights import lat2W
from esda.moran import Moran, Moran_Local


# ============================================================
# IONA | SPATIAL ROBUSTNESS TEST
# Queen vs Rook spatial weights
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/trends/cell_trends.csv"
OUTPUT_DIR = ROOT / "data/processed/trends/robustness"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 9999
SEED = 42


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)
    result[order] = adjusted

    return result


def analyze_weights(values, n_rows, n_cols, rook, label):

    print("\n" + "=" * 60)
    print(f"IONA | {label.upper()} WEIGHTS")
    print("=" * 60)

    weights = lat2W(
        n_rows,
        n_cols,
        rook=rook
    )

    weights.transform = "B"

    print("Cells:", len(values))
    print("Neighbors:", "Rook" if rook else "Queen")
    print("Spatial connections:", weights.s0)

    # --------------------------------------------------------
    # GLOBAL MORAN'S I
    # --------------------------------------------------------

    global_moran = Moran(
        values,
        weights,
        transformation="B",
        permutations=N_PERMUTATIONS
    )

    print("\nGLOBAL MORAN'S I")
    print("Observed I:", global_moran.I)
    print("Expected I:", global_moran.EI)
    print("Permutation p-value:", global_moran.p_sim)

    # --------------------------------------------------------
    # LOCAL MORAN / LISA
    # --------------------------------------------------------

    local_moran = Moran_Local(
        values,
        weights,
        transformation="B",
        permutations=N_PERMUTATIONS,
        seed=SEED,
        n_jobs=1
    )

    p_values = local_moran.p_sim
    q_values = benjamini_hochberg(p_values)

    significant = q_values < 0.05

    # PySAL quadrants:
    # 1 = High-High
    # 2 = Low-High
    # 3 = Low-Low
    # 4 = High-Low

    cluster = np.full(len(values), "not_significant", dtype=object)

    cluster[(significant) & (local_moran.q == 1)] = "high-high"
    cluster[(significant) & (local_moran.q == 2)] = "low-high"
    cluster[(significant) & (local_moran.q == 3)] = "low-low"
    cluster[(significant) & (local_moran.q == 4)] = "high-low"

    results = {
        "weights": label,
        "global_moran_i": float(global_moran.I),
        "global_expected_i": float(global_moran.EI),
        "global_p_value": float(global_moran.p_sim),
        "significant_cells_fdr": int(significant.sum()),
        "high_high": int(np.sum(cluster == "high-high")),
        "low_low": int(np.sum(cluster == "low-low")),
        "high_low": int(np.sum(cluster == "high-low")),
        "low_high": int(np.sum(cluster == "low-high")),
        "not_significant": int(np.sum(cluster == "not_significant"))
    }

    print("\nLOCAL MORAN / LISA")
    for key, value in results.items():
        if key != "weights":
            print(f"{key}: {value}")

    return results, cluster, q_values


# ============================================================
# LOAD AND ORDER DATA
# ============================================================

df = pd.read_csv(INPUT)

grid = (
    df.pivot(
        index="latitude",
        columns="longitude",
        values="sen_slope"
    )
    .sort_index()
    .sort_index(axis=1)
)

if grid.isna().any().any():
    raise ValueError("Missing values found in the spatial grid.")

values = grid.to_numpy().flatten(order="C")

n_rows, n_cols = grid.shape

print("=" * 60)
print("IONA | SPATIAL ROBUSTNESS ANALYSIS")
print("=" * 60)

print("Grid:", grid.shape)
print("Cells:", len(values))
print("Permutations:", N_PERMUTATIONS)


# ============================================================
# QUEEN ANALYSIS
# ============================================================

queen_results, queen_clusters, queen_q = analyze_weights(
    values,
    n_rows,
    n_cols,
    rook=False,
    label="Queen"
)


# ============================================================
# ROOK ANALYSIS
# ============================================================

rook_results, rook_clusters, rook_q = analyze_weights(
    values,
    n_rows,
    n_cols,
    rook=True,
    label="Rook"
)


# ============================================================
# COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "longitude": grid.columns.to_numpy().repeat(n_rows),
    "latitude": np.tile(grid.index.to_numpy(), n_cols),
    "sen_slope": values,
    "queen_cluster": queen_clusters,
    "rook_cluster": rook_clusters,
    "queen_q_value": queen_q,
    "rook_q_value": rook_q
})

# Agreement across all classifications
agreement = np.mean(queen_clusters == rook_clusters)

# Jaccard similarity for each significant cluster
def jaccard(a, b):
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    return float(np.sum(a & b) / union)


queen_hh = queen_clusters == "high-high"
rook_hh = rook_clusters == "high-high"

queen_ll = queen_clusters == "low-low"
rook_ll = rook_clusters == "low-low"

queen_sig = queen_clusters != "not_significant"
rook_sig = rook_clusters != "not_significant"

comparison_summary = {
    "queen": queen_results,
    "rook": rook_results,
    "classification_agreement": float(agreement),
    "significant_cluster_jaccard": jaccard(queen_sig, rook_sig),
    "high_high_jaccard": jaccard(queen_hh, rook_hh),
    "low_low_jaccard": jaccard(queen_ll, rook_ll),
    "interpretation": (
        "Sensitivity analysis comparing spatial classifications "
        "under Queen and Rook contiguity. Agreement measures "
        "classification stability, not temporal trend significance."
    )
}


# ============================================================
# SAVE OUTPUTS
# ============================================================

comparison.to_csv(
    OUTPUT_DIR / "queen_rook_cell_comparison.csv",
    index=False
)

with open(
    OUTPUT_DIR / "queen_rook_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        comparison_summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("IONA | QUEEN vs ROOK COMPARISON")
print("=" * 60)

print("Classification agreement:", agreement)
print("Significant cluster Jaccard:", comparison_summary["significant_cluster_jaccard"])
print("HH Jaccard:", comparison_summary["high_high_jaccard"])
print("LL Jaccard:", comparison_summary["low_low_jaccard"])

print("\nSaved files:")
print(OUTPUT_DIR / "queen_rook_cell_comparison.csv")
print(OUTPUT_DIR / "queen_rook_summary.json")
