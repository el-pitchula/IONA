
from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/trends/local_spatial_clusters.csv"
OUTPUT = ROOT / "data/processed/trends/cluster_contrast.json"

N_PERMUTATIONS = 9999
SEED = 42

df = pd.read_csv(INPUT)

hh = df.loc[df["cluster"] == "high-high", "sen_slope"].to_numpy()
ll = df.loc[df["cluster"] == "low-low", "sen_slope"].to_numpy()

if len(hh) == 0 or len(ll) == 0:
    raise ValueError("HH or LL group is empty.")

# Observed difference: HH minus LL
observed = np.mean(hh) - np.mean(ll)

# Descriptive permutation test.
# This does not account for spatial dependence or
# the data-driven construction of the LISA groups.

combined = np.concatenate([hh, ll])
n_hh = len(hh)

rng = np.random.default_rng(SEED)
permuted = np.empty(N_PERMUTATIONS)

for i in range(N_PERMUTATIONS):
    shuffled = rng.permutation(combined)
    permuted[i] = (
        shuffled[:n_hh].mean()
        - shuffled[n_hh:].mean()
    )

p_value = (
    np.sum(np.abs(permuted) >= abs(observed)) + 1
) / (N_PERMUTATIONS + 1)

results = {
    "comparison": "high-high minus low-low",
    "n_high_high": int(len(hh)),
    "n_low_low": int(len(ll)),
    "mean_high_high": float(np.mean(hh)),
    "mean_low_low": float(np.mean(ll)),
    "median_high_high": float(np.median(hh)),
    "median_low_low": float(np.median(ll)),
    "mean_difference": float(observed),
    "permutation_p_value": float(p_value),
    "permutations": N_PERMUTATIONS,
    "seed": SEED,
    "interpretation": (
        "Exploratory contrast; permutation test does not "
        "account for spatial dependence or data-driven LISA grouping."
    )
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("=" * 60)
print("IONA | HH vs LL CONTRAST")
print("=" * 60)

for key, value in results.items():
    print(f"{key}: {value}")

print("\nSaved:", OUTPUT)
