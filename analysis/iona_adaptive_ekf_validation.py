import csv
from pathlib import Path

import numpy as np

SOURCE = Path(
    "results/integrated_navigation/adaptive_ekf_trials.csv"
)

OUTPUT = Path(
    "results/integrated_navigation/adaptive_ekf_statistical_validation.csv"
)

with SOURCE.open("r", encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f))

rng = np.random.default_rng(20261119)
BOOTSTRAP_REPETITIONS = 20000

summary = []

print("\nIONA | PAIRED STATISTICAL VALIDATION")
print("Bootstrap repetitions:", BOOTSTRAP_REPETITIONS)
print("Positive difference = adaptive EKF has lower RMSE")

for scenario in ["quiet", "disturbed"]:
    for error in [0, 5, 10, 25]:

        fixed = {}
        adaptive = {}

        for row in records:
            if (
                row["scenario"] != scenario
                or int(row["tec_error_percent"]) != error
            ):
                continue

            trial = int(row["trial"])
            value = float(row["outage_rmse_m"])

            if row["method"] == "fixed":
                fixed[trial] = value
            else:
                adaptive[trial] = value

        trials = sorted(set(fixed) & set(adaptive))

        differences = np.array([
            fixed[t] - adaptive[t] for t in trials
        ])

        fixed_values = np.array([fixed[t] for t in trials])
        adaptive_values = np.array([adaptive[t] for t in trials])

        mean_difference = float(np.mean(differences))
        median_difference = float(np.median(differences))

        # Paired bootstrap: resample trial differences.
        bootstrap_means = np.empty(BOOTSTRAP_REPETITIONS)

        for b in range(BOOTSTRAP_REPETITIONS):
            sample = rng.choice(
                differences,
                size=len(differences),
                replace=True
            )
            bootstrap_means[b] = np.mean(sample)

        ci_low, ci_high = np.percentile(
            bootstrap_means, [2.5, 97.5]
        )

        fixed_mean = float(np.mean(fixed_values))
        adaptive_mean = float(np.mean(adaptive_values))

        relative_reduction = (
            100 * (fixed_mean - adaptive_mean) / fixed_mean
        )

        better_fraction = float(
            np.mean(differences > 0) * 100
        )

        result = {
            "scenario": scenario,
            "tec_error_percent": error,
            "trials": len(trials),
            "fixed_mean_rmse_m": fixed_mean,
            "adaptive_mean_rmse_m": adaptive_mean,
            "mean_paired_difference_m": mean_difference,
            "median_paired_difference_m": median_difference,
            "bootstrap_ci95_low_m": float(ci_low),
            "bootstrap_ci95_high_m": float(ci_high),
            "relative_mean_reduction_percent": relative_reduction,
            "adaptive_better_trials_percent": better_fraction
        }

        summary.append(result)

        print("\n" + "-" * 90)
        print(f"{scenario.upper()} | TEC uncertainty: {error}%")
        print(f"Paired trials: {len(trials)}")
        print(f"Fixed EKF mean:       {fixed_mean:.4f} m")
        print(f"Adaptive EKF mean:    {adaptive_mean:.4f} m")
        print(f"Mean paired benefit:  {mean_difference:+.4f} m")
        print(f"Median paired benefit:{median_difference:+.4f} m")
        print(
            f"95% bootstrap CI:     "
            f"[{ci_low:+.4f}, {ci_high:+.4f}] m"
        )
        print(
            f"Reduction from means: {relative_reduction:+.2f}%"
        )
        print(
            f"Adaptive better:     {better_fraction:.1f}%"
        )

        if ci_low > 0:
            print("Interpretation: positive paired benefit supported by this CI.")
        elif ci_high < 0:
            print("Interpretation: adaptive filter performed worse in this comparison.")
        else:
            print("Interpretation: CI includes zero; benefit is not conclusive.")

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

print("\nSaved:", OUTPUT.resolve())
print("\nSTATISTICAL VALIDATION COMPLETED")
