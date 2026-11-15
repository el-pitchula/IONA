import sys
import csv
from pathlib import Path

import numpy as np

# Import the existing integrated experiment.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import iona_integrated_navigation as sim


TRIALS = 100
BASE_SEED = 20261116

OUTPUT = Path("results/integrated_navigation")
OUTPUT.mkdir(parents=True, exist_ok=True)

true_position, true_velocity, true_acceleration = (
    sim.make_trajectory()
)

outage_time_mask = (
    (sim.TIME >= sim.OUTAGE_START)
    & (sim.TIME <= sim.OUTAGE_END)
)

full_mask = np.ones(sim.N, dtype=bool)

records = []

print("\n" + "=" * 100)
print("IONA | INTEGRATED MONTE CARLO VALIDATION")
print("=" * 100)
print(f"Independent trials per scenario: {TRIALS}")
print("Scenarios: quiet and disturbed synthetic TEC")
print("Methods: GNSS, IMU, EKF L1, ideal TEC correction, dual frequency")
print(f"GNSS outage: {sim.OUTAGE_START:.1f} to {sim.OUTAGE_END:.1f} seconds")
print("=" * 100)


for trial in range(TRIALS):

    # Reinitialize the random generator for reproducibility.
    # The same trial seed is used in both scenarios to enable
    # paired comparisons with equivalent noise realizations.

    trial_seed = BASE_SEED + trial

    for scenario in ["quiet", "disturbed"]:

        sim.rng = np.random.default_rng(trial_seed)

        vtec = sim.make_tec_scenario(scenario)

        fixes_l1, fixes_corrected, fixes_if = sim.simulate_gnss(
            true_position, vtec
        )

        ekf_l1 = sim.run_ekf(
            fixes_l1, true_acceleration, sim.R_L1
        )

        ekf_corrected = sim.run_ekf(
            fixes_corrected, true_acceleration, sim.R_L1
        )

        ekf_if = sim.run_ekf(
            fixes_if, true_acceleration, sim.R_IF
        )

        gnss_only = sim.hold_last_gnss(fixes_l1)

        imu_only = sim.imu_dead_reckoning(
            true_acceleration, fixes_l1[0]
        )

        estimates = {
            "GNSS L1 only": gnss_only,
            "IMU only": imu_only,
            "EKF L1": ekf_l1,
            "EKF L1 ideal TEC correction": ekf_corrected,
            "EKF dual-frequency L1+L2": ekf_if
        }

        for method, estimate in estimates.items():

            records.append({
                "trial": trial + 1,
                "scenario": scenario,
                "method": method,
                "full_rmse_m": sim.rmse(
                    estimate, true_position, full_mask
                ),
                "outage_rmse_m": sim.rmse(
                    estimate, true_position, outage_time_mask
                )
            })

    if (trial + 1) % 10 == 0:
        print(f"Completed {trial + 1}/{TRIALS} trials per scenario")


# Save individual trial results.
trial_path = OUTPUT / "monte_carlo_trial_results.csv"

with trial_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)


# Statistical summary.
summary = []

print("\n" + "=" * 112)
print("MONTE CARLO STATISTICAL SUMMARY")
print("=" * 112)

for scenario in ["quiet", "disturbed"]:

    print(f"\nSCENARIO: {scenario.upper()}")
    print("-" * 112)
    print(
        f"{'Method':34s} | "
        f"{'Full mean':>10s} | "
        f"{'Full SD':>9s} | "
        f"{'Full P95':>9s} | "
        f"{'Outage mean':>11s} | "
        f"{'Outage SD':>10s} | "
        f"{'Outage P95':>10s}"
    )
    print("-" * 112)

    for method in [
        "GNSS L1 only",
        "IMU only",
        "EKF L1",
        "EKF L1 ideal TEC correction",
        "EKF dual-frequency L1+L2"
    ]:

        selected = [
            row for row in records
            if row["scenario"] == scenario
            and row["method"] == method
        ]

        full = np.array([
            row["full_rmse_m"] for row in selected
        ])

        outage = np.array([
            row["outage_rmse_m"] for row in selected
        ])

        values = {
            "scenario": scenario,
            "method": method,
            "full_mean_m": float(np.mean(full)),
            "full_sd_m": float(np.std(full, ddof=1)),
            "full_median_m": float(np.median(full)),
            "full_p95_m": float(np.percentile(full, 95)),
            "outage_mean_m": float(np.mean(outage)),
            "outage_sd_m": float(np.std(outage, ddof=1)),
            "outage_median_m": float(np.median(outage)),
            "outage_p95_m": float(np.percentile(outage, 95))
        }

        summary.append(values)

        print(
            f"{method:34s} | "
            f"{values['full_mean_m']:10.3f} | "
            f"{values['full_sd_m']:9.3f} | "
            f"{values['full_p95_m']:9.3f} | "
            f"{values['outage_mean_m']:11.3f} | "
            f"{values['outage_sd_m']:10.3f} | "
            f"{values['outage_p95_m']:10.3f}"
        )


# Compare disturbed versus quiet scenarios for each method.
print("\n" + "=" * 100)
print("DISTURBED vs QUIET | OUTAGE RMSE")
print("=" * 100)

for method in [
    "GNSS L1 only",
    "IMU only",
    "EKF L1",
    "EKF L1 ideal TEC correction",
    "EKF dual-frequency L1+L2"
]:

    quiet = np.array([
        row["outage_rmse_m"] for row in records
        if row["scenario"] == "quiet"
        and row["method"] == method
    ])

    disturbed = np.array([
        row["outage_rmse_m"] for row in records
        if row["scenario"] == "disturbed"
        and row["method"] == method
    ])

    difference = disturbed - quiet

    print(
        f"{method:34s} | "
        f"Difference: {np.mean(difference):+8.3f} m | "
        f"Disturbed worse: {np.mean(difference > 0) * 100:5.1f}%"
    )


summary_path = OUTPUT / "monte_carlo_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)


print("\nFiles generated:")
print(trial_path.resolve())
print(summary_path.resolve())

print("\nMONTE CARLO VALIDATION COMPLETED")
