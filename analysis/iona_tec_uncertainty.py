import csv
from pathlib import Path

import numpy as np
import iona_integrated_navigation as sim

TRIALS = 100
BASE_SEED = 20261117

TEC_ERROR_LEVELS = [0, 5, 10, 25]

OUTPUT = Path("results/integrated_navigation")
OUTPUT.mkdir(parents=True, exist_ok=True)

true_position, _, true_acceleration = sim.make_trajectory()

full_mask = np.ones(sim.N, dtype=bool)

outage_mask = (
    (sim.TIME >= sim.OUTAGE_START)
    & (sim.TIME <= sim.OUTAGE_END)
)

records = []


def simulate_with_tec_uncertainty(vtec_series, error_percent):
    """Simulate L1 GNSS with imperfect TEC-based correction."""

    fixes = []

    for j, _ in enumerate(sim.GNSS_TIMES):
        vtec = max(float(vtec_series[j]), 0.1)

        satellite_vtec = vtec * (
            1 + sim.rng.normal(0, 0.04, len(sim.AZIMUTH))
        )
        satellite_vtec = np.maximum(satellite_vtec, 0.1)

        slant_tec = satellite_vtec * sim.MAPPING * 1e16

        true_delay = sim.ionospheric_delay(
            slant_tec, sim.F1
        )

        measurement_noise = sim.rng.normal(
            0, sim.CODE_NOISE, len(sim.AZIMUTH)
        )

        observed_error = true_delay + measurement_noise

        # The TEC estimate contains a common epoch error
        # and an independent satellite-specific error.
        if error_percent == 0:
            estimated_slant_tec = slant_tec.copy()
        else:
            level = error_percent / 100.0

            common_error = sim.rng.normal(0, level)
            satellite_error = sim.rng.normal(
                0, level * 0.5, len(sim.AZIMUTH)
            )

            relative_error = common_error + satellite_error

            estimated_slant_tec = slant_tec * (
                1 + relative_error
            )

            estimated_slant_tec = np.maximum(
                estimated_slant_tec, 0
            )

        estimated_delay = sim.ionospheric_delay(
            estimated_slant_tec, sim.F1
        )

        corrected_error = observed_error - estimated_delay

        solution = np.linalg.lstsq(
            sim.H_GNSS, corrected_error, rcond=None
        )[0]

        truth = true_position[sim.GNSS_INDICES[j]]

        fixes.append(truth + solution[:2])

    return np.asarray(fixes)


print("\n" + "=" * 100)
print("IONA | TEC ESTIMATION UNCERTAINTY EXPERIMENT")
print("=" * 100)
print(f"Trials per scenario and TEC condition: {TRIALS}")
print("TEC estimation error levels:", TEC_ERROR_LEVELS)
print("Scenarios: quiet and disturbed synthetic TEC")
print("=" * 100)


for trial in range(TRIALS):

    for scenario in ["quiet", "disturbed"]:

        vtec = sim.make_tec_scenario(scenario)

        # Same random seed across TEC conditions in a trial.
        # This creates paired comparisons.
        trial_seed = BASE_SEED + trial

        for error_percent in TEC_ERROR_LEVELS:

            sim.rng = np.random.default_rng(trial_seed)

            fixes = simulate_with_tec_uncertainty(
                vtec, error_percent
            )

            estimate = sim.run_ekf(
                fixes,
                true_acceleration,
                sim.R_L1
            )

            records.append({
                "trial": trial + 1,
                "scenario": scenario,
                "tec_error_percent": error_percent,
                "full_rmse_m": sim.rmse(
                    estimate, true_position, full_mask
                ),
                "outage_rmse_m": sim.rmse(
                    estimate, true_position, outage_mask
                )
            })

    if (trial + 1) % 10 == 0:
        print(f"Completed {trial + 1}/{TRIALS} trials")


trial_path = OUTPUT / "tec_uncertainty_trials.csv"

with trial_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)


summary = []

print("\n" + "=" * 105)
print("TEC UNCERTAINTY | STATISTICAL SUMMARY")
print("=" * 105)

for scenario in ["quiet", "disturbed"]:

    print(f"\nSCENARIO: {scenario.upper()}")
    print("-" * 105)
    print(
        f"{'TEC error':>10s} | "
        f"{'Full mean':>10s} | "
        f"{'Full SD':>9s} | "
        f"{'Full P95':>9s} | "
        f"{'Outage mean':>11s} | "
        f"{'Outage SD':>10s} | "
        f"{'Outage P95':>10s}"
    )
    print("-" * 105)

    for error_percent in TEC_ERROR_LEVELS:

        selected = [
            row for row in records
            if row["scenario"] == scenario
            and row["tec_error_percent"] == error_percent
        ]

        full = np.array([
            row["full_rmse_m"] for row in selected
        ])

        outage = np.array([
            row["outage_rmse_m"] for row in selected
        ])

        row = {
            "scenario": scenario,
            "tec_error_percent": error_percent,
            "full_mean_m": float(np.mean(full)),
            "full_sd_m": float(np.std(full, ddof=1)),
            "full_p95_m": float(np.percentile(full, 95)),
            "outage_mean_m": float(np.mean(outage)),
            "outage_sd_m": float(np.std(outage, ddof=1)),
            "outage_p95_m": float(np.percentile(outage, 95))
        }

        summary.append(row)

        print(
            f"{error_percent:9.0f}% | "
            f"{row['full_mean_m']:10.3f} | "
            f"{row['full_sd_m']:9.3f} | "
            f"{row['full_p95_m']:9.3f} | "
            f"{row['outage_mean_m']:11.3f} | "
            f"{row['outage_sd_m']:10.3f} | "
            f"{row['outage_p95_m']:10.3f}"
        )


print("\n" + "=" * 100)
print("EFFECT OF TEC ESTIMATION ERROR")
print("=" * 100)

for scenario in ["quiet", "disturbed"]:

    baseline = np.array([
        row["outage_rmse_m"] for row in records
        if row["scenario"] == scenario
        and row["tec_error_percent"] == 0
    ])

    print(f"\n{scenario.upper()}:")

    for error_percent in [5, 10, 25]:

        comparison = np.array([
            row["outage_rmse_m"] for row in records
            if row["scenario"] == scenario
            and row["tec_error_percent"] == error_percent
        ])

        difference = comparison - baseline

        print(
            f"TEC error {error_percent:2d}% | "
            f"RMSE change: {np.mean(difference):+.3f} m | "
            f"Worse than ideal: "
            f"{np.mean(difference > 0) * 100:.1f}%"
        )


summary_path = OUTPUT / "tec_uncertainty_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)


print("\nFiles generated:")
print(trial_path.resolve())
print(summary_path.resolve())

print("\nTEC UNCERTAINTY EXPERIMENT COMPLETED")
