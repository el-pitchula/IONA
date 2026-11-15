import csv
from pathlib import Path

import numpy as np
import iona_integrated_navigation as sim

TRIALS = 100
BASE_SEED = 20261118
TEC_ERROR_LEVELS = [0, 5, 10, 25]

OUTPUT = Path("results/integrated_navigation")
OUTPUT.mkdir(parents=True, exist_ok=True)

true_position, _, true_acceleration = sim.make_trajectory()

full_mask = np.ones(sim.N, dtype=bool)
outage_mask = (
    (sim.TIME >= sim.OUTAGE_START)
    & (sim.TIME <= sim.OUTAGE_END)
)

H = sim.H_GNSS
H_INV = np.linalg.inv(H.T @ H) @ H.T

records = []


def simulate_gnss_with_uncertainty(vtec_series, error_percent, seed):
    rng = np.random.default_rng(seed)

    fixes = []
    covariance_series = []

    for j, _ in enumerate(sim.GNSS_TIMES):
        vtec = max(float(vtec_series[j]), 0.1)

        satellite_vtec = vtec * (
            1 + rng.normal(0, 0.04, len(sim.AZIMUTH))
        )
        satellite_vtec = np.maximum(satellite_vtec, 0.1)

        slant_tec = satellite_vtec * sim.MAPPING * 1e16

        true_delay = sim.ionospheric_delay(
            slant_tec, sim.F1
        )

        code_noise = rng.normal(
            0, sim.CODE_NOISE, len(sim.AZIMUTH)
        )

        observed_error = true_delay + code_noise

        level = error_percent / 100.0

        if level == 0:
            estimated_slant_tec = slant_tec.copy()
        else:
            common_error = rng.normal(0, level)
            satellite_error = rng.normal(
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

        position_error = H_INV @ corrected_error
        truth = true_position[sim.GNSS_INDICES[j]]

        fixes.append(truth + position_error[:2])

        # Predicted residual covariance from TEC uncertainty.
        common_cov = (
            level**2 * np.outer(
                estimated_delay, estimated_delay
            )
        )

        independent_cov = (
            (0.5 * level)**2
            * np.diag(estimated_delay**2)
        )

        pseudorange_cov = (
            sim.CODE_NOISE**2 * np.eye(len(sim.AZIMUTH))
            + common_cov
            + independent_cov
        )

        position_cov = (
            H_INV @ pseudorange_cov @ H_INV.T
        )[:2, :2]

        covariance_series.append(position_cov)

    return np.asarray(fixes), np.asarray(covariance_series)


def run_adaptive_ekf(gnss_fixes, covariance_series):
    state = np.zeros(4)
    covariance = np.diag([10.0, 10.0, 4.0, 4.0])

    state[:2] = gnss_fixes[0]

    F = np.array([
        [1, 0, sim.DT, 0],
        [0, 1, 0, sim.DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    B = np.array([
        [0.5 * sim.DT**2, 0],
        [0, 0.5 * sim.DT**2],
        [sim.DT, 0],
        [0, sim.DT]
    ])

    Q = B @ (
        sim.IMU_NOISE**2 * np.eye(2)
    ) @ B.T
    Q += np.eye(4) * 1e-8

    H_STATE = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    estimates = np.zeros((sim.N, 2))

    gnss_lookup = {
        int(index): row
        for row, index in enumerate(sim.GNSS_INDICES)
    }

    for k in range(sim.N):

        if k > 0:
            imu_measurement = (
                true_acceleration[k-1]
                + sim.rng.normal(0, sim.IMU_NOISE, 2)
            )

            state = F @ state + B @ imu_measurement
            covariance = F @ covariance @ F.T + Q

        if k in gnss_lookup:
            j = gnss_lookup[k]

            if not sim.OUTAGE_MASK[j]:
                measurement = gnss_fixes[j]
                R = covariance_series[j]

                innovation = measurement - H_STATE @ state
                S = H_STATE @ covariance @ H_STATE.T + R
                gain = (
                    covariance @ H_STATE.T
                    @ np.linalg.inv(S)
                )

                state = state + gain @ innovation

                identity = np.eye(4)
                covariance = (
                    (identity - gain @ H_STATE)
                    @ covariance
                    @ (identity - gain @ H_STATE).T
                    + gain @ R @ gain.T
                )

        estimates[k] = state[:2]

    return estimates


print("\nIONA | FIXED vs ADAPTIVE EKF")
print("Trials:", TRIALS)

for trial in range(TRIALS):
    for scenario in ["quiet", "disturbed"]:

        vtec = sim.make_tec_scenario(scenario)
        seed = BASE_SEED + trial

        for error_percent in TEC_ERROR_LEVELS:

            fixes, covariance_series = (
                simulate_gnss_with_uncertainty(
                    vtec, error_percent, seed
                )
            )

            # Fixed-R EKF
            sim.rng = np.random.default_rng(seed)
            fixed_estimate = sim.run_ekf(
                fixes, true_acceleration, sim.R_L1
            )

            # Adaptive-R EKF: identical IMU noise sequence
            sim.rng = np.random.default_rng(seed)
            adaptive_estimate = run_adaptive_ekf(
                fixes, covariance_series
            )

            for method, estimate in [
                ("fixed", fixed_estimate),
                ("adaptive", adaptive_estimate)
            ]:
                records.append({
                    "trial": trial + 1,
                    "scenario": scenario,
                    "tec_error_percent": error_percent,
                    "method": method,
                    "full_rmse_m": sim.rmse(
                        estimate, true_position, full_mask
                    ),
                    "outage_rmse_m": sim.rmse(
                        estimate, true_position, outage_mask
                    )
                })

    if (trial + 1) % 10 == 0:
        print(f"Completed {trial + 1}/{TRIALS}")


trial_path = OUTPUT / "adaptive_ekf_trials.csv"

with trial_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)


summary = []

print("\n" + "=" * 105)
print("FIXED vs ADAPTIVE EKF | OUTAGE RMSE")
print("=" * 105)

for scenario in ["quiet", "disturbed"]:
    print(f"\n{scenario.upper()}")
    print("-" * 105)
    print(
        f"{'TEC error':>10s} | {'Method':>10s} | "
        f"{'Mean':>10s} | {'SD':>10s} | {'P95':>10s}"
    )

    for error_percent in TEC_ERROR_LEVELS:
        for method in ["fixed", "adaptive"]:

            selected = [
                row for row in records
                if row["scenario"] == scenario
                and row["tec_error_percent"] == error_percent
                and row["method"] == method
            ]

            full = np.array([
                row["full_rmse_m"] for row in selected
            ])

            outage = np.array([
                row["outage_rmse_m"] for row in selected
            ])

            item = {
                "scenario": scenario,
                "tec_error_percent": error_percent,
                "method": method,
                "full_mean_m": float(np.mean(full)),
                "full_sd_m": float(np.std(full, ddof=1)),
                "outage_mean_m": float(np.mean(outage)),
                "outage_sd_m": float(np.std(outage, ddof=1)),
                "outage_p95_m": float(np.percentile(outage, 95))
            }

            summary.append(item)

            print(
                f"{error_percent:9d}% | {method:>10s} | "
                f"{item['outage_mean_m']:10.3f} | "
                f"{item['outage_sd_m']:10.3f} | "
                f"{item['outage_p95_m']:10.3f}"
            )


print("\n" + "=" * 105)
print("ADAPTIVE EKF BENEFIT")
print("=" * 105)

for scenario in ["quiet", "disturbed"]:
    print(f"\n{scenario.upper()}")

    for error_percent in TEC_ERROR_LEVELS:

        fixed = np.array([
            row["outage_rmse_m"] for row in records
            if row["scenario"] == scenario
            and row["tec_error_percent"] == error_percent
            and row["method"] == "fixed"
        ])

        adaptive = np.array([
            row["outage_rmse_m"] for row in records
            if row["scenario"] == scenario
            and row["tec_error_percent"] == error_percent
            and row["method"] == "adaptive"
        ])

        improvement = fixed - adaptive
        percentage = 100 * improvement / fixed

        print(
            f"{error_percent:2d}% TEC uncertainty | "
            f"Mean RMSE change: {np.mean(improvement):+.3f} m | "
            f"Mean improvement: {np.mean(percentage):+.1f}% | "
            f"Adaptive better: {np.mean(improvement > 0)*100:.1f}%"
        )


summary_path = OUTPUT / "adaptive_ekf_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

print("\nFiles generated:")
print(trial_path.resolve())
print(summary_path.resolve())
print("\nADAPTIVE EKF EXPERIMENT COMPLETED")
