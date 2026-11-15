import csv
from pathlib import Path

import numpy as np
import iona_integrated_navigation as sim


TRIALS = 100
BASE_SEED = 20261118
TEC_ERROR_PERCENT = 25
SCENARIO = "disturbed"

OUTPUT = Path("results/integrated_navigation")
OUTPUT.mkdir(parents=True, exist_ok=True)

true_position, true_velocity, true_acceleration = sim.make_trajectory()

true_state = np.column_stack((
    true_position[:, 0],
    true_position[:, 1],
    true_velocity[:, 0],
    true_velocity[:, 1]
))

H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])

F = np.array([
    [1, 0, sim.DT, 0],
    [0, 1, 0, sim.DT],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

B = np.array([
    [0.5 * sim.DT**2, 0],
    [0, 0.5 * sim.DT**2],
    [sim.DT, 0],
    [0, sim.DT]
], dtype=float)

Q = B @ (sim.IMU_NOISE**2 * np.eye(2)) @ B.T
Q += np.eye(4) * 1e-8

IDENTITY = np.eye(4)

# Pointwise 95% chi-square interval, 2 degrees of freedom.
CHI2_LOW = 0.0506
CHI2_HIGH = 7.3778


def simulate_gnss(vtec_series, error_percent, seed):
    rng = np.random.default_rng(seed)

    fixes = []
    covariance_series = []

    H_GNSS = sim.H_GNSS
    H_INV = np.linalg.inv(H_GNSS.T @ H_GNSS) @ H_GNSS.T

    for j, _ in enumerate(sim.GNSS_TIMES):
        vtec = max(float(vtec_series[j]), 0.1)

        satellite_vtec = vtec * (
            1 + rng.normal(0, 0.04, len(sim.AZIMUTH))
        )
        satellite_vtec = np.maximum(satellite_vtec, 0.1)

        slant_tec = satellite_vtec * sim.MAPPING * 1e16

        true_delay = sim.ionospheric_delay(slant_tec, sim.F1)

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

        common_cov = (
            level**2 * np.outer(estimated_delay, estimated_delay)
        )

        independent_cov = (
            (0.5 * level)**2 * np.diag(estimated_delay**2)
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


def run_filter(fixes, covariance_series, seed):
    state = np.zeros(4)

    covariance = np.zeros((4, 4))
    covariance[:2, :2] = covariance_series[0]
    covariance[2:, 2:] = np.diag([4.0, 4.0])

    state[:2] = fixes[0]

    estimates = np.zeros((sim.N, 4))
    covariances = np.zeros((sim.N, 4, 4))

    gnss_lookup = {
        int(index): row
        for row, index in enumerate(sim.GNSS_INDICES)
    }

    rng = np.random.default_rng(seed)

    for k in range(sim.N):

        if k > 0:
            imu_measurement = (
                true_acceleration[k - 1]
                + rng.normal(0, sim.IMU_NOISE, 2)
            )

            state = F @ state + B @ imu_measurement
            covariance = F @ covariance @ F.T + Q

        if k in gnss_lookup:
            j = gnss_lookup[k]

            if k > 0 and not sim.OUTAGE_MASK[j]:
                measurement = fixes[j]
                R = covariance_series[j]

                innovation = measurement - H @ state
                S = H @ covariance @ H.T + R

                gain = np.linalg.solve(
                    S, H @ covariance
                ).T

                state = state + gain @ innovation

                covariance = (
                    (IDENTITY - gain @ H)
                    @ covariance
                    @ (IDENTITY - gain @ H).T
                    + gain @ R @ gain.T
                )

        estimates[k] = state
        covariances[k] = covariance

    return estimates, covariances


def safe_nees(error, covariance):
    return float(
        error @ np.linalg.solve(covariance, error)
    )


def phase_mask(phase):
    if phase == "pre":
        return sim.TIME < sim.OUTAGE_START
    if phase == "outage":
        return (
            (sim.TIME >= sim.OUTAGE_START)
            & (sim.TIME <= sim.OUTAGE_END)
        )
    return sim.TIME > sim.OUTAGE_END


def main():
    vtec = sim.make_tec_scenario(SCENARIO)
    trial_records = []

    print("IONA | VELOCITY CONSISTENCY DIAGNOSTIC")
    print("Trials:", TRIALS)
    print("Scenario:", SCENARIO)
    print("TEC uncertainty:", TEC_ERROR_PERCENT, "%")

    for trial in range(TRIALS):
        seed = BASE_SEED + trial

        fixes, covariance_series = simulate_gnss(
            vtec, TEC_ERROR_PERCENT, seed
        )

        estimates, covariances = run_filter(
            fixes,
            covariance_series,
            seed + 1000003
        )

        errors = estimates - true_state

        for phase in ["pre", "outage", "post"]:
            mask = phase_mask(phase)

            marginal_values = []
            conditional_values = []
            position_values = []

            for k in np.where(mask)[0]:
                P = covariances[k]
                error = errors[k]

                P_pp = P[:2, :2]
                P_pv = P[:2, 2:]
                P_vp = P[2:, :2]
                P_vv = P[2:, 2:]

                e_p = error[:2]
                e_v = error[2:]

                # Marginal position NEES.
                position_values.append(
                    safe_nees(e_p, P_pp)
                )

                # Marginal velocity NEES.
                marginal_values.append(
                    safe_nees(e_v, P_vv)
                )

                # Conditional velocity NEES given position error.
                conditional_error = (
                    e_v - P_vp @ np.linalg.solve(P_pp, e_p)
                )

                conditional_covariance = (
                    P_vv - P_vp @ np.linalg.solve(P_pp, P_pv)
                )

                conditional_covariance = (
                    conditional_covariance
                    + conditional_covariance.T
                ) / 2

                conditional_values.append(
                    safe_nees(
                        conditional_error,
                        conditional_covariance
                    )
                )

            selected_errors = errors[mask]
            selected_covariances = covariances[mask]

            velocity_error = selected_errors[:, 2:]
            velocity_variance = np.diagonal(
                selected_covariances[:, 2:, 2:],
                axis1=1,
                axis2=2
            )

            marginal_values = np.asarray(marginal_values)
            conditional_values = np.asarray(conditional_values)
            position_values = np.asarray(position_values)

            trial_records.append({
                "trial": trial + 1,
                "phase": phase,
                "samples": int(np.sum(mask)),

                "velocity_error_east_rmse": float(
                    np.sqrt(np.mean(velocity_error[:, 0]**2))
                ),
                "velocity_error_north_rmse": float(
                    np.sqrt(np.mean(velocity_error[:, 1]**2))
                ),
                "velocity_error_2d_rmse": float(
                    np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
                ),

                "velocity_std_east_mean": float(
                    np.mean(np.sqrt(velocity_variance[:, 0]))
                ),
                "velocity_std_north_mean": float(
                    np.mean(np.sqrt(velocity_variance[:, 1]))
                ),

                "velocity_nees_marginal_mean": float(
                    np.mean(marginal_values)
                ),
                "velocity_nees_marginal_coverage_percent": float(
                    np.mean(
                        (marginal_values >= CHI2_LOW)
                        & (marginal_values <= CHI2_HIGH)
                    ) * 100
                ),

                "velocity_nees_conditional_mean": float(
                    np.mean(conditional_values)
                ),
                "velocity_nees_conditional_coverage_percent": float(
                    np.mean(
                        (conditional_values >= CHI2_LOW)
                        & (conditional_values <= CHI2_HIGH)
                    ) * 100
                ),

                "position_nees_marginal_mean": float(
                    np.mean(position_values)
                ),

                "velocity_covariance_trace_mean": float(
                    np.mean(
                        np.trace(
                            selected_covariances[:, 2:, 2:],
                            axis1=1,
                            axis2=2
                        )
                    )
                )
            })

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{TRIALS} trials")


    trials_path = OUTPUT / "iona_velocity_consistency_diagnostic.csv"

    with trials_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(trial_records[0].keys())
        )
        writer.writeheader()
        writer.writerows(trial_records)

    summary = []

    for phase in ["pre", "outage", "post"]:
        selected = [
            row for row in trial_records
            if row["phase"] == phase
        ]

        fields = [
            "velocity_error_east_rmse",
            "velocity_error_north_rmse",
            "velocity_error_2d_rmse",
            "velocity_std_east_mean",
            "velocity_std_north_mean",
            "velocity_nees_marginal_mean",
            "velocity_nees_marginal_coverage_percent",
            "velocity_nees_conditional_mean",
            "velocity_nees_conditional_coverage_percent",
            "position_nees_marginal_mean",
            "velocity_covariance_trace_mean"
        ]

        item = {
            "scenario": SCENARIO,
            "tec_error_percent": TEC_ERROR_PERCENT,
            "phase": phase,
            "trials": len(selected)
        }

        for field in fields:
            values = np.asarray([
                row[field] for row in selected
            ])

            item[field] = float(np.mean(values))
            item[field + "_trial_sd"] = float(
                np.std(values, ddof=1)
            )

        summary.append(item)

    summary_path = (
        OUTPUT / "iona_velocity_consistency_summary.csv"
    )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nVELOCITY CONSISTENCY SUMMARY")
    print("-" * 100)

    for item in summary:
        print(
            f"{item['phase']:>7s} | "
            f"velocity RMSE {item['velocity_error_2d_rmse']:.4f} | "
            f"marginal NEES {item['velocity_nees_marginal_mean']:.4f} | "
            f"conditional NEES {item['velocity_nees_conditional_mean']:.4f} | "
            f"marginal coverage "
            f"{item['velocity_nees_marginal_coverage_percent']:.2f}% | "
            f"conditional coverage "
            f"{item['velocity_nees_conditional_coverage_percent']:.2f}%"
        )

    print("\nFILES GENERATED:")
    print(trials_path.resolve())
    print(summary_path.resolve())
    print("\nExpected mean for 2D velocity NEES: approximately 2.0")
    print("Pointwise 95% interval: [0.0506, 7.3778]")
    print("\nDIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()
