import csv
from pathlib import Path

import numpy as np
import iona_integrated_navigation as sim

TRIALS = 100
BASE_SEED = 20261118
TEC_ERROR_LEVELS = [25]
SCENARIOS = ["disturbed"]

OUTPUT = Path("results/integrated_navigation")
OUTPUT.mkdir(parents=True, exist_ok=True)

true_position, true_velocity, true_acceleration = sim.make_trajectory()

true_state = np.column_stack((
    true_position[:, 0],
    true_position[:, 1],
    true_velocity[:, 0],
    true_velocity[:, 1]
))

H_STATE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

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

IDENTITY = np.eye(4)

# Pointwise 95% chi-square intervals.
# NIS: 2 degrees of freedom.
NIS_LOW = 0.0506
NIS_HIGH = 7.3778

# NEES: 4 degrees of freedom.
NEES_LOW = 0.4844
NEES_HIGH = 11.1433


def simulate_gnss_with_uncertainty(vtec_series, error_percent, seed):
    rng = np.random.default_rng(seed)

    fixes = []
    covariance_series = []

    H = sim.H_GNSS
    H_INV = np.linalg.inv(H.T @ H) @ H.T

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

        common_cov = (
            level**2
            * np.outer(estimated_delay, estimated_delay)
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


def run_filter(fixes, covariance_series, method):
    state = np.zeros(4)

    initial_R = (
        sim.R_L1
        if method == "fixed"
        else covariance_series[0]
    )

    covariance = np.zeros((4, 4))
    covariance[:2, :2] = initial_R
    covariance[2:, 2:] = np.diag([4.0, 4.0])

    state[:2] = fixes[0]

    estimates = np.zeros((sim.N, 4))

    nis_values = []
    nees_values = []
    nees_outage_values = []

    nees_pre_outage = []
    nees_during_outage = []
    nees_post_outage = []

    nees_position_values = []
    nees_velocity_conditional_values = []
    nees_decomposition_residuals = []

    gnss_lookup = {
        int(index): row
        for row, index in enumerate(sim.GNSS_INDICES)
    }

    for k in range(sim.N):

        if k > 0:
            imu_measurement = (
                true_acceleration[k - 1]
                + sim.rng.normal(0, sim.IMU_NOISE, 2)
            )

            state = F @ state + B @ imu_measurement
            covariance = F @ covariance @ F.T + Q

        if k in gnss_lookup:
            j = gnss_lookup[k]

            if k > 0 and not sim.OUTAGE_MASK[j]:
                measurement = fixes[j]

                if method == "fixed":
                    R = sim.R_L1
                else:
                    R = covariance_series[j]

                innovation = measurement - H_STATE @ state

                S = (
                    H_STATE @ covariance @ H_STATE.T
                    + R
                )

                # Exclude k=0: its GNSS observation was
                # already used to initialize the state.
                if k > 0:
                    nis = float(
                        innovation @ np.linalg.solve(S, innovation)
                    )
                    nis_values.append(nis)

                gain = np.linalg.solve(
                    S,
                    H_STATE @ covariance
                ).T

                state = state + gain @ innovation

                covariance = (
                    (IDENTITY - gain @ H_STATE)
                    @ covariance
                    @ (IDENTITY - gain @ H_STATE).T
                    + gain @ R @ gain.T
                )

        estimates[k] = state

        error = state - true_state[k]

        nees = float(
            error @ np.linalg.solve(covariance, error)
        )

        nees_values.append(nees)

        # Conditional NEES decomposition:
        # total = marginal position + conditional velocity.
        P_pp = covariance[:2, :2]
        P_pv = covariance[:2, 2:]
        P_vp = covariance[2:, :2]
        P_vv = covariance[2:, 2:]

        error_position = error[:2]
        error_velocity = error[2:]

        nees_position = float(
            error_position
            @ np.linalg.solve(P_pp, error_position)
        )

        conditional_velocity_error = (
            error_velocity
            - P_vp @ np.linalg.solve(P_pp, error_position)
        )

        conditional_velocity_covariance = (
            P_vv
            - P_vp @ np.linalg.solve(P_pp, P_pv)
        )

        nees_velocity_conditional = float(
            conditional_velocity_error
            @ np.linalg.solve(
                conditional_velocity_covariance,
                conditional_velocity_error
            )
        )

        nees_position_values.append(nees_position)
        nees_velocity_conditional_values.append(
            nees_velocity_conditional
        )

        nees_decomposition_residuals.append(
            abs(nees - nees_position - nees_velocity_conditional)
        )

        current_time = sim.TIME[k]

        if current_time < sim.OUTAGE_START:
            nees_pre_outage.append(nees)

        elif current_time <= sim.OUTAGE_END:
            nees_during_outage.append(nees)
            nees_outage_values.append(nees)

        else:
            nees_post_outage.append(nees)

    return (
        np.asarray(nis_values),
        np.asarray(nees_values),
        np.asarray(nees_outage_values),
        np.asarray(nees_pre_outage),
        np.asarray(nees_during_outage),
        np.asarray(nees_post_outage),
        np.asarray(nees_position_values),
        np.asarray(nees_velocity_conditional_values),
        np.asarray(nees_decomposition_residuals)
    )


def metrics(values, lower, upper):
    return {
        "mean": float(np.mean(values)),
        "coverage": float(
            np.mean((values >= lower) & (values <= upper)) * 100
        )
    }


trial_records = []

print("\nIONA | FILTER CONSISTENCY EXPERIMENT")
print("Trials per condition:", TRIALS)
print("Method: adaptive EKF")
print("Metrics: NIS, NEES full trajectory, NEES outage")

for scenario in SCENARIOS:
    vtec = sim.make_tec_scenario(scenario)

    for error_percent in TEC_ERROR_LEVELS:
        for trial in range(TRIALS):

            seed = BASE_SEED + trial

            fixes, covariance_series = (
                simulate_gnss_with_uncertainty(
                    vtec, error_percent, seed
                )
            )

            for method in ["adaptive"]:

                # Identical IMU noise sequence for both filters.
                sim.rng = np.random.default_rng(seed)

                (
                    nis, nees, nees_outage,
                    nees_pre, nees_during, nees_post,
                    nees_pos, nees_vel_cond, decomp_residual
                ) = run_filter(
                    fixes,
                    covariance_series,
                    method
                )

                nis_result = metrics(
                    nis, NIS_LOW, NIS_HIGH
                )

                nees_result = metrics(
                    nees, NEES_LOW, NEES_HIGH
                )

                outage_result = metrics(
                    nees_outage, NEES_LOW, NEES_HIGH
                )

                pre_result = metrics(
                    nees_pre, NEES_LOW, NEES_HIGH
                )

                during_result = metrics(
                    nees_during, NEES_LOW, NEES_HIGH
                )

                post_result = metrics(
                    nees_post, NEES_LOW, NEES_HIGH
                )

                trial_records.append({
                    "scenario": scenario,
                    "tec_error_percent": error_percent,
                    "trial": trial + 1,
                    "method": method,
                    "nis_mean": nis_result["mean"],
                    "nis_coverage_percent": nis_result["coverage"],
                    "nees_full_mean": nees_result["mean"],
                    "nees_full_coverage_percent": nees_result["coverage"],
                    "nees_outage_mean": outage_result["mean"],
                    "nees_outage_coverage_percent": outage_result["coverage"],

                    "nees_pre_mean": pre_result["mean"],
                    "nees_pre_coverage_percent": pre_result["coverage"],

                    "nees_during_mean": during_result["mean"],
                    "nees_during_coverage_percent": during_result["coverage"],

                    "nees_post_mean": post_result["mean"],
                    "nees_post_coverage_percent": post_result["coverage"],

                    "nees_position_pre_mean": float(np.mean(
                        nees_pos[sim.TIME < sim.OUTAGE_START]
                    )),
                    "nees_position_during_mean": float(np.mean(
                        nees_pos[
                            (sim.TIME >= sim.OUTAGE_START)
                            & (sim.TIME <= sim.OUTAGE_END)
                        ]
                    )),
                    "nees_position_post_mean": float(np.mean(
                        nees_pos[sim.TIME > sim.OUTAGE_END]
                    )),

                    "nees_velocity_cond_pre_mean": float(np.mean(
                        nees_vel_cond[sim.TIME < sim.OUTAGE_START]
                    )),
                    "nees_velocity_cond_during_mean": float(np.mean(
                        nees_vel_cond[
                            (sim.TIME >= sim.OUTAGE_START)
                            & (sim.TIME <= sim.OUTAGE_END)
                        ]
                    )),
                    "nees_velocity_cond_post_mean": float(np.mean(
                        nees_vel_cond[sim.TIME > sim.OUTAGE_END]
                    )),

                    "decomposition_max_residual": float(
                        np.max(decomp_residual)
                    )
                })

            if (trial + 1) % 25 == 0:
                print(
                    f"{scenario} | TEC {error_percent}% | "
                    f"{trial + 1}/{TRIALS}"
                )


trial_path = OUTPUT / "filter_consistency_decomposition_trials.csv"

with trial_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(trial_records[0].keys())
    )
    writer.writeheader()
    writer.writerows(trial_records)


summary = []

print("\n" + "=" * 115)
print("FILTER CONSISTENCY SUMMARY")
print("=" * 115)

for scenario in SCENARIOS:
    for error_percent in TEC_ERROR_LEVELS:

        print(f"\n{scenario.upper()} | TEC uncertainty {error_percent}%")

        for method in ["adaptive"]:

            selected = [
                row for row in trial_records
                if row["scenario"] == scenario
                and row["tec_error_percent"] == error_percent
                and row["method"] == method
            ]

            fields = [
                "nis_mean",
                "nis_coverage_percent",
                "nees_full_mean",
                "nees_full_coverage_percent",
                "nees_outage_mean",
                "nees_outage_coverage_percent",
                "nees_position_pre_mean",
                "nees_position_during_mean",
                "nees_position_post_mean",
                "nees_velocity_cond_pre_mean",
                "nees_velocity_cond_during_mean",
                "nees_velocity_cond_post_mean",
                "decomposition_max_residual"
            ]

            item = {
                "scenario": scenario,
                "tec_error_percent": error_percent,
                "method": method,
                "trials": len(selected)
            }

            for field in fields:
                values = np.array([
                    row[field] for row in selected
                ])

                item[field] = float(np.mean(values))
                item[field + "_trial_sd"] = float(
                    np.std(values, ddof=1)
                )

            summary.append(item)

            print(
                f"{method:>8s} | "
                f"NIS {item['nis_mean']:.3f} "
                f"(coverage {item['nis_coverage_percent']:.1f}%) | "
                f"NEES {item['nees_full_mean']:.3f} "
                f"(coverage {item['nees_full_coverage_percent']:.1f}%) | "
                f"NEES outage {item['nees_outage_mean']:.3f} "
                f"(coverage {item['nees_outage_coverage_percent']:.1f}%)"
            )


summary_path = OUTPUT / "filter_consistency_decomposition_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)


print("\nFILES GENERATED:")
print(trial_path.resolve())
print(summary_path.resolve())

print("\nREFERENCE EXPECTATIONS:")
print("Mean NIS approximately 2.0")
print("Mean NEES approximately 4.0")
print("Pointwise 95% NIS interval: [0.0506, 7.3778]")
print("Pointwise 95% NEES interval: [0.4844, 11.1433]")

print("\nFILTER CONSISTENCY EXPERIMENT COMPLETED")




