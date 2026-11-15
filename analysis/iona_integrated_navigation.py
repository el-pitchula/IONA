import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# IONA | INTEGRATED IONOSPHERE + GNSS + IMU EXPERIMENT
# Synthetic 2D navigation simulation
# ============================================================

SEED = 20261115
rng = np.random.default_rng(SEED)

# Mission parameters
DURATION = 120.0
DT = 0.02
IMU_RATE = 1.0 / DT
GNSS_INTERVAL = 1.0
OUTAGE_START = 45.0
OUTAGE_END = 75.0

# Sensor parameters
IMU_NOISE = 0.08
CODE_NOISE = 0.30

# GPS frequencies
F1 = 1575.42e6
F2 = 1227.60e6
K_IONO = 40.3

# Thin-shell ionosphere model
EARTH_RADIUS = 6371.0
SHELL_HEIGHT = 350.0

# Six simulated satellites
AZIMUTH = np.radians([0, 60, 120, 180, 240, 300])
ELEVATION = np.radians([25, 35, 45, 55, 65, 70])

mapping_ratio = (
    EARTH_RADIUS * np.cos(ELEVATION)
    / (EARTH_RADIUS + SHELL_HEIGHT)
)
MAPPING = 1.0 / np.sqrt(1.0 - mapping_ratio**2)

# Linearized pseudorange observation matrix:
# unknowns = east position error, north position error, clock error
H_GNSS = np.column_stack((
    -np.cos(ELEVATION) * np.sin(AZIMUTH),
    -np.cos(ELEVATION) * np.cos(AZIMUTH),
    np.ones(len(AZIMUTH))
))

# Dual-frequency ionosphere-free coefficients
A1 = F1**2 / (F1**2 - F2**2)
A2 = -F2**2 / (F1**2 - F2**2)

# Measurement covariance approximations
GEOMETRY_COV = np.linalg.inv(H_GNSS.T @ H_GNSS)[:2, :2]
R_L1 = CODE_NOISE**2 * GEOMETRY_COV
R_IF = (
    CODE_NOISE**2 * (A1**2 + A2**2) * GEOMETRY_COV
)

OUTPUT = Path("results/integrated_navigation")
OUTPUT.mkdir(parents=True, exist_ok=True)

TIME = np.arange(0, DURATION + DT, DT)
N = len(TIME)

GNSS_STEP = round(GNSS_INTERVAL / DT)
GNSS_INDICES = np.arange(0, N, GNSS_STEP)
GNSS_TIMES = TIME[GNSS_INDICES]

OUTAGE_MASK = (
    (GNSS_TIMES >= OUTAGE_START)
    & (GNSS_TIMES <= OUTAGE_END)
)


def ionospheric_delay(tec, frequency):
    return K_IONO * tec / frequency**2


def make_trajectory():
    """Create a smooth synthetic 2D reference trajectory."""

    acceleration_east = 0.08 * np.sin(2 * np.pi * TIME / 45)
    acceleration_north = 0.06 * np.cos(2 * np.pi * TIME / 60)

    acceleration = np.column_stack((
        acceleration_east,
        acceleration_north
    ))

    velocity = np.zeros((N, 2))
    position = np.zeros((N, 2))

    velocity[0] = [1.5, 0.5]

    for k in range(1, N):
        velocity[k] = velocity[k-1] + acceleration[k-1] * DT
        position[k] = (
            position[k-1]
            + velocity[k-1] * DT
            + 0.5 * acceleration[k-1] * DT**2
        )

    return position, velocity, acceleration


def make_tec_scenario(name):
    """Generate illustrative, not observational, VTEC scenarios."""

    if name == "quiet":
        return 10 + 1.5 * np.sin(2 * np.pi * GNSS_TIMES / 120)

    return (
        25
        + 8 * np.sin(2 * np.pi * GNSS_TIMES / 80)
        + 3 * np.sin(2 * np.pi * GNSS_TIMES / 17)
    )


def simulate_gnss(true_position, vtec_series):
    """Generate GNSS position fixes from simulated pseudorange errors."""

    fixes_l1 = []
    fixes_corrected = []
    fixes_if = []

    for j, t in enumerate(GNSS_TIMES):
        vtec = max(float(vtec_series[j]), 0.1)

        # Small independent satellite-to-satellite TEC variation.
        satellite_vtec = vtec * (
            1 + rng.normal(0, 0.04, len(AZIMUTH))
        )
        satellite_vtec = np.maximum(satellite_vtec, 0.1)

        slant_tec = satellite_vtec * MAPPING * 1e16

        delay_l1 = ionospheric_delay(slant_tec, F1)
        delay_l2 = ionospheric_delay(slant_tec, F2)

        noise_l1 = rng.normal(0, CODE_NOISE, len(AZIMUTH))
        noise_l2 = rng.normal(0, CODE_NOISE, len(AZIMUTH))

        # Single-frequency L1 observations.
        pseudorange_l1_error = delay_l1 + noise_l1

        # Idealized correction: exact simulated TEC is known.
        pseudorange_corrected_error = (
            pseudorange_l1_error - delay_l1
        )

        # Dual-frequency first-order ionosphere-free combination.
        pseudorange_l2_error = delay_l2 + noise_l2

        pseudorange_if_error = (
            A1 * pseudorange_l1_error
            + A2 * pseudorange_l2_error
        )

        # Solve east, north and receiver clock error.
        solution_l1 = np.linalg.lstsq(
            H_GNSS, pseudorange_l1_error, rcond=None
        )[0]

        solution_corrected = np.linalg.lstsq(
            H_GNSS, pseudorange_corrected_error, rcond=None
        )[0]

        solution_if = np.linalg.lstsq(
            H_GNSS, pseudorange_if_error, rcond=None
        )[0]

        truth = true_position[GNSS_INDICES[j]]

        fixes_l1.append(truth + solution_l1[:2])
        fixes_corrected.append(truth + solution_corrected[:2])
        fixes_if.append(truth + solution_if[:2])

    return (
        np.asarray(fixes_l1),
        np.asarray(fixes_corrected),
        np.asarray(fixes_if)
    )


def run_ekf(gnss_fixes, true_acceleration, measurement_covariance):
    """2D constant-velocity EKF with IMU acceleration input."""

    state = np.zeros(4)
    covariance = np.diag([10.0, 10.0, 4.0, 4.0])

    # Initialize position from the first GNSS fix.
    state[:2] = gnss_fixes[0]

    F = np.array([
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    B = np.array([
        [0.5 * DT**2, 0],
        [0, 0.5 * DT**2],
        [DT, 0],
        [0, DT]
    ])

    G = B.copy()
    Q = G @ (IMU_NOISE**2 * np.eye(2)) @ G.T
    Q += np.eye(4) * 1e-8

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    R = measurement_covariance
    estimates = np.zeros((N, 2))

    gnss_lookup = {
        int(index): row
        for row, index in enumerate(GNSS_INDICES)
    }

    for k in range(N):

        if k > 0:
            imu_measurement = (
                true_acceleration[k-1]
                + rng.normal(0, IMU_NOISE, 2)
            )

            state = F @ state + B @ imu_measurement
            covariance = F @ covariance @ F.T + Q

        if k in gnss_lookup:
            j = gnss_lookup[k]

            if not OUTAGE_MASK[j]:
                measurement = gnss_fixes[j]
                innovation = measurement - H @ state
                S = H @ covariance @ H.T + R
                gain = covariance @ H.T @ np.linalg.inv(S)

                state = state + gain @ innovation

                # Joseph form for numerical stability.
                identity = np.eye(4)
                covariance = (
                    (identity - gain @ H)
                    @ covariance
                    @ (identity - gain @ H).T
                    + gain @ R @ gain.T
                )

        estimates[k] = state[:2]

    return estimates


def hold_last_gnss(fixes):
    """GNSS-only baseline: hold last available fix during outage."""

    output = np.zeros((N, 2))
    last_fix = fixes[0]
    lookup = {
        int(index): row
        for row, index in enumerate(GNSS_INDICES)
    }

    for k in range(N):
        if k in lookup:
            j = lookup[k]
            if not OUTAGE_MASK[j]:
                last_fix = fixes[j]

        output[k] = last_fix

    return output


def imu_dead_reckoning(true_acceleration, initial_position):
    """IMU-only navigation, initialized from the first GNSS position."""

    position = np.zeros((N, 2))
    velocity = np.zeros((N, 2))

    position[0] = initial_position

    for k in range(1, N):
        measured_acceleration = (
            true_acceleration[k-1]
            + rng.normal(0, IMU_NOISE, 2)
        )

        velocity[k] = (
            velocity[k-1] + measured_acceleration * DT
        )

        position[k] = (
            position[k-1]
            + velocity[k-1] * DT
            + 0.5 * measured_acceleration * DT**2
        )

    return position


def rmse(estimate, truth, mask):
    difference = estimate[mask] - truth[mask]
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def run_scenario(name, true_position, true_acceleration):
    vtec = make_tec_scenario(name)

    fixes_l1, fixes_corrected, fixes_if = simulate_gnss(
        true_position, vtec
    )

    ekf_l1 = run_ekf(fixes_l1, true_acceleration, R_L1)
    ekf_corrected = run_ekf(
        fixes_corrected, true_acceleration, R_L1
    )
    ekf_if = run_ekf(fixes_if, true_acceleration, R_IF)

    gnss_only = hold_last_gnss(fixes_l1)

    imu_only = imu_dead_reckoning(
        true_acceleration, fixes_l1[0]
    )

    estimates = {
        "GNSS L1 only": gnss_only,
        "IMU only": imu_only,
        "EKF L1": ekf_l1,
        "EKF L1 ideal TEC correction": ekf_corrected,
        "EKF dual-frequency L1+L2": ekf_if
    }

    outage_time_mask = (
        (TIME >= OUTAGE_START)
        & (TIME <= OUTAGE_END)
    )

    rows = []

    print("\n" + "=" * 94)
    print(f"SCENARIO: {name.upper()} | VTEC SYNTHETIC")
    print("=" * 94)
    print(
        f"{'Method':34s} | "
        f"{'Full RMSE (m)':>14s} | "
        f"{'Outage RMSE (m)':>16s}"
    )
    print("-" * 94)

    for method, estimate in estimates.items():
        full_error = rmse(
            estimate, true_position, np.ones(N, dtype=bool)
        )
        outage_error = rmse(
            estimate, true_position, outage_time_mask
        )

        print(
            f"{method:34s} | "
            f"{full_error:14.3f} | "
            f"{outage_error:16.3f}"
        )

        rows.append({
            "scenario": name,
            "method": method,
            "full_rmse_m": full_error,
            "outage_rmse_m": outage_error
        })

    return rows, estimates, vtec


def main():
    true_position, true_velocity, true_acceleration = (
        make_trajectory()
    )

    all_rows = []
    disturbed_estimates = None
    disturbed_vtec = None

    for scenario in ["quiet", "disturbed"]:
        rows, estimates, vtec = run_scenario(
            scenario, true_position, true_acceleration
        )

        all_rows.extend(rows)

        if scenario == "disturbed":
            disturbed_estimates = estimates
            disturbed_vtec = vtec

    csv_path = OUTPUT / "integrated_navigation_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Plot the disturbed scenario.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        true_position[:, 0], true_position[:, 1],
        label="Reference trajectory", linewidth=2
    )

    for method in [
        "GNSS L1 only",
        "EKF L1",
        "EKF L1 ideal TEC correction",
        "EKF dual-frequency L1+L2"
    ]:
        estimate = disturbed_estimates[method]
        axes[0].plot(
            estimate[:, 0], estimate[:, 1],
            label=method, alpha=0.8
        )

    axes[0].axvline(
        x=true_position[np.argmin(abs(TIME - OUTAGE_START)), 0],
        linestyle=":", alpha=0.5
    )
    axes[0].set_title("Navigation estimates | disturbed TEC")
    axes[0].set_xlabel("East position (m)")
    axes[0].set_ylabel("North position (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        GNSS_TIMES, disturbed_vtec,
        label="Synthetic VTEC"
    )
    axes[1].axvspan(
        OUTAGE_START, OUTAGE_END,
        alpha=0.2, label="Independent GNSS outage"
    )
    axes[1].set_title("Ionospheric scenario and GNSS outage")
    axes[1].set_xlabel("Mission time (s)")
    axes[1].set_ylabel("VTEC (TECU)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("IONA | Integrated GNSS + IMU + Ionosphere")
    fig.tight_layout()

    figure_path = OUTPUT / "integrated_navigation.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nFiles generated:")
    print(csv_path.resolve())
    print(figure_path.resolve())
    print("\nINTEGRATED NAVIGATION EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
