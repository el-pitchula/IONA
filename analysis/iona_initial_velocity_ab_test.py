import numpy as np
import iona_velocity_consistency_diagnostic as diag


def run_filter_variant(fixes, covariance_series, seed, initial_velocity):
    state = np.zeros(4)
    state[:2] = fixes[0]
    state[2:] = initial_velocity

    covariance = np.zeros((4, 4))
    covariance[:2, :2] = covariance_series[0]
    covariance[2:, 2:] = np.diag([4.0, 4.0])

    estimates = np.zeros((diag.sim.N, 4))

    gnss_lookup = {
        int(index): row
        for row, index in enumerate(diag.sim.GNSS_INDICES)
    }

    rng = np.random.default_rng(seed)

    for k in range(diag.sim.N):
        if k > 0:
            imu_measurement = (
                diag.true_acceleration[k - 1]
                + rng.normal(0, diag.sim.IMU_NOISE, 2)
            )

            state = diag.F @ state + diag.B @ imu_measurement
            covariance = (
                diag.F @ covariance @ diag.F.T + diag.Q
            )

        if k in gnss_lookup:
            j = gnss_lookup[k]

            if k > 0 and not diag.sim.OUTAGE_MASK[j]:
                measurement = fixes[j]
                R = covariance_series[j]

                innovation = measurement - diag.H @ state
                S = diag.H @ covariance @ diag.H.T + R

                gain = np.linalg.solve(
                    S, diag.H @ covariance
                ).T

                state = state + gain @ innovation

                covariance = (
                    (diag.IDENTITY - gain @ diag.H)
                    @ covariance
                    @ (diag.IDENTITY - gain @ diag.H).T
                    + gain @ R @ gain.T
                )

        estimates[k] = state

    return estimates


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    variants = {
        "A_ZERO": [],
        "B_IDEAL": [],
        "C_GNSS": [],
    }

    phases = ["pre", "outage", "post"]

    print("IONA | INITIAL VELOCITY A/B/C TEST")
    print("Trials:", diag.TRIALS)
    print("TEC uncertainty:", diag.TEC_ERROR_PERCENT, "%")
    print("Scenario:", diag.SCENARIO)

    gnss_dt = (
        diag.sim.GNSS_TIMES[1] - diag.sim.GNSS_TIMES[0]
    )

    for trial in range(diag.TRIALS):
        seed = diag.BASE_SEED + trial

        fixes, R_series = diag.simulate_gnss(
            vtec, diag.TEC_ERROR_PERCENT, seed
        )

        velocity_zero = np.zeros(2)
        velocity_ideal = diag.true_state[0, 2:].copy()
        velocity_gnss = (fixes[1] - fixes[0]) / gnss_dt

        initializations = {
            "A_ZERO": velocity_zero,
            "B_IDEAL": velocity_ideal,
            "C_GNSS": velocity_gnss,
        }

        for name, initial_velocity in initializations.items():
            estimates = run_filter_variant(
                fixes,
                R_series,
                seed + 1000003,
                initial_velocity,
            )

            errors = estimates - diag.true_state
            variants[name].append(errors)

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    print("\nINITIAL VELOCITY ESTIMATE")
    print("A_ZERO: [0, 0]")
    print("B_IDEAL:", diag.true_state[0, 2:])
    print("C_GNSS: finite difference of first two GNSS fixes")
    print("GNSS interval:", gnss_dt, "s")

    print("\nMEAN ERROR BY VARIANT AND PHASE")
    print("Values are ensemble means of trial-level phase means.")
    print("Variant | Phase | Pos E | Pos N | Vel E | Vel N")

    for name, trial_errors in variants.items():
        trial_errors = np.asarray(trial_errors)

        for phase in phases:
            mask = diag.phase_mask(phase)

            selected = trial_errors[:, mask, :]
            per_trial_mean = np.mean(selected, axis=1)
            mean_error = np.mean(per_trial_mean, axis=0)
            sd_error = np.std(per_trial_mean, axis=0, ddof=1)

            print(
                f"{name:8s} | {phase:7s} | "
                f"{mean_error[0]: .5f} | "
                f"{mean_error[1]: .5f} | "
                f"{mean_error[2]: .5f} | "
                f"{mean_error[3]: .5f}"
            )

            print(
                " " * 10 +
                "trial SD: "
                + np.array2string(sd_error, precision=5)
            )


if __name__ == "__main__":
    main()
