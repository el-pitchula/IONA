import numpy as np
import iona_velocity_consistency_diagnostic as diag


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    phases = ["pre", "post"]
    fields = [
        "position_gain",
        "innovation",
        "gain_times_mean_innovation",
        "gain_innovation_covariance",
        "position_correction",
        "cross_covariance",
        "pre_update_error",
        "post_update_error",
    ]

    records = {
        phase: {field: [] for field in fields}
        for phase in phases
    }

    print("IONA | KALMAN GAIN DIAGNOSTIC")
    print("Trials:", diag.TRIALS)

    for trial in range(diag.TRIALS):
        seed = diag.BASE_SEED + trial

        fixes, R_series = diag.simulate_gnss(
            vtec, diag.TEC_ERROR_PERCENT, seed
        )

        state = np.zeros(4)
        state[:2] = fixes[0]

        P = np.zeros((4, 4))
        P[:2, :2] = R_series[0]
        P[2:, 2:] = np.diag([4.0, 4.0])

        rng = np.random.default_rng(seed + 1000003)

        gnss_lookup = {
            int(index): row
            for row, index in enumerate(diag.sim.GNSS_INDICES)
        }

        trial_records = {
            phase: {field: [] for field in fields}
            for phase in phases
        }

        for k in range(diag.sim.N):
            if k > 0:
                imu = (
                    diag.true_acceleration[k - 1]
                    + rng.normal(0, diag.sim.IMU_NOISE, 2)
                )

                state = diag.F @ state + diag.B @ imu
                P = diag.F @ P @ diag.F.T + diag.Q

            if k not in gnss_lookup:
                continue

            j = gnss_lookup[k]

            if k == 0 or diag.sim.OUTAGE_MASK[j]:
                continue

            phase = (
                "pre"
                if diag.sim.GNSS_TIMES[j] < diag.sim.OUTAGE_START
                else "post"
            )

            truth = diag.true_position[k]
            innovation = fixes[j] - diag.H @ state

            pre_error = state[:2] - truth

            S = diag.H @ P @ diag.H.T + R_series[j]
            K = np.linalg.solve(S, diag.H @ P).T

            position_gain = K[:2, :]
            correction = position_gain @ innovation
            trial_records[phase]["innovation"].append(innovation)
            cross_covariance = P[:2, 2:].copy()

            state = state + K @ innovation

            P = (
                (diag.IDENTITY - K @ diag.H)
                @ P
                @ (diag.IDENTITY - K @ diag.H).T
                + K @ R_series[j] @ K.T
            )

            post_error = state[:2] - truth

            trial_records[phase]["position_gain"].append(
                position_gain
            )
            trial_records[phase]["position_correction"].append(
                correction
            )
            trial_records[phase]["cross_covariance"].append(
                cross_covariance
            )
            trial_records[phase]["pre_update_error"].append(
                pre_error
            )
            trial_records[phase]["post_update_error"].append(
                post_error
            )

        for phase in phases:
            gains = trial_records[phase]["position_gain"]
            innovations = trial_records[phase]["innovation"]

            if gains and innovations:
                mean_gain = np.mean(gains, axis=0)
                mean_innovation = np.mean(innovations, axis=0)

                trial_records[phase]["gain_times_mean_innovation"].append(
                    mean_gain @ mean_innovation
                )

                gain_array = np.asarray(gains)
                innovation_array = np.asarray(innovations)

                centered_gain = gain_array - mean_gain
                centered_innovation = innovation_array - mean_innovation

                temporal_covariance = np.mean(
                    np.einsum(
                        "nij,nj->ni",
                        centered_gain,
                        centered_innovation
                    ),
                    axis=0
                )

                trial_records[phase]["gain_innovation_covariance"].append(
                    temporal_covariance
                )

            for field in fields:
                values = trial_records[phase][field]

                if values:
                    records[phase][field].append(
                        np.mean(values, axis=0)
                    )

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    print("\nENSEMBLE MEAN OF TRIAL-LEVEL MEANS")

    for phase in phases:
        print("\nPHASE:", phase.upper())

        for field in fields:
            values = np.asarray(records[phase][field])

            mean = np.mean(values, axis=0)
            sd = np.std(values, axis=0, ddof=1)

            print(f"\n{field}")
            print("Mean:")
            print(np.array2string(mean, precision=6))
            print("Across-trial SD:")
            print(np.array2string(sd, precision=6))


if __name__ == "__main__":
    main()
