import numpy as np
import iona_velocity_consistency_diagnostic as diag


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    phases = ["pre", "outage", "post"]
    fields = [
        "gnss_error",
        "innovation",
        "pre_update_error",
        "post_update_error",
    ]

    records = {
        phase: {field: [] for field in fields}
        for phase in phases
    }

    print("IONA | GNSS INNOVATION DIAGNOSTIC")
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
            phase = (
                "outage" if diag.sim.OUTAGE_MASK[j]
                else "pre" if diag.sim.GNSS_TIMES[j] < diag.sim.OUTAGE_START
                else "post"
            )

            if k == 0 or diag.sim.OUTAGE_MASK[j]:
                continue

            truth = diag.true_position[k]
            measurement = fixes[j]
            R = R_series[j]

            predicted_position = diag.H @ state
            innovation = measurement - predicted_position

            pre_error = state[:2] - truth
            gnss_error = measurement - truth

            S = diag.H @ P @ diag.H.T + R
            K = np.linalg.solve(S, diag.H @ P).T

            state = state + K @ innovation

            P = (
                (diag.IDENTITY - K @ diag.H)
                @ P
                @ (diag.IDENTITY - K @ diag.H).T
                + K @ R @ K.T
            )

            post_error = state[:2] - truth

            trial_records[phase]["gnss_error"].append(gnss_error)
            trial_records[phase]["innovation"].append(innovation)
            trial_records[phase]["pre_update_error"].append(pre_error)
            trial_records[phase]["post_update_error"].append(post_error)

        for phase in phases:
            for field in fields:
                values = trial_records[phase][field]
                if values:
                    records[phase][field].append(
                        np.mean(values, axis=0)
                    )

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    print("\nENSEMBLE MEAN OF TRIAL-LEVEL MEANS")
    print("Phase | Quantity | East (m) | North (m)")

    for phase in phases:
        print("\nPHASE:", phase.upper())

        for field in fields:
            values = np.asarray(records[phase][field])

            if len(values) == 0:
                continue

            mean = np.mean(values, axis=0)
            sd = np.std(values, axis=0, ddof=1)
            se = sd / np.sqrt(len(values))

            print(
                f"{field:20s} | "
                f"{mean[0]: .6f} | {mean[1]: .6f}"
            )
            print(
                f"{'':20s} | SD "
                f"{sd[0]:.6f} | {sd[1]:.6f}"
            )
            print(
                f"{'':20s} | SE "
                f"{se[0]:.6f} | {se[1]:.6f}"
            )


if __name__ == "__main__":
    main()
