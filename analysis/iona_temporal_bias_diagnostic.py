import numpy as np
import iona_velocity_consistency_diagnostic as diag


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    position_errors = []
    velocity_errors = []

    print("IONA | TEMPORAL BIAS EVOLUTION")
    print("Trials:", diag.TRIALS)

    for trial in range(diag.TRIALS):
        seed = diag.BASE_SEED + trial

        fixes, R_series = diag.simulate_gnss(
            vtec, diag.TEC_ERROR_PERCENT, seed
        )

        estimates, _ = diag.run_filter(
            fixes, R_series, seed + 1000003
        )

        errors = estimates - diag.true_state

        position_errors.append(errors[:, :2])
        velocity_errors.append(errors[:, 2:])

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    position_errors = np.asarray(position_errors)
    velocity_errors = np.asarray(velocity_errors)

    selected_seconds = [
        0, 1, 2, 5, 10, 20, 30, 40, 44,
        45, 50, 55, 60, 65, 70, 75, 76,
        80, 85, 90, 100, 110, 120
    ]

    print("\nENSEMBLE MEAN STATE ERROR")
    print("Time | Pos E | Pos N | Vel E | Vel N")

    for seconds in selected_seconds:
        k = int(round(seconds / diag.sim.DT))

        if k >= diag.sim.N:
            continue

        pe = np.mean(position_errors[:, k, :], axis=0)
        ve = np.mean(velocity_errors[:, k, :], axis=0)

        print(
            f"{seconds:5.1f} | "
            f"{pe[0]: .5f} | {pe[1]: .5f} | "
            f"{ve[0]: .5f} | {ve[1]: .5f}"
        )

    print("\nINITIAL CONDITIONS")
    print("True initial velocity:", diag.true_state[0, 2:])
    print("Filter initial velocity: [0.0, 0.0]")
    print("Initial velocity error:", -diag.true_state[0, 2:])


if __name__ == "__main__":
    main()
