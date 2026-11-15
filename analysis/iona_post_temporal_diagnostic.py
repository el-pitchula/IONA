from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iona_velocity_consistency_diagnostic as diag


WINDOWS = [
    ("75 s", 75.0, 76.0),
    ("76 s", 76.0, 77.0),
    ("77 s", 77.0, 78.0),
    ("78 s", 78.0, 79.0),
    ("79-90 s", 79.0, 90.0),
    ("90-105 s", 90.0, 105.0),
    ("105-120 s", 105.0, 121.0),
]

FIELDS = [
    "gain",
    "innovation",
    "correction",
    "product_of_means",
    "covariance_term",
    "pre_error",
    "post_error",
]


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    results = {
        name: {field: [] for field in FIELDS}
        for name, _, _ in WINDOWS
    }

    print("IONA | POST-OUTAGE TEMPORAL DIAGNOSTIC")
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

        trial_data = {
            name: {
                "gain": [],
                "innovation": [],
                "correction": [],
                "pre_error": [],
                "post_error": [],
            }
            for name, _, _ in WINDOWS
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

            t = float(diag.sim.GNSS_TIMES[j])

            selected_window = None
            for name, start, end in WINDOWS:
                if start <= t < end:
                    selected_window = name
                    break

            if selected_window is None:
                continue

            truth = diag.true_position[k]
            innovation = fixes[j] - diag.H @ state
            pre_error = state[:2] - truth

            S = diag.H @ P @ diag.H.T + R_series[j]
            K = np.linalg.solve(S, diag.H @ P).T

            position_gain = K[:2, :]
            correction = position_gain @ innovation

            state = state + K @ innovation

            P = (
                (diag.IDENTITY - K @ diag.H)
                @ P
                @ (diag.IDENTITY - K @ diag.H).T
                + K @ R_series[j] @ K.T
            )

            post_error = state[:2] - truth

            data = trial_data[selected_window]
            data["gain"].append(position_gain)
            data["innovation"].append(innovation)
            data["correction"].append(correction)
            data["pre_error"].append(pre_error)
            data["post_error"].append(post_error)

        for name, _, _ in WINDOWS:
            data = trial_data[name]

            if not data["gain"]:
                continue

            gains = np.asarray(data["gain"])
            innovations = np.asarray(data["innovation"])

            mean_gain = np.mean(gains, axis=0)
            mean_innovation = np.mean(innovations, axis=0)

            product = mean_gain @ mean_innovation

            covariance_term = np.mean(
                np.einsum(
                    "nij,nj->ni",
                    gains - mean_gain,
                    innovations - mean_innovation
                ),
                axis=0
            )

            results[name]["gain"].append(mean_gain)
            results[name]["innovation"].append(mean_innovation)
            results[name]["correction"].append(
                np.mean(data["correction"], axis=0)
            )
            results[name]["product_of_means"].append(product)
            results[name]["covariance_term"].append(covariance_term)
            results[name]["pre_error"].append(
                np.mean(data["pre_error"], axis=0)
            )
            results[name]["post_error"].append(
                np.mean(data["post_error"], axis=0)
            )

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    print("\nENSEMBLE MEAN OF TRIAL-LEVEL MEANS")

    for name, _, _ in WINDOWS:
        print("\nWINDOW:", name)

        for field in FIELDS:
            values = np.asarray(results[name][field])

            if len(values) == 0:
                print(field, ": no data")
                continue

            mean = np.mean(values, axis=0)
            sd = np.std(values, axis=0, ddof=1)

            print("\n" + field)
            print("Mean:")
            print(np.array2string(mean, precision=6))
            print("Across-trial SD:")
            print(np.array2string(sd, precision=6))


if __name__ == "__main__":
    main()
