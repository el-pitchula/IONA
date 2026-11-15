import csv
import numpy as np

import iona_velocity_consistency_diagnostic as diag


PHASES = ["pre", "outage", "post"]
OUTPUT = diag.OUTPUT / "iona_cross_covariance_diagnostic.csv"


def symmetrize(matrix):
    return (matrix + matrix.T) / 2


def correlation_from_covariance(covariance):
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    return covariance / np.outer(std, std)


def whitened_eigenvalues(empirical, predicted):
    predicted = symmetrize(predicted)
    empirical = symmetrize(empirical)

    values, vectors = np.linalg.eigh(predicted)
    values = np.maximum(values, 1e-15)

    inv_sqrt = (
        vectors
        @ np.diag(1.0 / np.sqrt(values))
        @ vectors.T
    )

    normalized = symmetrize(
        inv_sqrt @ empirical @ inv_sqrt
    )

    return np.linalg.eigvalsh(normalized)


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    all_errors = []
    all_covariances = []

    print("IONA | CROSS-COVARIANCE DIAGNOSTIC")
    print("Trials:", diag.TRIALS)
    print("Scenario:", diag.SCENARIO)
    print("TEC uncertainty:", diag.TEC_ERROR_PERCENT, "%")

    for trial in range(diag.TRIALS):
        seed = diag.BASE_SEED + trial

        fixes, covariance_series = diag.simulate_gnss(
            vtec,
            diag.TEC_ERROR_PERCENT,
            seed
        )

        estimates, covariances = diag.run_filter(
            fixes,
            covariance_series,
            seed + 1000003
        )

        all_errors.append(estimates - diag.true_state)
        all_covariances.append(covariances)

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    errors = np.asarray(all_errors)
    covariances = np.asarray(all_covariances)

    records = []

    for phase in PHASES:
        mask = diag.phase_mask(phase)
        phase_errors = errors[:, mask, :]
        phase_covariances = covariances[:, mask, :, :]

        # Ensemble mean error at each epoch.
        mean_error = np.mean(phase_errors, axis=0)

        # Empirical centered covariance across trials,
        # averaged over epochs.
        empirical_centered = np.zeros((4, 4))
        empirical_second_moment = np.zeros((4, 4))

        for t in range(phase_errors.shape[1]):
            epoch_errors = phase_errors[:, t, :]
            centered = epoch_errors - np.mean(
                epoch_errors, axis=0
            )

            empirical_centered += (
                centered.T @ centered
            ) / diag.TRIALS

            empirical_second_moment += (
                epoch_errors.T @ epoch_errors
            ) / diag.TRIALS

        empirical_centered /= phase_errors.shape[1]
        empirical_second_moment /= phase_errors.shape[1]

        predicted = np.mean(
            phase_covariances,
            axis=(0, 1)
        )

        empirical_centered = symmetrize(empirical_centered)
        empirical_second_moment = symmetrize(
            empirical_second_moment
        )
        predicted = symmetrize(predicted)

        # Position-velocity cross-covariance.
        empirical_pv = empirical_centered[:2, 2:]
        predicted_pv = predicted[:2, 2:]

        empirical_corr = correlation_from_covariance(
            empirical_centered
        )
        predicted_corr = correlation_from_covariance(
            predicted
        )

        # Conditional velocity error and covariance.
        conditional_errors = []
        conditional_predictions = []

        for trial in range(diag.TRIALS):
            for t in range(phase_errors.shape[1]):
                P = phase_covariances[trial, t]
                e = phase_errors[trial, t]

                Ppp = P[:2, :2]
                Ppv = P[:2, 2:]
                Pvp = P[2:, :2]
                Pvv = P[2:, 2:]

                ep = e[:2]
                ev = e[2:]

                conditional_error = (
                    ev - Pvp @ np.linalg.solve(Ppp, ep)
                )

                conditional_covariance = (
                    Pvv - Pvp @ np.linalg.solve(Ppp, Ppv)
                )

                conditional_errors.append(conditional_error)
                conditional_predictions.append(
                    symmetrize(conditional_covariance)
                )

        conditional_errors = np.asarray(conditional_errors)
        conditional_predictions = np.asarray(
            conditional_predictions
        )

        conditional_mean = np.mean(
            conditional_errors, axis=0
        )

        centered_conditional = (
            conditional_errors - conditional_mean
        )

        conditional_empirical = (
            centered_conditional.T @ centered_conditional
        ) / len(conditional_errors)

        conditional_second_moment = (
            conditional_errors.T @ conditional_errors
        ) / len(conditional_errors)

        conditional_predicted = np.mean(
            conditional_predictions, axis=0
        )

        conditional_empirical = symmetrize(
            conditional_empirical
        )
        conditional_second_moment = symmetrize(
            conditional_second_moment
        )
        conditional_predicted = symmetrize(
            conditional_predicted
        )

        eig_full = whitened_eigenvalues(
            empirical_centered,
            predicted
        )

        eig_conditional = whitened_eigenvalues(
            conditional_empirical,
            conditional_predicted
        )

        eig_pv = whitened_eigenvalues(
            empirical_pv @ empirical_pv.T,
            predicted_pv @ predicted_pv.T
            + np.eye(2) * 1e-12
        )

        print("\n" + "=" * 65)
        print("PHASE:", phase.upper())
        print("=" * 65)

        print("\nMean error [E, N, VE, VN]:")
        print(np.mean(phase_errors, axis=(0, 1)))

        print("\nPredicted position-velocity correlation:")
        print(predicted_corr[:2, 2:])

        print("\nEmpirical position-velocity correlation:")
        print(empirical_corr[:2, 2:])

        print("\nPredicted P_pv:")
        print(predicted_pv)

        print("\nEmpirical P_pv:")
        print(empirical_pv)

        print("\nPredicted conditional velocity covariance:")
        print(conditional_predicted)

        print("\nEmpirical conditional velocity covariance:")
        print(conditional_empirical)

        print("\nConditional error second moment:")
        print(conditional_second_moment)

        print("\nFull-state covariance ratio eigenvalues:")
        print(eig_full)

        print("\nConditional covariance ratio eigenvalues:")
        print(eig_conditional)

        print("\nCross-covariance magnitude ratio eigenvalues:")
        print(eig_pv)

        print("\nTrace ratios:")
        print(
            "Full centered:",
            np.trace(empirical_centered)
            / np.trace(predicted)
        )
        print(
            "Conditional centered:",
            np.trace(conditional_empirical)
            / np.trace(conditional_predicted)
        )
        print(
            "Conditional second moment:",
            np.trace(conditional_second_moment)
            / np.trace(conditional_predicted)
        )

        record = {
            "phase": phase,
            "samples_per_trial": int(np.sum(mask)),
            "trials": diag.TRIALS,
            "full_trace_ratio": float(
                np.trace(empirical_centered)
                / np.trace(predicted)
            ),
            "conditional_trace_ratio": float(
                np.trace(conditional_empirical)
                / np.trace(conditional_predicted)
            ),
            "conditional_second_moment_ratio": float(
                np.trace(conditional_second_moment)
                / np.trace(conditional_predicted)
            ),
            "full_eigen_min": float(np.min(eig_full)),
            "full_eigen_max": float(np.max(eig_full)),
            "conditional_eigen_min": float(
                np.min(eig_conditional)
            ),
            "conditional_eigen_max": float(
                np.max(eig_conditional)
            ),
        }

        for i in range(2):
            for j in range(2):
                record[f"predicted_corr_{i}_{j}"] = float(
                    predicted_corr[i, j + 2]
                )
                record[f"empirical_corr_{i}_{j}"] = float(
                    empirical_corr[i, j + 2]
                )
                record[f"predicted_pv_{i}_{j}"] = float(
                    predicted_pv[i, j]
                )
                record[f"empirical_pv_{i}_{j}"] = float(
                    empirical_pv[i, j]
                )

        records.append(record)

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nDiagnostic saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
