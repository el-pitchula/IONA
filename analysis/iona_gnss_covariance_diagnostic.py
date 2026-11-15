import numpy as np
import iona_integrated_navigation as sim

TRIALS = 200
BASE_SEED = 20261200
LEVELS = [0, 5, 10, 25]

true_position, _, _ = sim.make_trajectory()

H_INV = np.linalg.inv(
    sim.H_GNSS.T @ sim.H_GNSS
) @ sim.H_GNSS.T

vtec = sim.make_tec_scenario("disturbed")

print("\nIONA | GNSS COVARIANCE CALIBRATION")
print("Scenario: disturbed")
print("Trials per TEC level:", TRIALS)

for error_percent in LEVELS:

    all_errors = []
    all_covariances = []

    for trial in range(TRIALS):

        rng = np.random.default_rng(BASE_SEED + trial)

        errors = []
        covariances = []

        for j, _ in enumerate(sim.GNSS_TIMES):

            current_vtec = max(float(vtec[j]), 0.1)

            satellite_vtec = current_vtec * (
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

            truth = true_position[sim.GNSS_INDICES[j]]

            errors.append(position_error[:2])
            covariances.append(position_cov)

        all_errors.append(errors)
        all_covariances.append(covariances)

    errors_array = np.asarray(all_errors)
    cov_array = np.asarray(all_covariances)

    empirical_covariances = []
    predicted_covariances = []
    nis_values = []
    epoch_biases = []

    for j in range(len(sim.GNSS_TIMES)):

        epoch_errors = errors_array[:, j, :]
        epoch_covariances = cov_array[:, j, :, :]

        empirical = np.cov(epoch_errors, rowvar=False)
        predicted = np.mean(epoch_covariances, axis=0)

        empirical_covariances.append(empirical)
        predicted_covariances.append(predicted)

        epoch_biases.append(np.mean(epoch_errors, axis=0))

        for trial in range(TRIALS):
            e = epoch_errors[trial]
            P = epoch_covariances[trial]

            nis_values.append(
                float(e @ np.linalg.solve(P, e))
            )

    empirical_mean = np.mean(empirical_covariances, axis=0)
    predicted_mean = np.mean(predicted_covariances, axis=0)

    ratios = np.linalg.eigvals(
        np.linalg.solve(predicted_mean, empirical_mean)
    ).real

    nis_values = np.asarray(nis_values)

    coverage = np.mean(
        (nis_values >= 0.0506) & (nis_values <= 7.3778)
    ) * 100

    bias_rms = np.sqrt(
        np.mean(np.sum(np.asarray(epoch_biases)**2, axis=1))
    )

    print("\n" + "=" * 58)
    print(f"TEC UNCERTAINTY: {error_percent}%")
    print("Empirical position covariance:")
    print(np.round(empirical_mean, 5))
    print("Predicted position covariance:")
    print(np.round(predicted_mean, 5))
    print("Empirical / predicted eigenvalue ratios:")
    print(np.round(ratios, 3))
    print(f"Mean normalized squared error: {nis_values.mean():.3f}")
    print(f"95% NIS coverage: {coverage:.2f}%")
    print(f"RMS epoch position bias: {bias_rms:.4f} m")

print("\nGNSS COVARIANCE DIAGNOSTIC COMPLETED")
