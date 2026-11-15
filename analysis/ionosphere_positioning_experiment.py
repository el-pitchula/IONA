import numpy as np
import csv
from pathlib import Path

# ============================================================
# IONA | IONOSPHERIC POSITIONING ERROR EXPERIMENT
# Single-frequency vs TEC correction vs dual-frequency GNSS
# ============================================================

SEED = 20261115
TRIALS = 1000

F1 = 1575.42e6
F2 = 1227.60e6
K = 40.3

RE = 6371.0
H_IONO = 350.0

CODE_NOISE = 0.30
TEC_SCENARIOS = [5, 10, 20, 30, 50]

# Six simulated satellites with different azimuth/elevation angles.
AZIMUTH_DEG = np.array([0, 60, 120, 180, 240, 300])
ELEVATION_DEG = np.array([25, 35, 45, 55, 65, 70])

az = np.radians(AZIMUTH_DEG)
el = np.radians(ELEVATION_DEG)

def mapping_factor(elevation):
    ratio = RE * np.cos(elevation) / (RE + H_IONO)
    return 1.0 / np.sqrt(1.0 - ratio**2)

mapping = mapping_factor(el)

# Linearized observation model:
# unknowns = east error, north error, receiver clock error
H = np.column_stack((
    -np.cos(el) * np.sin(az),
    -np.cos(el) * np.cos(az),
    np.ones(len(az))
))

OUTPUT = Path("results/ionosphere")
OUTPUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


def delay(tec_electrons, frequency):
    return K * tec_electrons / frequency**2


def solve_position(range_errors):
    solution, _, _, _ = np.linalg.lstsq(H, range_errors, rcond=None)
    horizontal_error = np.linalg.norm(solution[:2])
    return float(horizontal_error), solution


results = []

print("\n" + "=" * 82)
print("IONA | IONOSPHERIC POSITIONING ERROR EXPERIMENT")
print("=" * 82)
print(f"Trials per scenario: {TRIALS}")
print(f"Satellites: {len(AZIMUTH_DEG)}")
print(f"Code measurement noise: {CODE_NOISE:.2f} m")
print("\nHORIZONTAL POSITION ERROR")
print("-" * 82)

for nominal_vtec in TEC_SCENARIOS:
    errors_l1 = []
    errors_corrected = []
    errors_if = []

    for _ in range(TRIALS):
        # Small satellite-to-satellite TEC variation.
        satellite_vtec = nominal_vtec * (
            1 + rng.normal(0, 0.05, len(az))
        )
        satellite_vtec = np.maximum(satellite_vtec, 0.1)

        slant_tec = satellite_vtec * mapping * 1e16

        iono_l1 = delay(slant_tec, F1)
        iono_l2 = delay(slant_tec, F2)

        noise_l1 = rng.normal(0, CODE_NOISE, len(az))
        noise_l2 = rng.normal(0, CODE_NOISE, len(az))

        # L1: uncorrected first-order ionospheric delay.
        observed_l1 = iono_l1 + noise_l1

        # L1: ideal correction using exact simulated TEC.
        corrected_l1 = observed_l1 - iono_l1

        # Dual-frequency ionosphere-free combination.
        observed_l2 = iono_l2 + noise_l2

        observed_if = (
            F1**2 * observed_l1 - F2**2 * observed_l2
        ) / (F1**2 - F2**2)

        errors_l1.append(solve_position(observed_l1)[0])
        errors_corrected.append(solve_position(corrected_l1)[0])
        errors_if.append(solve_position(observed_if)[0])

    row = {
        "vtec_tecu": nominal_vtec,
        "l1_mean_m": np.mean(errors_l1),
        "l1_p95_m": np.percentile(errors_l1, 95),
        "tec_corrected_mean_m": np.mean(errors_corrected),
        "tec_corrected_p95_m": np.percentile(errors_corrected, 95),
        "dual_frequency_mean_m": np.mean(errors_if),
        "dual_frequency_p95_m": np.percentile(errors_if, 95)
    }

    results.append(row)

    print(
        f"{nominal_vtec:5.1f} TECU | "
        f"L1: {row['l1_mean_m']:7.3f} m "
        f"(P95 {row['l1_p95_m']:7.3f}) | "
        f"TEC corrected: {row['tec_corrected_mean_m']:7.3f} m | "
        f"L1+L2: {row['dual_frequency_mean_m']:7.3f} m"
    )

csv_path = OUTPUT / "positioning_ionosphere_results.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\nSatellite geometry:")
for i in range(len(az)):
    print(
        f"Satellite {i+1}: "
        f"azimuth={AZIMUTH_DEG[i]} deg, "
        f"elevation={ELEVATION_DEG[i]} deg, "
        f"mapping={mapping[i]:.3f}"
    )

print("\nCSV saved:", csv_path.resolve())
print("POSITIONING EXPERIMENT COMPLETED")
