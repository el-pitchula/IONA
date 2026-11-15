
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# IONA | EARTH-MIND v2.0
# Two-part precipitation anomaly detection
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CUBE_PATH = ROOT / "data/processed/precipitation_cube.npy"
COORD_PATH = ROOT / "data/processed/coordinates.npz"
TIME_PATH = ROOT / "data/processed/time_index.csv"

OUTPUT_DIR = ROOT / "data/processed/earth_mind_v2"

MIN_WET_YEARS = 8
ROBUST_Z_THRESHOLD = 3.5
OCCURRENCE_THRESHOLD = 0.05
JEFFREYS_ALPHA = 0.5
EPSILON = 1e-9


def load_data():
    cube = np.load(CUBE_PATH)
    coords = np.load(COORD_PATH)
    time_index = pd.read_csv(TIME_PATH)

    longitude = coords["longitude"]
    latitude = coords["latitude"]

    if cube.shape != (len(time_index), len(longitude), len(latitude)):
        raise ValueError(
            "Incompatible dimensions between precipitation cube, "
            "time index and coordinates."
        )

    if len(time_index) != 288:
        raise ValueError(f"Expected 288 observations, got {len(time_index)}.")

    if not np.isfinite(cube).all():
        raise ValueError("Precipitation cube contains NaN or infinite values.")

    if (cube < 0).any():
        raise ValueError("Negative precipitation values detected.")

    return cube, longitude, latitude, time_index


def classify_occurrence(wet, wet_reference, reference_n):
    """
    Jeffreys-smoothed posterior predictive probability.

    Beta-Binomial posterior:
        p | data ~ Beta(wet_reference + 0.5,
                        dry_reference + 0.5)

    Predictive probability of the observed binary outcome:
        P(wet) or P(dry)
    """
    dry_reference = reference_n - wet_reference

    alpha = wet_reference + JEFFREYS_ALPHA
    beta = dry_reference + JEFFREYS_ALPHA

    if wet:
        probability = alpha / (alpha + beta)
    else:
        probability = beta / (alpha + beta)

    if probability <= OCCURRENCE_THRESHOLD:
        classification = (
            "unusual_wet_occurrence"
            if wet
            else "unusual_dry_occurrence"
        )
    else:
        classification = "normal_occurrence"

    return probability, classification


def classify_intensity(value, reference_values):
    """
    Evaluate precipitation intensity conditional on positive rainfall.
    """
    n = len(reference_values)

    if n < MIN_WET_YEARS:
        return np.nan, np.nan, np.nan, "insufficient_sample"

    median = float(np.median(reference_values))
    mad = float(np.median(np.abs(reference_values - median)))

    if mad <= EPSILON:
        return median, mad, np.nan, "insufficient_dispersion"

    robust_scale = 1.4826 * mad
    robust_z = (value - median) / robust_scale

    if robust_z >= ROBUST_Z_THRESHOLD:
        classification = "high_intensity_anomaly"
    elif robust_z <= -ROBUST_Z_THRESHOLD:
        classification = "low_intensity_anomaly"
    else:
        classification = "normal_intensity"

    return median, mad, float(robust_z), classification


def main():
    print("=" * 60)
    print("IONA | EARTH-MIND v2.0")
    print("TWO-PART PRECIPITATION ANOMALY DETECTION")
    print("=" * 60)

    cube, longitude, latitude, time_index = load_data()

    n_time, n_lon, n_lat = cube.shape
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cube shape: {cube.shape}")
    print(f"Spatial cells: {n_lon * n_lat}")
    print(f"Observations: {n_time}")
    print(f"Period: {time_index['year'].min()}–{time_index['year'].max()}")

    records = []

    for t in range(n_time):
        year = int(time_index.iloc[t]["year"])
        month = int(time_index.iloc[t]["month"])
        date = str(time_index.iloc[t]["date"])

        # Same calendar month, excluding the evaluated year.
        reference_indices = np.where(
            (time_index["month"].to_numpy() == month)
            & (time_index["year"].to_numpy() != year)
        )[0]

        for i in range(n_lon):
            for j in range(n_lat):

                value = float(cube[t, i, j])
                wet = value > 0.0

                reference = cube[reference_indices, i, j]
                reference_wet = reference > 0.0

                reference_n = len(reference)
                wet_reference_n = int(reference_wet.sum())
                dry_reference_n = reference_n - wet_reference_n

                occurrence_probability, occurrence_class = (
                    classify_occurrence(
                        wet,
                        wet_reference_n,
                        reference_n
                    )
                )

                if wet:
                    positive_reference = reference[reference_wet]

                    (
                        intensity_median,
                        intensity_mad,
                        robust_z,
                        intensity_class
                    ) = classify_intensity(
                        value,
                        positive_reference
                    )
                else:
                    intensity_median = np.nan
                    intensity_mad = np.nan
                    robust_z = np.nan
                    intensity_class = "not_applicable"

                occurrence_anomaly = occurrence_class in (
                    "unusual_wet_occurrence",
                    "unusual_dry_occurrence"
                )

                intensity_anomaly = intensity_class in (
                    "high_intensity_anomaly",
                    "low_intensity_anomaly"
                )

                if occurrence_anomaly and intensity_anomaly:
                    combined_class = "occurrence_and_intensity_anomaly"
                elif occurrence_anomaly:
                    combined_class = "occurrence_anomaly"
                elif intensity_anomaly:
                    combined_class = "intensity_anomaly"
                elif intensity_class in (
                    "insufficient_sample",
                    "insufficient_dispersion"
                ):
                    combined_class = "insufficient_intensity_evidence"
                else:
                    combined_class = "no_anomaly_detected"

                records.append({
                    "date": date,
                    "year": year,
                    "month": month,
                    "longitude": float(longitude[i]),
                    "latitude": float(latitude[j]),
                    "precipitation_mm": value,
                    "wet": bool(wet),
                    "reference_years": reference_n,
                    "wet_reference_years": wet_reference_n,
                    "dry_reference_years": dry_reference_n,
                    "occurrence_probability": occurrence_probability,
                    "occurrence_class": occurrence_class,
                    "intensity_reference_n": (
                        int(len(reference[reference_wet]))
                        if wet else 0
                    ),
                    "intensity_baseline_median_mm": intensity_median,
                    "intensity_baseline_mad_mm": intensity_mad,
                    "robust_z_score": robust_z,
                    "intensity_class": intensity_class,
                    "occurrence_anomaly": occurrence_anomaly,
                    "intensity_anomaly": intensity_anomaly,
                    "combined_class": combined_class
                })

        if (t + 1) % 12 == 0:
            print(
                f"[PROGRESS] {t + 1}/{n_time} observations processed"
            )

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Export detailed observations
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "temporal_anomalies_v2.csv"
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # Monthly diagnostics
    # --------------------------------------------------------

    monthly = (
        df.groupby("month")
        .agg(
            observations=("wet", "size"),
            wet_observations=("wet", "sum"),
            occurrence_anomalies=("occurrence_anomaly", "sum"),
            intensity_anomalies=("intensity_anomaly", "sum"),
            insufficient_intensity=(
                "intensity_class",
                lambda x: x.isin([
                    "insufficient_sample",
                    "insufficient_dispersion"
                ]).sum()
            )
        )
        .reset_index()
    )

    monthly["wet_percentage"] = (
        monthly["wet_observations"]
        / monthly["observations"] * 100
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_diagnostics_v2.csv",
        index=False
    )

    # --------------------------------------------------------
    # Classification distribution
    # --------------------------------------------------------

    combined_distribution = (
        df["combined_class"]
        .value_counts()
        .to_dict()
    )

    occurrence_distribution = (
        df["occurrence_class"]
        .value_counts()
        .to_dict()
    )

    intensity_distribution = (
        df["intensity_class"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "project": "IONA",
        "module": "EARTH-MIND",
        "version": "2.0",
        "method": "Two-part occurrence and conditional intensity detection",
        "observations": int(len(df)),
        "spatial_cells": int(n_lon * n_lat),
        "years": [
            int(time_index["year"].min()),
            int(time_index["year"].max())
        ],
        "parameters": {
            "minimum_wet_reference_years": MIN_WET_YEARS,
            "robust_z_threshold": ROBUST_Z_THRESHOLD,
            "occurrence_predictive_probability_threshold": (
                OCCURRENCE_THRESHOLD
            ),
            "beta_prior": {
                "alpha": JEFFREYS_ALPHA,
                "beta": JEFFREYS_ALPHA
            },
            "reference_strategy": "leave-one-year-out",
            "intensity_reference": "positive precipitation only"
        },
        "counts": {
            "wet_observations": int(df["wet"].sum()),
            "occurrence_anomalies": int(
                df["occurrence_anomaly"].sum()
            ),
            "intensity_anomalies": int(
                df["intensity_anomaly"].sum()
            ),
            "combined_classes": {
                str(k): int(v)
                for k, v in combined_distribution.items()
            }
        },
        "occurrence_class_distribution": {
            str(k): int(v)
            for k, v in occurrence_distribution.items()
        },
        "intensity_class_distribution": {
            str(k): int(v)
            for k, v in intensity_distribution.items()
        },
        "monthly_diagnostics": monthly.to_dict(orient="records"),
        "methodological_notes": [
            "Occurrence and intensity are evaluated separately.",
            "Occurrence uses a Jeffreys-smoothed Beta-Binomial posterior predictive probability.",
            "Intensity uses positive historical observations only.",
            "Intensity references with fewer than the minimum sample are not classified.",
            "Zero-MAD intensity references are not classified as anomalies.",
            "Anomaly thresholds are operational criteria, not proof of physical extremes.",
            "Spatial coherence is not evaluated in this version."
        ]
    }

    json_path = OUTPUT_DIR / "temporal_anomaly_summary_v2.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n=== OCCURRENCE CLASS DISTRIBUTION ===")
    print(df["occurrence_class"].value_counts().to_string())

    print("\n=== INTENSITY CLASS DISTRIBUTION ===")
    print(df["intensity_class"].value_counts().to_string())

    print("\n=== COMBINED CLASS DISTRIBUTION ===")
    print(df["combined_class"].value_counts().to_string())

    print("\n=== MONTHLY DIAGNOSTICS ===")
    print(monthly.to_string(index=False))

    print("\n=== OUTPUT FILES ===")
    print(csv_path)
    print(json_path)
    print(OUTPUT_DIR / "monthly_diagnostics_v2.csv")

    print("\nEARTH-MIND v2.0 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
