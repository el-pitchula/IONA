from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "processed" / "earth_mind_v2"
OUTPUT_DIR = ROOT / "public" / "data"

ANOMALY_FILE = INPUT_DIR / "temporal_anomalies_v2.csv"
MONTHLY_FILE = INPUT_DIR / "monthly_diagnostics_v2.csv"
SUMMARY_FILE = INPUT_DIR / "temporal_anomaly_summary_v2.json"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("IONA | EARTH-MIND v2 PUBLIC EXPORT")
print("=" * 45)

# --------------------------------------------------
# LOAD
# --------------------------------------------------

df = pd.read_csv(ANOMALY_FILE)
monthly = pd.read_csv(MONTHLY_FILE)

with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    original_summary = json.load(f)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

assert len(df) == 180000, f"Unexpected row count: {len(df)}"
assert df["date"].nunique() == 288
assert df[["longitude", "latitude"]].drop_duplicates().shape[0] == 625
assert not df.duplicated(["date", "longitude", "latitude"]).any()
assert monthly["observations"].sum() == 180000

df["occurrence_anomaly"] = df["occurrence_anomaly"].astype(int)
df["intensity_anomaly"] = df["intensity_anomaly"].astype(int)

anomalies = df[
    (df["occurrence_anomaly"] == 1) |
    (df["intensity_anomaly"] == 1)
].copy()

assert len(anomalies) == 5693, (
    f"Unexpected anomaly count: {len(anomalies)}"
)

print(f"[OK] Input records: {len(df)}")
print(f"[OK] Spatial cells: 625")
print(f"[OK] Anomaly records: {len(anomalies)}")

# --------------------------------------------------
# JSON CONVERSION
# --------------------------------------------------

def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return value


def clean_record(record):
    return {
        key: clean_value(value)
        for key, value in record.items()
    }


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            content,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )
    print(f"[OK] {path.name}")

# --------------------------------------------------
# 1. SUMMARY
# --------------------------------------------------

combined_counts = {
    str(key): int(value)
    for key, value in df["combined_class"].value_counts().items()
}

summary = {
    "product": "IONA EARTH-MIND",
    "version": "2.0",
    "description": (
        "Two-part statistical anomaly detection "
        "for precipitation occurrence and intensity."
    ),
    "period": {
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
        "years": int(df["year"].nunique()),
        "months": int(df["date"].nunique())
    },
    "spatial_coverage": {
        "cells": int(
            df[["longitude", "latitude"]]
            .drop_duplicates()
            .shape[0]
        ),
        "longitude_min": float(df["longitude"].min()),
        "longitude_max": float(df["longitude"].max()),
        "latitude_min": float(df["latitude"].min()),
        "latitude_max": float(df["latitude"].max())
    },
    "observations": {
        "total": int(len(df)),
        "wet": int(df["wet"].sum()),
        "dry": int(len(df) - df["wet"].sum())
    },
    "anomalies": {
        "total": int(len(anomalies)),
        "occurrence": int(df["occurrence_anomaly"].sum()),
        "intensity": int(df["intensity_anomaly"].sum()),
        "combined_classes": combined_counts
    },
    "methodology": {
        "reference": "Leave-one-year-out, same calendar month",
        "occurrence_model": "Jeffreys-smoothed Beta-Binomial predictive probability",
        "occurrence_threshold": 0.05,
        "intensity_model": "Median and MAD robust anomaly score",
        "intensity_threshold": 3.5,
        "minimum_wet_reference_years": 8,
        "multiple_testing_correction": False
    },
    "limitations": [
        "The occurrence threshold is operational and is not a multiple-testing-adjusted hypothesis test.",
        "Detected anomalies are statistical candidates, not independently confirmed extreme weather events.",
        "Insufficient evidence is kept separate from normal observations."
    ],
    "source_summary": original_summary
}

write_json(
    OUTPUT_DIR / "earth_mind_summary.json",
    summary
)

# --------------------------------------------------
# 2. MONTHLY DIAGNOSTICS
# --------------------------------------------------

monthly_records = []

for _, row in monthly.iterrows():
    month_number = int(row["month"])

    month_anomalies = df[df["month"] == month_number]

    monthly_records.append({
        "month": month_number,
        "observations": int(row["observations"]),
        "wet_observations": int(row["wet_observations"]),
        "wet_percentage": float(row["wet_percentage"]),
        "occurrence_anomalies": int(row["occurrence_anomalies"]),
        "intensity_anomalies": int(row["intensity_anomalies"]),
        "insufficient_intensity": int(row["insufficient_intensity"]),
        "no_anomaly_detected": int(
            (month_anomalies["combined_class"] ==
             "no_anomaly_detected").sum()
        )
    })

monthly_export = {
    "product": "IONA EARTH-MIND",
    "version": "2.0",
    "unit": "calendar_month",
    "months": monthly_records
}

write_json(
    OUTPUT_DIR / "earth_mind_monthly.json",
    monthly_export
)

# --------------------------------------------------
# 3. ANOMALIES GEOJSON
# --------------------------------------------------

features = []

for record in anomalies.to_dict(orient="records"):
    record = clean_record(record)

    longitude = float(record["longitude"])
    latitude = float(record["latitude"])

    properties = {
        key: value
        for key, value in record.items()
        if key not in ("longitude", "latitude")
    }

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [longitude, latitude]
        },
        "properties": properties
    })

geojson = {
    "type": "FeatureCollection",
    "name": "IONA_EARTH_MIND_v2_anomalies",
    "features": features
}

write_json(
    OUTPUT_DIR / "earth_mind_anomalies.geojson",
    geojson
)

# --------------------------------------------------
# 4. EARTH-MIND MANIFEST
# --------------------------------------------------

manifest_entry = {
    "version": "2.0",
    "summary": "earth_mind_summary.json",
    "monthly": "earth_mind_monthly.json",
    "anomalies_geojson": "earth_mind_anomalies.geojson",
    "anomaly_features": int(len(features)),
    "period_start": str(df["date"].min()),
    "period_end": str(df["date"].max())
}

if MANIFEST_FILE.exists():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
else:
    manifest = {}

manifest["earth_mind_v2"] = manifest_entry

write_json(MANIFEST_FILE, manifest)

# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------

with open(OUTPUT_DIR / "earth_mind_anomalies.geojson",
          "r", encoding="utf-8") as f:
    check_geojson = json.load(f)

assert len(check_geojson["features"]) == 5693
assert len(monthly_records) == 12

print("\n=== EXPORT VALIDATION ===")
print("Total source observations:", len(df))
print("Occurrence anomalies:", int(df["occurrence_anomaly"].sum()))
print("Intensity anomalies:", int(df["intensity_anomaly"].sum()))
print("GeoJSON features:", len(check_geojson["features"]))
print("Monthly records:", len(monthly_records))
print("Output directory:", OUTPUT_DIR)

print("\nEARTH-MIND v2 PUBLIC EXPORT COMPLETED SUCCESSFULLY")
