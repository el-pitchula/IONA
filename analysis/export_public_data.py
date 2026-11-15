
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# IONA | SCIENTIFIC DATA EXPORT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
EXPLORATORY = PROCESSED / "exploratory"
TRENDS = PROCESSED / "trends"
ROBUSTNESS = TRENDS / "robustness"

OUTPUT = ROOT / "public" / "data"
OUTPUT.mkdir(parents=True, exist_ok=True)

VERSION = "1.0.0"


# ============================================================
# UTILITIES
# ============================================================

def clean_value(value):
    """Convert pandas/numpy values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): clean_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]

    if isinstance(value, np.ndarray):
        return [clean_value(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if pd.isna(value):
        return None

    return value


def write_json(filename, content):
    path = OUTPUT / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            clean_value(content),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )

    print(f"[OK] {filename}")


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path)


def merge_coordinates(left, right, suffixes=("", "_right")):
    """Merge spatial tables using rounded coordinates."""

    left = left.copy()
    right = right.copy()

    left["_lon_key"] = left["longitude"].round(6)
    left["_lat_key"] = left["latitude"].round(6)

    right["_lon_key"] = right["longitude"].round(6)
    right["_lat_key"] = right["latitude"].round(6)

    merged = left.merge(
        right.drop(columns=["longitude", "latitude"]),
        on=["_lon_key", "_lat_key"],
        how="left",
        validate="one_to_one",
        suffixes=suffixes
    )

    return merged.drop(columns=["_lon_key", "_lat_key"])


# ============================================================
# 1. ANNUAL REGIONAL SERIES
# ============================================================

annual = read_csv(
    TRENDS / "regional_annual_series.csv"
)

annual_records = annual.to_dict(orient="records")

write_json(
    "annual_series.json",
    {
        "metadata": {
            "period": "2001-2024",
            "variable": "annual_precipitation",
            "unit": "mm",
            "aggregation": (
                "Sum of monthly accumulated precipitation "
                "from the cosine-latitude weighted regional series"
            )
        },
        "data": annual_records
    }
)


# ============================================================
# 2. MONTHLY CLIMATOLOGY
# ============================================================

monthly = read_csv(
    EXPLORATORY / "monthly_climatology.csv"
)

monthly_records = monthly.to_dict(orient="records")

write_json(
    "monthly_climatology.json",
    {
        "metadata": {
            "period": "2001-2024",
            "variable": "monthly_precipitation_climatology",
            "unit": "mm/month",
            "months": 12
        },
        "data": monthly_records
    }
)


# ============================================================
# 3. SPATIAL CELL DATA
# ============================================================

trends = read_csv(
    TRENDS / "cell_trends.csv"
)

lisa = read_csv(
    TRENDS / "local_spatial_clusters.csv"
)

queen_rook = read_csv(
    ROBUSTNESS / "queen_rook_cell_comparison.csv"
)

if len(trends) != 625:
    raise ValueError(
        f"Expected 625 trend cells, found {len(trends)}"
    )

if len(lisa) != 625:
    raise ValueError(
        f"Expected 625 LISA cells, found {len(lisa)}"
    )

if len(queen_rook) != 625:
    raise ValueError(
        f"Expected 625 Queen/Rook cells, found {len(queen_rook)}"
    )


# Avoid duplicating the Sen slope already present in trends.
lisa_columns = [
    "longitude",
    "latitude",
    "local_moran_i",
    "p_value",
    "q_value",
    "quadrant",
    "cluster",
    "significant_fdr"
]

lisa_export = lisa[lisa_columns].rename(
    columns={
        "p_value": "lisa_p_value",
        "q_value": "lisa_q_value",
        "significant_fdr": "lisa_significant_fdr"
    }
)

comparison_columns = [
    "longitude",
    "latitude",
    "queen_cluster",
    "rook_cluster",
    "queen_q_value",
    "rook_q_value"
]

comparison_export = queen_rook[comparison_columns]

spatial = merge_coordinates(trends, lisa_export)

spatial = merge_coordinates(
    spatial.rename(
        columns={
            "longitude": "longitude",
            "latitude": "latitude"
        }
    ),
    comparison_export
)

if spatial["cluster"].isna().any():
    raise ValueError("Missing LISA classifications after spatial merge.")

if spatial["rook_cluster"].isna().any():
    raise ValueError("Missing Rook classifications after spatial merge.")


# ============================================================
# 4. GEOJSON GENERATION
# ============================================================

features = []

for _, row in spatial.iterrows():

    longitude = float(row["longitude"])
    latitude = float(row["latitude"])

    properties = {
        column: clean_value(row[column])
        for column in spatial.columns
        if column not in ("longitude", "latitude")
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
    "name": "IONA_spatial_trend_analysis",
    "metadata": {
        "period": "2001-2024",
        "cells": len(features),
        "grid_resolution_degrees": 0.1,
        "trend_estimator": "Sen's slope",
        "spatial_clustering": "Local Moran's I (LISA)",
        "primary_weights": "Queen contiguity",
        "sensitivity_weights": "Rook contiguity"
    },
    "features": features
}

write_json(
    "spatial_cells.geojson",
    geojson
)


# ============================================================
# 5. SCIENTIFIC SUMMARIES
# ============================================================

trend_summary = read_json(
    TRENDS / "trend_summary.json"
)

spatial_summary = read_json(
    TRENDS / "spatial_autocorrelation.json"
)

lisa_summary = read_json(
    TRENDS / "local_spatial_summary.json"
)

cluster_contrast = read_json(
    TRENDS / "cluster_contrast.json"
)

distribution = read_json(
    EXPLORATORY / "distribution_diagnostics.json"
)

robustness = read_json(
    ROBUSTNESS / "queen_rook_summary.json"
)


write_json(
    "overview.json",
    {
        "project": "IONA",
        "version": VERSION,
        "research_question": (
            "Does the average precipitation in the Brazilian "
            "Cerrado hide opposing regional trends?"
        ),
        "study_area": {
            "name": "DF and surrounding pilot region",
            "west": -48.5,
            "south": -16.5,
            "east": -46.0,
            "north": -14.0,
            "is_official_cerrado_polygon": False
        },
        "dataset": {
            "product": "GPM_3IMERGM",
            "version": "V07B",
            "period": "2001-2024",
            "months": 288,
            "spatial_cells": 625,
            "resolution_degrees": 0.1,
            "doi": "10.5067/GPM/IMERG/3B-MONTH/07"
        },
        "regional_trend": trend_summary["regional_trend"],
        "spatial_results": trend_summary["spatial_results"],
        "spatial_autocorrelation": spatial_summary,
        "local_spatial_analysis": lisa_summary,
        "cluster_contrast": cluster_contrast,
        "robustness": robustness
    }
)


# ============================================================
# 6. QUEEN / ROOK COMPARISON
# ============================================================

write_json(
    "queen_rook_comparison.json",
    robustness
)


# ============================================================
# 7. DATA QUALITY
# ============================================================

quality = {
    "dataset": {
        "period": "2001-2024",
        "expected_months": 288,
        "observations": distribution["total_observations"],
        "valid_observations": distribution["valid_observations"],
        "missing_observations": (
            distribution["total_observations"]
            - distribution["valid_observations"]
        ),
        "zero_observations": distribution["zero_observations"],
        "zero_percentage": distribution["zero_percentage"]
    },
    "distribution": distribution,
    "source": {
        "provider": "NASA",
        "product": "GPM IMERG Monthly Final Run",
        "short_name": "GPM_3IMERGM",
        "version": "V07B",
        "doi": "10.5067/GPM/IMERG/3B-MONTH/07"
    },
    "processing": {
        "source_unit": "mm/hr",
        "accumulation_conversion": (
            "monthly_mean_rate_mm_hr * 24 * days_in_month"
        ),
        "spatial_weighting": "cosine of latitude"
    }
}

write_json(
    "data_quality.json",
    quality
)


# ============================================================
# 8. MANIFEST
# ============================================================

manifest = {
    "project": "IONA",
    "version": VERSION,
    "description": "Earth Intelligence & Pattern Discovery",
    "period": "2001-2024",
    "generated_datasets": [
        "overview.json",
        "annual_series.json",
        "monthly_climatology.json",
        "spatial_cells.geojson",
        "queen_rook_comparison.json",
        "data_quality.json"
    ],
    "spatial_cells": len(features),
    "source": {
        "provider": "NASA GPM",
        "product": "3IMERGM",
        "version": "V07B",
        "doi": "10.5067/GPM/IMERG/3B-MONTH/07"
    }
}

write_json("manifest.json", manifest)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("IONA | SCIENTIFIC DATA EXPORT")
print("=" * 60)

for path in sorted(OUTPUT.glob("*")):
    if path.is_file():
        print(
            f"{path.name:35} "
            f"{path.stat().st_size / 1024:10.2f} KB"
        )

print("\nSpatial features:", len(features))
print("Expected spatial features: 625")
print("Export directory:", OUTPUT)
print("\nEXPORT COMPLETED SUCCESSFULLY")
