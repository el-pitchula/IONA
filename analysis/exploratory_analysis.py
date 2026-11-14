
from pathlib import Path
import csv
import json

import numpy as np


# ==========================================
# CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "exploratory"

CUBE_FILE = DATA_DIR / "precipitation_cube.npy"
COORDINATES_FILE = DATA_DIR / "coordinates.npz"

START_YEAR = 2001
END_YEAR = 2024


# ==========================================
# CARREGAMENTO
# ==========================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("IONA | EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    cube = np.load(CUBE_FILE)

    coordinates = np.load(COORDINATES_FILE)

    longitude = coordinates["longitude"]
    latitude = coordinates["latitude"]

    if cube.shape != (288, 25, 25):
        raise ValueError(
            f"Dimensões inesperadas: {cube.shape}"
        )

    print("\nDataset carregado.")
    print(f"Dimensões: {cube.shape}")

    # ======================================
    # PONDERAÇÃO ESPACIAL
    # ======================================

    # A área das células de uma grade regular
    # latitude/longitude varia com a latitude.
    # cos(latitude) é uma aproximação da área.

    weights = np.cos(
        np.deg2rad(latitude)
    )[None, :]

    weights = np.broadcast_to(
        weights,
        (len(longitude), len(latitude))
    )

    # ======================================
    # MÉDIA REGIONAL MENSAL
    # ======================================

    regional_monthly = []

    for month_data in cube:

        valid = np.isfinite(month_data)

        if not np.any(valid):
            regional_monthly.append(np.nan)
            continue

        weighted_mean = np.average(
            month_data[valid],
            weights=weights[valid]
        )

        regional_monthly.append(weighted_mean)

    regional_monthly = np.array(regional_monthly)

    # ======================================
    # CLIMATOLOGIA MENSAL
    # ======================================

    monthly_climatology = []

    for month in range(1, 13):

        values = regional_monthly[month - 1::12]

        monthly_climatology.append({
            "month": month,
            "mean_precipitation_mm": float(
                np.mean(values)
            ),
            "std_precipitation_mm": float(
                np.std(values, ddof=1)
            ),
            "minimum_mm": float(np.min(values)),
            "maximum_mm": float(np.max(values))
        })

    # ======================================
    # ACUMULAÇÃO ANUAL REGIONAL
    # ======================================

    annual_results = []

    for index, year in enumerate(
        range(START_YEAR, END_YEAR + 1)
    ):

        start = index * 12
        end = start + 12

        annual_total = np.sum(
            regional_monthly[start:end]
        )

        annual_results.append({
            "year": year,
            "annual_precipitation_mm": float(
                annual_total
            )
        })

    # ======================================
    # DIAGNÓSTICO DA DISTRIBUIÇÃO
    # ======================================

    valid_values = cube[np.isfinite(cube)]

    zero_count = np.count_nonzero(
        np.isclose(valid_values, 0.0)
    )

    diagnostics = {
        "total_observations": int(cube.size),
        "valid_observations": int(valid_values.size),
        "zero_observations": int(zero_count),
        "zero_percentage": float(
            zero_count / valid_values.size * 100
        ),
        "mean": float(np.mean(valid_values)),
        "median": float(np.median(valid_values)),
        "standard_deviation": float(
            np.std(valid_values)
        ),
        "percentile_95": float(
            np.percentile(valid_values, 95)
        ),
        "percentile_99": float(
            np.percentile(valid_values, 99)
        ),
        "minimum": float(np.min(valid_values)),
        "maximum": float(np.max(valid_values))
    }

    # ======================================
    # SALVAR CLIMATOLOGIA
    # ======================================

    climatology_file = (
        OUTPUT_DIR / "monthly_climatology.csv"
    )

    with open(
        climatology_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=[
                "month",
                "mean_precipitation_mm",
                "std_precipitation_mm",
                "minimum_mm",
                "maximum_mm"
            ]
        )

        writer.writeheader()
        writer.writerows(monthly_climatology)

    # ======================================
    # SALVAR SÉRIE ANUAL
    # ======================================

    annual_file = (
        OUTPUT_DIR / "annual_precipitation.csv"
    )

    with open(
        annual_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=[
                "year",
                "annual_precipitation_mm"
            ]
        )

        writer.writeheader()
        writer.writerows(annual_results)

    # ======================================
    # SALVAR DIAGNÓSTICO
    # ======================================

    diagnostics_file = (
        OUTPUT_DIR / "distribution_diagnostics.json"
    )

    with open(
        diagnostics_file,
        "w",
        encoding="utf-8"
    ) as output:

        json.dump(
            diagnostics,
            output,
            indent=4,
            ensure_ascii=False
        )

    # ======================================
    # RESULTADOS
    # ======================================

    print("\nCLIMATOLOGIA MENSAL")
    print("-" * 45)

    for item in monthly_climatology:

        print(
            f"Mês {item['month']:02d}: "
            f"{item['mean_precipitation_mm']:.2f} mm"
        )

    print("\nPRECIPITAÇÃO ANUAL")
    print("-" * 45)

    for item in annual_results:

        print(
            f"{item['year']}: "
            f"{item['annual_precipitation_mm']:.2f} mm"
        )

    print("\nDIAGNÓSTICO")
    print("-" * 45)

    for key, value in diagnostics.items():

        print(f"{key}: {value}")

    print("\nARQUIVOS GERADOS")
    print("-" * 45)

    print(climatology_file)
    print(annual_file)
    print(diagnostics_file)

    print("\nAnálise exploratória concluída.")


if __name__ == "__main__":
    main()
