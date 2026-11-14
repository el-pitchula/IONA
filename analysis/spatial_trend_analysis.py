
from pathlib import Path
import csv
import json

import numpy as np
from scipy.stats import theilslopes, kendalltau


# ==========================================
# CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "trends"

CUBE_FILE = DATA_DIR / "precipitation_cube.npy"
COORDINATES_FILE = DATA_DIR / "coordinates.npz"

START_YEAR = 2001
END_YEAR = 2024

ALPHA = 0.05


# ==========================================
# BENJAMINI-HOCHBERG (FDR)
# ==========================================

def benjamini_hochberg(p_values, alpha=0.05):

    p_values = np.asarray(p_values, dtype=float)

    q_values = np.full(p_values.shape, np.nan)

    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return q_values

    valid_p = p_values[valid_indices]

    order = np.argsort(valid_p)

    sorted_p = valid_p[order]

    n = len(sorted_p)

    ranks = np.arange(1, n + 1)

    adjusted = sorted_p * n / ranks

    # Garantir monotonicidade dos valores ajustados.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    sorted_q = np.empty(n)
    sorted_q[order] = adjusted

    q_values[valid_indices] = sorted_q

    return q_values


# ==========================================
# AUTOCORRELAÇÃO LAG-1
# ==========================================

def lag1_autocorrelation(series):

    series = np.asarray(series, dtype=float)

    if len(series) < 3:
        return np.nan

    first = series[:-1]
    second = series[1:]

    if np.std(first) == 0 or np.std(second) == 0:
        return np.nan

    return float(np.corrcoef(first, second)[0, 1])


# ==========================================
# ANÁLISE DE UMA CÉLULA
# ==========================================

def analyze_cell(series, years):

    series = np.asarray(series, dtype=float)

    valid = np.isfinite(series)

    if valid.sum() < 10:
        return None

    values = series[valid]
    valid_years = years[valid]

    if np.unique(values).size < 2:
        return None

    sen = theilslopes(
        values,
        valid_years,
        alpha=0.95
    )

    mk = kendalltau(
        valid_years,
        values,
        alternative="two-sided"
    )

    return {
        "sen_slope": float(sen.slope),
        "sen_ci_lower": float(sen.low_slope),
        "sen_ci_upper": float(sen.high_slope),
        "kendall_tau": float(mk.statistic),
        "p_value": float(mk.pvalue),
        "lag1_autocorrelation": lag1_autocorrelation(values),
        "observations": int(valid.sum())
    }


# ==========================================
# MAIN
# ==========================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("IONA | SPATIAL TREND ANALYSIS")
    print("=" * 60)

    # --------------------------------------
    # Carregar dados
    # --------------------------------------

    cube = np.load(CUBE_FILE)

    coordinates = np.load(COORDINATES_FILE)

    longitude = coordinates["longitude"]
    latitude = coordinates["latitude"]

    years = np.arange(
        START_YEAR,
        END_YEAR + 1
    )

    expected_shape = (288, 25, 25)

    if cube.shape != expected_shape:
        raise ValueError(
            f"Dimensão inesperada: {cube.shape}"
        )

    print(f"\nCubo original: {cube.shape}")

    # --------------------------------------
    # Agregação anual por célula
    # --------------------------------------

    annual_cube = cube.reshape(
        24, 12, 25, 25
    ).sum(axis=1)

    print(
        f"Cubo anual: {annual_cube.shape}"
    )

    # --------------------------------------
    # Tendência regional agregada
    # --------------------------------------

    weights = np.cos(
        np.deg2rad(latitude)
    )[None, :]

    weights = np.broadcast_to(
        weights,
        (len(longitude), len(latitude))
    )

    regional_annual = []

    for annual_map in annual_cube:

        valid = np.isfinite(annual_map)

        regional_mean = np.average(
            annual_map[valid],
            weights=weights[valid]
        )

        regional_annual.append(regional_mean)

    regional_annual = np.asarray(regional_annual)

    regional_sen = theilslopes(
        regional_annual,
        years,
        alpha=0.95
    )

    regional_mk = kendalltau(
        years,
        regional_annual,
        alternative="two-sided"
    )

    # --------------------------------------
    # Análise espacial célula por célula
    # --------------------------------------

    print("\nCalculando tendências espaciais...\n")

    results = []

    p_values = []

    for i, lon in enumerate(longitude):

        for j, lat in enumerate(latitude):

            series = annual_cube[:, i, j]

            result = analyze_cell(
                series,
                years
            )

            if result is None:
                continue

            record = {
                "longitude": float(lon),
                "latitude": float(lat),
                **result
            }

            results.append(record)

            p_values.append(result["p_value"])

        if (i + 1) % 5 == 0:
            print(
                f"Colunas processadas: {i + 1}/"
                f"{len(longitude)}"
            )

    # --------------------------------------
    # Correção FDR
    # --------------------------------------

    q_values = benjamini_hochberg(
        p_values,
        ALPHA
    )

    for record, q_value in zip(results, q_values):

        record["q_value"] = float(q_value)

        record["significant_fdr"] = bool(
            q_value < ALPHA
        )

        if record["significant_fdr"]:
            if record["sen_slope"] > 0:
                record["trend_class"] = "increasing"
            elif record["sen_slope"] < 0:
                record["trend_class"] = "decreasing"
            else:
                record["trend_class"] = "neutral"
        else:
            record["trend_class"] = "not_significant"

    # --------------------------------------
    # Estatísticas espaciais
    # --------------------------------------

    significant = [
        item for item in results
        if item["significant_fdr"]
    ]

    increasing = [
        item for item in significant
        if item["trend_class"] == "increasing"
    ]

    decreasing = [
        item for item in significant
        if item["trend_class"] == "decreasing"
    ]

    lag_values = [
        item["lag1_autocorrelation"]
        for item in results
        if np.isfinite(item["lag1_autocorrelation"])
    ]

    summary = {
        "period": f"{START_YEAR}-{END_YEAR}",
        "annual_observations_per_cell": 24,
        "total_spatial_cells": len(results),
        "alpha": ALPHA,
        "regional_trend": {
            "sen_slope_mm_per_year": float(
                regional_sen.slope
            ),
            "confidence_interval_95": [
                float(regional_sen.low_slope),
                float(regional_sen.high_slope)
            ],
            "kendall_tau": float(
                regional_mk.statistic
            ),
            "p_value": float(
                regional_mk.pvalue
            )
        },
        "spatial_results": {
            "significant_cells_fdr": len(significant),
            "increasing_cells": len(increasing),
            "decreasing_cells": len(decreasing),
            "not_significant_cells": (
                len(results) - len(significant)
            ),
            "mean_lag1_autocorrelation": (
                float(np.mean(lag_values))
                if lag_values else None
            )
        }
    }

    # --------------------------------------
    # Exportar tendências por célula
    # --------------------------------------

    csv_file = OUTPUT_DIR / "cell_trends.csv"

    fields = [
        "longitude",
        "latitude",
        "sen_slope",
        "sen_ci_lower",
        "sen_ci_upper",
        "kendall_tau",
        "p_value",
        "q_value",
        "significant_fdr",
        "trend_class",
        "lag1_autocorrelation",
        "observations"
    ]

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------
    # Exportar série regional
    # --------------------------------------

    regional_file = (
        OUTPUT_DIR / "regional_annual_series.csv"
    )

    with open(
        regional_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.writer(output)

        writer.writerow([
            "year",
            "precipitation_mm"
        ])

        for year, value in zip(
            years,
            regional_annual
        ):

            writer.writerow([
                int(year),
                float(value)
            ])

    # --------------------------------------
    # Exportar resumo
    # --------------------------------------

    summary_file = OUTPUT_DIR / "trend_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as output:

        json.dump(
            summary,
            output,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------
    # Relatório no terminal
    # --------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADOS DAS TENDÊNCIAS")
    print("=" * 60)

    print("\nTENDÊNCIA REGIONAL")

    print(
        f"Sen's slope: "
        f"{regional_sen.slope:.3f} mm/ano por ano"
    )

    print(
        f"IC 95%: "
        f"[{regional_sen.low_slope:.3f}, "
        f"{regional_sen.high_slope:.3f}]"
    )

    print(
        f"Kendall tau: "
        f"{regional_mk.statistic:.4f}"
    )

    print(
        f"p-valor: {regional_mk.pvalue:.6f}"
    )

    print("\nRESULTADOS ESPACIAIS")

    print(f"Células analisadas: {len(results)}")

    print(
        f"Tendências significativas após FDR: "
        f"{len(significant)}"
    )

    print(f"Tendências crescentes: {len(increasing)}")
    print(f"Tendências decrescentes: {len(decreasing)}")

    print(
        f"Sem significância após FDR: "
        f"{len(results) - len(significant)}"
    )

    print(
        "\nAutocorrelação lag-1 média: "
        f"{summary['spatial_results']['mean_lag1_autocorrelation']}"
    )

    print("\nARQUIVOS GERADOS")

    print(csv_file)
    print(regional_file)
    print(summary_file)

    print("\nAnálise de tendências concluída.")

    print(
        "\nNOTA: os p-valores do Mann-Kendall são "
        "preliminares até avaliarmos a autocorrelação "
        "temporal e a adequação do teste."
    )


if __name__ == "__main__":
    main()
