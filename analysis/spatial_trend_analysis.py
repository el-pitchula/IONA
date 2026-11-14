
from pathlib import Path
import csv
import json

import numpy as np
from scipy.stats import theilslopes, kendalltau, pearsonr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "trends"

CUBE_FILE = DATA_DIR / "precipitation_cube.npy"
COORDINATES_FILE = DATA_DIR / "coordinates.npz"

START_YEAR = 2001
END_YEAR = 2024

ALPHA = 0.05
MIN_OBSERVATIONS = 10


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================

def benjamini_hochberg(p_values):

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

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    sorted_q = np.empty(n)
    sorted_q[order] = adjusted

    q_values[valid_indices] = sorted_q

    return q_values


# ============================================================
# LAG-1 AUTOCORRELATION
# ============================================================

def lag1_autocorrelation(series):

    series = np.asarray(series, dtype=float)

    if len(series) < 3:
        return np.nan

    first = series[:-1]
    second = series[1:]

    if np.std(first) == 0 or np.std(second) == 0:
        return np.nan

    return float(np.corrcoef(first, second)[0, 1])


# ============================================================
# TREND-FREE PRE-WHITENING
# ============================================================

def trend_free_prewhitening(values, years):

    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=float)

    result = {
        "residual_lag1": np.nan,
        "tfpw_tau": np.nan,
        "tfpw_p_value": np.nan,
        "tfpw_observations": 0
    }

    if len(values) < MIN_OBSERVATIONS:
        return result

    # TFPW requires consecutive annual observations.
    if not np.all(np.diff(years) == 1):
        return result

    # Estimate the original monotonic trend.
    sen = theilslopes(
        values,
        years,
        alpha=0.95
    )

    # Remove the estimated trend.
    trend = sen.intercept + sen.slope * years
    residuals = values - trend

    # Estimate lag-1 autocorrelation on detrended residuals.
    r1 = lag1_autocorrelation(residuals)

    result["residual_lag1"] = (
        float(r1) if np.isfinite(r1) else np.nan
    )

    if not np.isfinite(r1):
        return result

    # Pre-whiten the detrended residuals.
    whitened_residuals = (
        residuals[1:] - r1 * residuals[:-1]
    )

    # Reintroduce the original estimated trend.
    tfpw_values = (
        trend[1:] + whitened_residuals
    )

    # Apply Mann-Kendall to the adjusted series.
    tfpw_mk = kendalltau(
        years[1:],
        tfpw_values,
        alternative="two-sided"
    )

    result["tfpw_tau"] = float(tfpw_mk.statistic)
    result["tfpw_p_value"] = float(tfpw_mk.pvalue)
    result["tfpw_observations"] = len(tfpw_values)

    return result


# ============================================================
# ANALYSIS OF ONE TIME SERIES
# ============================================================

def analyze_series(series, years):

    series = np.asarray(series, dtype=float)
    years = np.asarray(years, dtype=float)

    valid = np.isfinite(series)

    if valid.sum() < MIN_OBSERVATIONS:
        return None

    values = series[valid]
    valid_years = years[valid]

    if np.unique(values).size < 2:
        return None

    # Original Sen's slope.
    sen = theilslopes(
        values,
        valid_years,
        alpha=0.95
    )

    # Conventional Mann-Kendall.
    mk = kendalltau(
        valid_years,
        values,
        alternative="two-sided"
    )

    # Original-series autocorrelation.
    raw_lag1 = lag1_autocorrelation(values)

    # Trend-free pre-whitening sensitivity analysis.
    tfpw = trend_free_prewhitening(
        values,
        valid_years
    )

    return {
        "sen_slope": float(sen.slope),
        "sen_ci_lower": float(sen.low_slope),
        "sen_ci_upper": float(sen.high_slope),

        "kendall_tau": float(mk.statistic),
        "p_value": float(mk.pvalue),

        "raw_lag1_autocorrelation": raw_lag1,

        "residual_lag1_autocorrelation": (
            tfpw["residual_lag1"]
        ),

        "tfpw_kendall_tau": tfpw["tfpw_tau"],
        "tfpw_p_value": tfpw["tfpw_p_value"],
        "tfpw_observations": tfpw["tfpw_observations"],

        "observations": int(valid.sum())
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("IONA | AUTOCORRELATION AND TREND ANALYSIS")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    cube = np.load(CUBE_FILE)

    coordinates = np.load(COORDINATES_FILE)

    longitude = coordinates["longitude"]
    latitude = coordinates["latitude"]

    years = np.arange(
        START_YEAR,
        END_YEAR + 1
    )

    expected_shape = (
        len(years) * 12,
        len(longitude),
        len(latitude)
    )

    if cube.shape != expected_shape:
        raise ValueError(
            f"Unexpected cube dimensions: {cube.shape}. "
            f"Expected: {expected_shape}"
        )

    print(f"\nOriginal cube: {cube.shape}")

    # --------------------------------------------------------
    # ANNUAL AGGREGATION
    # --------------------------------------------------------

    monthly_cube = cube.reshape(
        len(years),
        12,
        len(longitude),
        len(latitude)
    )

    valid_months = np.isfinite(monthly_cube).sum(axis=1)

    annual_cube = np.nansum(
        monthly_cube,
        axis=1
    )

    # Do not treat incomplete years as complete annual totals.
    annual_cube[valid_months < 12] = np.nan

    print(f"Annual cube: {annual_cube.shape}")

    # --------------------------------------------------------
    # AREA-WEIGHTED REGIONAL ANNUAL SERIES
    # --------------------------------------------------------

    weights = np.cos(
        np.deg2rad(latitude)
    )[None, :]

    weights = np.broadcast_to(
        weights,
        (len(longitude), len(latitude))
    )

    regional_annual = []

    for annual_map in annual_cube:

        valid = (
            np.isfinite(annual_map)
            & np.isfinite(weights)
        )

        if not np.any(valid):
            regional_annual.append(np.nan)
            continue

        regional_mean = np.average(
            annual_map[valid],
            weights=weights[valid]
        )

        regional_annual.append(regional_mean)

    regional_annual = np.asarray(regional_annual)

    regional_result = analyze_series(
        regional_annual,
        years
    )

    if regional_result is None:
        raise ValueError(
            "Insufficient valid observations "
            "for regional analysis."
        )

    # --------------------------------------------------------
    # SPATIAL CELL-BY-CELL ANALYSIS
    # --------------------------------------------------------

    print("\nProcessing spatial trends...\n")

    results = []

    for i, lon in enumerate(longitude):

        for j, lat in enumerate(latitude):

            series = annual_cube[:, i, j]

            result = analyze_series(
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

        if (i + 1) % 5 == 0:
            print(
                f"Columns processed: "
                f"{i + 1}/{len(longitude)}"
            )

    if not results:
        raise ValueError(
            "No valid spatial cells were analyzed."
        )

    # --------------------------------------------------------
    # FDR: CONVENTIONAL MANN-KENDALL
    # --------------------------------------------------------

    raw_p_values = [
        item["p_value"]
        for item in results
    ]

    raw_q_values = benjamini_hochberg(
        raw_p_values
    )

    # --------------------------------------------------------
    # FDR: TFPW
    # --------------------------------------------------------

    tfpw_p_values = [
        item["tfpw_p_value"]
        for item in results
    ]

    tfpw_q_values = benjamini_hochberg(
        tfpw_p_values
    )

    # --------------------------------------------------------
    # CLASSIFICATION AND COMPARISON
    # --------------------------------------------------------

    for record, raw_q, tfpw_q in zip(
        results,
        raw_q_values,
        tfpw_q_values
    ):

        record["q_value"] = float(raw_q)

        record["tfpw_q_value"] = float(tfpw_q)

        raw_significant = (
            np.isfinite(raw_q)
            and raw_q < ALPHA
        )

        tfpw_significant = (
            np.isfinite(tfpw_q)
            and tfpw_q < ALPHA
        )

        record["significant_fdr"] = bool(
            raw_significant
        )

        record["significant_tfpw_fdr"] = bool(
            tfpw_significant
        )

        if raw_significant and tfpw_significant:
            comparison = "significant_both"

        elif raw_significant:
            comparison = "only_conventional"

        elif tfpw_significant:
            comparison = "only_tfpw"

        else:
            comparison = "not_significant"

        record["test_comparison"] = comparison

        if comparison == "significant_both":
            slope = record["sen_slope"]

            if slope > 0:
                trend_class = "increasing"
            elif slope < 0:
                trend_class = "decreasing"
            else:
                trend_class = "neutral"
        else:
            trend_class = "not_significant"

        record["trend_class"] = trend_class

    # --------------------------------------------------------
    # SPATIAL SUMMARY
    # --------------------------------------------------------

    significant_raw = [
        item for item in results
        if item["significant_fdr"]
    ]

    significant_tfpw = [
        item for item in results
        if item["significant_tfpw_fdr"]
    ]

    significant_both = [
        item for item in results
        if item["test_comparison"] == "significant_both"
    ]

    only_conventional = [
        item for item in results
        if item["test_comparison"] == "only_conventional"
    ]

    only_tfpw = [
        item for item in results
        if item["test_comparison"] == "only_tfpw"
    ]

    increasing = [
        item for item in significant_both
        if item["trend_class"] == "increasing"
    ]

    decreasing = [
        item for item in significant_both
        if item["trend_class"] == "decreasing"
    ]

    raw_lags = [
        item["raw_lag1_autocorrelation"]
        for item in results
        if np.isfinite(item["raw_lag1_autocorrelation"])
    ]

    residual_lags = [
        item["residual_lag1_autocorrelation"]
        for item in results
        if np.isfinite(
            item["residual_lag1_autocorrelation"]
        )
    ]

    # --------------------------------------------------------
    # REGIONAL SUMMARY
    # --------------------------------------------------------

    regional_summary = {
        "sen_slope_mm_per_year_per_year": (
            regional_result["sen_slope"]
        ),

        "sen_ci_95": [
            regional_result["sen_ci_lower"],
            regional_result["sen_ci_upper"]
        ],

        "kendall_tau": regional_result["kendall_tau"],
        "conventional_p_value": regional_result["p_value"],

        "raw_lag1_autocorrelation": (
            regional_result["raw_lag1_autocorrelation"]
        ),

        "residual_lag1_autocorrelation": (
            regional_result["residual_lag1_autocorrelation"]
        ),

        "tfpw_kendall_tau": (
            regional_result["tfpw_kendall_tau"]
        ),

        "tfpw_p_value": (
            regional_result["tfpw_p_value"]
        ),

        "observations": regional_result["observations"]
    }

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = {
        "period": f"{START_YEAR}-{END_YEAR}",
        "annual_observations_per_cell": 24,
        "total_spatial_cells": len(results),
        "alpha": ALPHA,

        "methodology": {
            "trend_estimator": "Sen's slope",
            "conventional_test": "Mann-Kendall",
            "sensitivity_test": (
                "Trend-Free Pre-Whitening (TFPW)"
            ),
            "multiple_testing": (
                "Benjamini-Hochberg FDR"
            ),
            "regional_weighting": (
                "Cosine of latitude"
            ),
            "annual_aggregation": (
                "Sum of 12 monthly accumulated values"
            )
        },

        "regional_trend": regional_summary,

        "spatial_results": {
            "conventional_significant_fdr": (
                len(significant_raw)
            ),

            "tfpw_significant_fdr": (
                len(significant_tfpw)
            ),

            "significant_in_both_tests": (
                len(significant_both)
            ),

            "only_conventional_significant": (
                len(only_conventional)
            ),

            "only_tfpw_significant": (
                len(only_tfpw)
            ),

            "increasing_cells_robust": len(increasing),
            "decreasing_cells_robust": len(decreasing),

            "not_significant_in_both": (
                len(results) - len(significant_raw)
                - len(only_tfpw)
            ),

            "mean_raw_lag1_autocorrelation": (
                float(np.mean(raw_lags))
                if raw_lags else None
            ),

            "mean_residual_lag1_autocorrelation": (
                float(np.mean(residual_lags))
                if residual_lags else None
            )
        }
    }

    # --------------------------------------------------------
    # EXPORT CELL RESULTS
    # --------------------------------------------------------

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
        "raw_lag1_autocorrelation",
        "residual_lag1_autocorrelation",
        "tfpw_kendall_tau",
        "tfpw_p_value",
        "tfpw_q_value",
        "tfpw_observations",
        "significant_tfpw_fdr",
        "test_comparison",
        "trend_class",
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

    # --------------------------------------------------------
    # EXPORT REGIONAL SERIES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXPORT SUMMARY
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_DIR / "trend_summary.json"
    )

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

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("IONA | TREND ANALYSIS RESULTS")
    print("=" * 65)

    print("\nREGIONAL TREND")

    print(
        f"Sen's slope: "
        f"{regional_result['sen_slope']:.3f} "
        f"mm/year per year"
    )

    print(
        f"95% CI: "
        f"[{regional_result['sen_ci_lower']:.3f}, "
        f"{regional_result['sen_ci_upper']:.3f}]"
    )

    print(
        f"Conventional MK p-value: "
        f"{regional_result['p_value']:.6f}"
    )

    print(
        f"Residual lag-1 autocorrelation: "
        f"{regional_result['residual_lag1_autocorrelation']:.4f}"
    )

    print(
        f"TFPW MK p-value: "
        f"{regional_result['tfpw_p_value']:.6f}"
    )

    print("\nSPATIAL RESULTS")

    print(f"Cells analyzed: {len(results)}")

    print(
        f"Conventional MK significant after FDR: "
        f"{len(significant_raw)}"
    )

    print(
        f"TFPW significant after FDR: "
        f"{len(significant_tfpw)}"
    )

    print(
        f"Significant in both tests: "
        f"{len(significant_both)}"
    )

    print(
        f"Only conventional: "
        f"{len(only_conventional)}"
    )

    print(
        f"Only TFPW: "
        f"{len(only_tfpw)}"
    )

    print(
        f"Robust increasing cells: "
        f"{len(increasing)}"
    )

    print(
        f"Robust decreasing cells: "
        f"{len(decreasing)}"
    )

    print("\nAUTOCORRELATION DIAGNOSTICS")

    print(
        f"Mean raw lag-1: "
        f"{summary['spatial_results']['mean_raw_lag1_autocorrelation']}"
    )

    print(
        f"Mean residual lag-1: "
        f"{summary['spatial_results']['mean_residual_lag1_autocorrelation']}"
    )

    print("\nOUTPUT FILES")

    print(csv_file)
    print(regional_file)
    print(summary_file)

    print("\nAnalysis completed.")

    print(
        "\nSCIENTIFIC NOTE: TFPW is being used as a "
        "sensitivity analysis, not as an unquestionable "
        "replacement for the conventional test. "
        "Interpret both results alongside the short "
        "24-year time series and spatial dependence."
    )


if __name__ == "__main__":
    main()
