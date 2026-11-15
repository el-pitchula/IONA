# IONA — Earth Intelligence & Pattern Discovery

**Revealing what the averages hide.**

IONA is a scientific data analysis platform developed for the **NASA Space Apps Challenge 2026**. It investigates spatial and temporal precipitation patterns to reveal regional behaviors that may remain hidden behind aggregated climate averages.

## Research Question

**Does the average precipitation in the Brazilian Cerrado hide opposing regional trends?**

IONA investigates this question through a spatially explicit analysis of satellite-derived precipitation data, combining statistical methods, spatial autocorrelation, and machine learning approaches.

> **Study scope:** The current pilot study covers a rectangular region surrounding the Federal District of Brazil. It does not represent the official geographical boundary of the entire Cerrado biome.

## Scientific Data

The project uses NASA's **Global Precipitation Measurement (GPM) IMERG Monthly Final Run**, Version 07.

| Attribute | Description |
|---|---|
| Product | GPM IMERG Monthly Final Run |
| Product ID | 3IMERGM |
| Version | V07 |
| Study period | 2001–2024 |
| Temporal resolution | Monthly |
| Spatial resolution | 0.1° |
| Spatial grid | 25 × 25 cells |
| Total observations | 180,000 cell-month observations |
| DOI | 10.5067/GPM/IMERG/3B-MON/07 |

**Data source:** [NASA GPM IMERG](https://gpm.nasa.gov/data/imerg)

## Scientific Methodology

IONA combines multiple analytical approaches:

- **Precipitation analysis:** Monthly climatology, annual accumulation, and distribution diagnostics.
- **Temporal trends:** Sen's slope estimator and Mann–Kendall trend testing.
- **Multiple testing correction:** Benjamini–Hochberg False Discovery Rate (FDR).
- **Spatial autocorrelation:** Global Moran's I.
- **Local spatial analysis:** Local Indicators of Spatial Association (LISA).
- **Robustness analysis:** Comparison of Queen and Rook spatial contiguity definitions.
- **Machine learning:** Development of the EARTH-MIND anomaly detection and interpretation module.

Monthly precipitation rates are converted into monthly accumulated precipitation using the number of days in each month.

## Core Modules

| Module | Purpose |
|---|---|
| Earth Explorer | Explore precipitation data and geographical patterns. |
| Trend Laboratory | Investigate temporal trends and statistical evidence. |
| Pattern Detective | Examine spatial clusters and regional contrasts. |
| EARTH-MIND | Detect unusual precipitation behavior and generate evidence-based interpretations. |
| Evidence Report | Present scientific findings, methods, and limitations. |

*Module implementation status may vary during development.*

## Preliminary Scientific Findings

The current analysis has produced the following results:

- 288 monthly observations covering 2001–2024.
- 625 spatial cells in the pilot region.
- A regional Sen's slope estimate of approximately −11.05 mm/year².
- The regional temporal trend is not statistically significant at the 5% level.
- No individual cell retained a statistically significant temporal trend after FDR correction.
- Global Moran's I indicates strong positive spatial autocorrelation in the estimated Sen's slopes.
- Local spatial classifications show sensitivity to the selected neighborhood definition.

These findings are exploratory and must be interpreted within the spatial and temporal scope of the pilot study. Spatial autocorrelation does not establish causality or imply statistically significant temporal trends.

## Spatial Trend Cancellation Index (STCI)

IONA introduces a project-defined descriptive metric:

\[
STCI = 1 -
\frac{\left|\sum_i w_i\beta_i\right|}
{\sum_i w_i|\beta_i|}
\]

where:

- \(\beta_i\) represents the Sen's slope estimated for spatial cell \(i\).
- \(w_i\) represents the cosine-of-latitude weighting.

The index measures the degree to which positive and negative regional slope estimates offset one another.

**STCI is an experimental project-defined metric, not an established climate index.**

## Technology Stack

**Scientific computing**
- Python 3.11
- NumPy
- Pandas
- SciPy
- h5py
- xarray
- PySAL / libpysal
- esda
- Matplotlib

**Application**
- React
- TypeScript
- Vite
