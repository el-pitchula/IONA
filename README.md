# IONA
### Earth Intelligence & Pattern Discovery

**NASA Space Apps Challenge 2026**

> Revealing what the averages hide.

## Overview

IONA is a scientific exploration platform designed to investigate spatial and temporal patterns in Earth's systems using NASA Earth Science data.

The project explores how regional averages can conceal opposing local trends, potentially hiding important environmental changes.

### Initial Research Question

**Does the average precipitation in the Brazilian Cerrado hide opposing regional trends?**

## The Problem

Regional averages are useful, but they can hide important spatial differences.

For example, an overall precipitation trend may appear stable even when some areas are becoming wetter while others are becoming drier.

IONA aims to make these differences visible, measurable, and scientifically interpretable.

## Project Goals

- Explore NASA Earth Science datasets.
- Visualize temporal and spatial changes.
- Estimate trends and evaluate statistical significance.
- Compare local patterns with regional averages.
- Identify anomalies and group regions with similar behavior.
- Generate evidence-based explanations of the results.

## Main Modules

| Module | Purpose |
|---|---|
| **Earth Explorer** | Explore datasets, variables, regions, and time periods. |
| **Trend Laboratory** | Calculate trends, magnitudes, and statistical significance. |
| **Pattern Detective** | Reveal spatial differences hidden by regional averages. |
| **EARTH-MIND** | Apply machine learning to identify anomalies and behavioral patterns. |
| **Evidence Report** | Present findings, methods, data sources, and limitations. |

## Data Sources

Candidate NASA data sources include:

- [NASA GPM](https://gpm.nasa.gov/data) — precipitation observations.
- [NASA GES DISC](https://disc.gsfc.nasa.gov/) — Earth science data access.
- [NASA Earthdata Search](https://search.earthdata.nasa.gov/) — dataset discovery.
- [NASA POWER](https://power.larc.nasa.gov/) — access to environmental data products.
- [MERRA-2](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) — atmospheric reanalysis.

**Note:** The final dataset, product version, temporal coverage, spatial resolution, and units will be documented after data validation.

## Methodology

The planned scientific workflow includes:

1. Data acquisition and quality assessment.
2. Data cleaning and preprocessing.
3. Temporal trend analysis.
4. Spatial comparison and visualization.
5. Statistical significance assessment.
6. Machine learning for anomaly detection and regional clustering.
7. Scientific interpretation and evidence reporting.

Potential statistical methods include:

- Linear Regression
- Mann–Kendall Test
- Sen's Slope
- False Discovery Rate (FDR) correction

## Technology Stack

| Area | Technologies |
|---|---|
| Data Analysis | Python, NumPy, Pandas, SciPy |
| Geospatial Analysis | xarray, GeoPandas |
| Machine Learning | Scikit-learn |
| Frontend | React, TypeScript |
| Visualization | Plotly, Interactive Maps |
| Version Control | Git, GitHub |

## Repository Structure

```text
IONA/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── analysis/
│   ├── download_data.py
│   ├── preprocessing.py
│   ├── trend_analysis.py
│   ├── spatial_analysis.py
│   ├── earth_mind.py
│   └── validation.py
├── public/
│   └── data/
├── src/
│   ├── components/
│   ├── pages/
│   └── visualizations/
├── docs/
│   ├── methodology.md
│   └── data_sources.md
└── README.md
