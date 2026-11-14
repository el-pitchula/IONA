from pathlib import Path
import calendar
import csv
import re

import h5py
import numpy as np


# ==========================================================
# IONA | PREPROCESSAMENTO INICIAL GPM IMERG
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "data" / "raw" / "gpm_imerg"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "gpm_imerg_pilot_summary.csv"

# Bounding box: west, south, east, north
WEST = -48.5
SOUTH = -16.5
EAST = -46.0
NORTH = -14.0


def extract_month(file_path):

    match = re.search(r"3IMERG\.(\d{4})(\d{2})01", file_path.name)

    if not match:
        raise ValueError(f"Não foi possível identificar a data: {file_path.name}")

    year = int(match.group(1))
    month = int(match.group(2))

    return year, month


def process_file(file_path):

    year, month = extract_month(file_path)
    days_in_month = calendar.monthrange(year, month)[1]
    hours_in_month = days_in_month * 24

    with h5py.File(file_path, "r") as file:

        grid = file["Grid"]

        longitude = grid["lon"][:]
        latitude = grid["lat"][:]

        # Confirmar ordenação das coordenadas
        if not np.all(np.diff(longitude) > 0):
            raise ValueError("Longitude não está em ordem crescente.")

        if not np.all(np.diff(latitude) > 0):
            raise ValueError("Latitude não está em ordem crescente.")

        # Selecionar índices dentro do recorte geográfico
        lon_indices = np.where(
            (longitude >= WEST) & (longitude <= EAST)
        )[0]

        lat_indices = np.where(
            (latitude >= SOUTH) & (latitude <= NORTH)
        )[0]

        if len(lon_indices) == 0 or len(lat_indices) == 0:
            raise ValueError("Nenhuma célula encontrada no recorte.")

        lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
        lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1

        lon_subset = longitude[lon_start:lon_end]
        lat_subset = latitude[lat_start:lat_end]

        # A ordem dos dados é: tempo, longitude, latitude
        dataset = grid["precipitation"]

        precipitation_rate = dataset[
            0,
            lon_start:lon_end,
            lat_start:lat_end
        ].astype(np.float64)

        fill_value = dataset.attrs["_FillValue"]

        # Remover valores ausentes e não finitos
        valid = (
            np.isfinite(precipitation_rate)
            & ~np.isclose(precipitation_rate, fill_value)
        )

        valid_values = precipitation_rate[valid]

        if valid_values.size == 0:
            raise ValueError("Nenhum dado válido encontrado.")

        # Peso aproximado da área de cada célula:
        # em uma grade latitude-longitude, a área varia com cos(latitude).
        weights = np.cos(np.deg2rad(lat_subset))[np.newaxis, :]

        valid_weights = np.broadcast_to(
            weights, precipitation_rate.shape
        )[valid]

        regional_mean_rate = np.average(
            valid_values,
            weights=valid_weights
        )

        # Taxa média mensal (mm/hr) -> acumulação equivalente (mm)
        monthly_accumulation = regional_mean_rate * hours_in_month

        total_cells = precipitation_rate.size
        valid_cells = valid.sum()
        missing_cells = total_cells - valid_cells

        result = {
            "year": year,
            "month": month,
            "product": "GPM_3IMERGM",
            "version": "V07B",
            "variable": "precipitation",
            "source_unit": "mm/hr",
            "monthly_rate_mm_hr": regional_mean_rate,
            "monthly_accumulation_mm": monthly_accumulation,
            "days_in_month": days_in_month,
            "hours_in_month": hours_in_month,
            "valid_cells": int(valid_cells),
            "missing_cells": int(missing_cells),
            "coverage_percent": 100 * valid_cells / total_cells,
            "minimum_rate_mm_hr": float(np.min(valid_values)),
            "maximum_rate_mm_hr": float(np.max(valid_values)),
            "grid_resolution_degrees": 0.1,
            "west": WEST,
            "south": SOUTH,
            "east": EAST,
            "north": NORTH
        }

        return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.HDF5"))

    if not files:
        print("Nenhum arquivo HDF5 encontrado.")
        return

    print("=" * 60)
    print("IONA | PREPROCESSAMENTO GPM IMERG")
    print("=" * 60)

    results = []

    for file_path in files:

        print(f"\nProcessando: {file_path.name}")

        result = process_file(file_path)
        results.append(result)

        print(f"Período: {result['month']:02d}/{result['year']}")
        print(f"Taxa média: {result['monthly_rate_mm_hr']:.5f} mm/hr")
        print(
            "Acumulação mensal equivalente: "
            f"{result['monthly_accumulation_mm']:.2f} mm"
        )
        print(f"Células válidas: {result['valid_cells']}")
        print(f"Cobertura: {result['coverage_percent']:.2f}%")

    # Exportar resumo para CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"Arquivo gerado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()