
from pathlib import Path
import calendar
import json
import re
import csv

import h5py
import numpy as np


# ==========================================
# CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "gpm_imerg"
OUTPUT_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"

WEST, SOUTH, EAST, NORTH = (
    -48.5, -16.5, -46.0, -14.0
)

START_YEAR = 2001
END_YEAR = 2024

# ==========================================
# EXTRAÇÃO DO PERÍODO
# ==========================================

def extract_date(path):

    match = re.search(
        r"3IMERG\.(\d{4})(\d{2})01",
        path.name
    )

    if not match:
        raise ValueError(
            f"Nome de arquivo inválido: {path.name}"
        )

    year = int(match.group(1))
    month = int(match.group(2))

    return year, month


# ==========================================
# LEITURA DE UM ARQUIVO
# ==========================================

def read_monthly_file(path):

    year, month = extract_date(path)

    hours_in_month = (
        calendar.monthrange(year, month)[1] * 24
    )

    with h5py.File(path, "r") as file:

        grid = file["Grid"]

        longitude = grid["lon"][:]
        latitude = grid["lat"][:]

        lon_indices = np.where(
            (longitude >= WEST) &
            (longitude <= EAST)
        )[0]

        lat_indices = np.where(
            (latitude >= SOUTH) &
            (latitude <= NORTH)
        )[0]

        if len(lon_indices) == 0 or len(lat_indices) == 0:
            raise ValueError(
                f"Área-piloto não encontrada em {path.name}"
            )

        precipitation = grid["precipitation"]

        data = precipitation[
            0,
            lon_indices[0]:lon_indices[-1] + 1,
            lat_indices[0]:lat_indices[-1] + 1
        ].astype(np.float64)

        fill_value = precipitation.attrs["_FillValue"]

        valid = (
            np.isfinite(data) &
            ~np.isclose(data, fill_value)
        )

        # Converter taxa média (mm/hr)
        # em acumulação mensal (mm/month).
        monthly_precipitation = (
            data * hours_in_month
        )

        monthly_precipitation[~valid] = np.nan

        return (
            year,
            month,
            longitude[lon_indices],
            latitude[lat_indices],
            monthly_precipitation
        )


# ==========================================
# CONSTRUÇÃO DA MATRIZ
# ==========================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("IONA | SPATIOTEMPORAL DATASET BUILDER")
    print("=" * 60)

    files = sorted(INPUT_DIR.glob("*.HDF5"))

    expected_months = (
        (END_YEAR - START_YEAR + 1) * 12
    )

    if len(files) != expected_months:
        raise RuntimeError(
            f"Esperados {expected_months} arquivos, "
            f"encontrados {len(files)}."
        )

    records = {}

    longitude = None
    latitude = None

    # --------------------------------------
    # Processar arquivos
    # --------------------------------------

    print(f"\nArquivos encontrados: {len(files)}")
    print("\nProcessando série histórica...\n")

    for index, path in enumerate(files, start=1):

        year, month, lon, lat, data = (
            read_monthly_file(path)
        )

        key = (year, month)

        if key in records:
            raise RuntimeError(
                f"Mês duplicado: {year}-{month:02d}"
            )

        if longitude is None:

            longitude = lon
            latitude = lat

        else:

            if not np.array_equal(longitude, lon):
                raise RuntimeError(
                    "Grade de longitude inconsistente."
                )

            if not np.array_equal(latitude, lat):
                raise RuntimeError(
                    "Grade de latitude inconsistente."
                )

        records[key] = data

        if index % 12 == 0 or index == len(files):

            print(
                f"[{index:03d}/{len(files)}] "
                f"{year}-{month:02d} concluído."
            )

    # --------------------------------------
    # Verificar sequência temporal
    # --------------------------------------

    time_index = [
        (year, month)
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    ]

    missing = [
        key for key in time_index
        if key not in records
    ]

    if missing:
        raise RuntimeError(
            f"Meses ausentes: {missing}"
        )

    # --------------------------------------
    # Construir cubo espaço-temporal
    # --------------------------------------

    cube = np.stack(
        [records[key] for key in time_index],
        axis=0
    )

    expected_shape = (
        expected_months,
        len(longitude),
        len(latitude)
    )

    if cube.shape != expected_shape:
        raise RuntimeError(
            f"Dimensão inesperada: {cube.shape}"
        )

    # --------------------------------------
    # Estatísticas básicas
    # --------------------------------------

    valid_values = cube[np.isfinite(cube)]

    if valid_values.size == 0:
        raise RuntimeError(
            "O conjunto não possui valores válidos."
        )

    total_values = cube.size
    valid_count = valid_values.size

    missing_percentage = (
        1 - valid_count / total_values
    ) * 100

    # --------------------------------------
    # Salvar arquivos
    # --------------------------------------

    cube_path = OUTPUT_DIR / "precipitation_cube.npy"

    coordinates_path = (
        OUTPUT_DIR / "coordinates.npz"
    )

    time_path = OUTPUT_DIR / "time_index.csv"

    metadata_path = (
        METADATA_DIR / "spatiotemporal_dataset.json"
    )

    np.save(cube_path, cube)

    np.savez_compressed(
        coordinates_path,
        longitude=longitude,
        latitude=latitude
    )

    with open(
        time_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.writer(output)

        writer.writerow([
            "year",
            "month",
            "date"
        ])

        for year, month in time_index:

            writer.writerow([
                year,
                month,
                f"{year}-{month:02d}-01"
            ])

    metadata = {
        "project": "IONA",
        "dataset": "GPM IMERG Monthly Final Run",
        "product": "GPM_3IMERGM",
        "version": "V07B",
        "doi": "10.5067/GPM/IMERG/3B-MONTH/07",
        "period": {
            "start": f"{START_YEAR}-01",
            "end": f"{END_YEAR}-12",
            "months": expected_months
        },
        "region": {
            "west": WEST,
            "south": SOUTH,
            "east": EAST,
            "north": NORTH,
            "description": (
                "Pilot rectangular region around "
                "Distrito Federal and surroundings"
            )
        },
        "variable": "monthly_precipitation",
        "units": "mm/month",
        "dimensions": {
            "time": int(cube.shape[0]),
            "longitude": int(cube.shape[1]),
            "latitude": int(cube.shape[2])
        },
        "valid_values": int(valid_count),
        "total_values": int(total_values),
        "missing_percentage": float(missing_percentage),
        "statistics": {
            "minimum": float(valid_values.min()),
            "maximum": float(valid_values.max()),
            "mean": float(valid_values.mean()),
            "median": float(np.median(valid_values)),
            "standard_deviation": float(
                valid_values.std()
            )
        },
        "files": {
            "cube": str(cube_path.relative_to(ROOT)),
            "coordinates": str(
                coordinates_path.relative_to(ROOT)
            ),
            "time_index": str(time_path.relative_to(ROOT))
        }
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as output:

        json.dump(
            metadata,
            output,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------
    # Resumo
    # --------------------------------------

    print("\n" + "=" * 60)
    print("DATASET ESPAÇO-TEMPORAL CONSTRUÍDO")
    print("=" * 60)

    print(f"\nDimensões: {cube.shape}")
    print(f"Meses: {cube.shape[0]}")
    print(f"Células espaciais: {cube.shape[1] * cube.shape[2]}")

    print(f"\nMínimo: {valid_values.min():.2f} mm/mês")
    print(f"Máximo: {valid_values.max():.2f} mm/mês")
    print(f"Média: {valid_values.mean():.2f} mm/mês")
    print(f"Mediana: {np.median(valid_values):.2f} mm/mês")

    print(
        f"Valores ausentes: {missing_percentage:.4f}%"
    )

    print("\nArquivos gerados:")

    print(f" - {cube_path}")
    print(f" - {coordinates_path}")
    print(f" - {time_path}")
    print(f" - {metadata_path}")

    print("\nProcessamento concluído.")


if __name__ == "__main__":
    main()
