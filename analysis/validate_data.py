
from pathlib import Path
import re
import calendar

import h5py
import numpy as np


# ==========================================
# CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw" / "gpm_imerg"
REPORT_DIR = ROOT / "data" / "metadata"

START_YEAR = 2001
END_YEAR = 2024

WEST, SOUTH, EAST, NORTH = (
    -48.5, -16.5, -46.0, -14.0
)

EXPECTED_SHAPE = (1, 3600, 1800)
EXPECTED_CELLS = 625


# ==========================================
# VALIDAÇÃO INDIVIDUAL
# ==========================================

def validate_file(path):

    result = {
        "file": path.name,
        "status": "valid",
        "errors": [],
        "year": None,
        "month": None,
        "valid_cells": 0
    }

    match = re.search(
        r"3IMERG\.(\d{4})(\d{2})01",
        path.name
    )

    if not match:
        result["status"] = "invalid"
        result["errors"].append(
            "Nome do arquivo fora do padrão esperado."
        )
        return result

    year = int(match.group(1))
    month = int(match.group(2))

    result["year"] = year
    result["month"] = month

    try:

        with h5py.File(path, "r") as file:

            if "Grid" not in file:
                raise ValueError("Grupo Grid ausente.")

            grid = file["Grid"]

            required = [
                "precipitation",
                "lon",
                "lat",
                "time"
            ]

            for name in required:
                if name not in grid:
                    raise ValueError(
                        f"Variável ausente: Grid/{name}"
                    )

            dataset = grid["precipitation"]

            if dataset.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Dimensão inesperada: {dataset.shape}"
                )

            units = dataset.attrs.get("units", b"")

            if isinstance(units, bytes):
                units = units.decode("utf-8")

            if units != "mm/hr":
                raise ValueError(
                    f"Unidade inesperada: {units}"
                )

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
                    "Área-piloto não encontrada na grade."
                )

            data = dataset[
                0,
                lon_indices[0]:lon_indices[-1] + 1,
                lat_indices[0]:lat_indices[-1] + 1
            ].astype(np.float64)

            fill_value = dataset.attrs["_FillValue"]

            valid = (
                np.isfinite(data) &
                ~np.isclose(data, fill_value)
            )

            result["valid_cells"] = int(valid.sum())

            if result["valid_cells"] == 0:
                raise ValueError(
                    "Nenhuma célula válida na área-piloto."
                )

            if data.shape != (
                len(lon_indices),
                len(lat_indices)
            ):
                raise ValueError(
                    "Dimensão espacial inconsistente."
                )

            result["coverage"] = (
                result["valid_cells"] / EXPECTED_CELLS
            ) * 100

            result["minimum_rate"] = float(
                data[valid].min()
            )

            result["maximum_rate"] = float(
                data[valid].max()
            )

    except Exception as error:

        result["status"] = "invalid"
        result["errors"].append(str(error))

    return result


# ==========================================
# VALIDAÇÃO DA SÉRIE TEMPORAL
# ==========================================

def main():

    print("\n" + "=" * 60)
    print("IONA | HISTORICAL DATA VALIDATION")
    print("=" * 60)

    files = sorted(DATA_DIR.glob("*.HDF5"))

    print(f"\nArquivos encontrados: {len(files)}")

    results = [
        validate_file(path)
        for path in files
    ]

    # Identificar meses ausentes e duplicados
    expected_months = {
        (year, month)
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    }

    observed_months = [
        (r["year"], r["month"])
        for r in results
        if r["year"] is not None
    ]

    observed_set = set(observed_months)

    missing = sorted(
        expected_months - observed_set
    )

    duplicates = sorted(
        month for month in observed_set
        if observed_months.count(month) > 1
    )

    invalid = [
        r for r in results
        if r["status"] != "valid"
    ]

    valid = [
        r for r in results
        if r["status"] == "valid"
    ]

    # ======================================
    # RELATÓRIO
    # ======================================

    print("\n" + "-" * 60)
    print("RESULTADO DA VALIDAÇÃO")
    print("-" * 60)

    print(f"Meses esperados: {len(expected_months)}")
    print(f"Meses encontrados: {len(observed_set)}")
    print(f"Arquivos válidos: {len(valid)}")
    print(f"Arquivos inválidos: {len(invalid)}")

    print(f"Meses ausentes: {len(missing)}")
    print(f"Meses duplicados: {len(duplicates)}")

    if missing:
        print("\nMESES AUSENTES:")

        for year, month in missing:
            print(f" - {year}-{month:02d}")

    if duplicates:
        print("\nMESES DUPLICADOS:")

        for year, month in duplicates:
            print(f" - {year}-{month:02d}")

    if invalid:
        print("\nARQUIVOS INVÁLIDOS:")

        for item in invalid:
            print(f"\n - {item['file']}")

            for error in item["errors"]:
                print(f"   Erro: {error}")

    # ======================================
    # ESTATÍSTICAS DE COBERTURA
    # ======================================

    if valid:

        coverage = np.array([
            item["coverage"]
            for item in valid
        ])

        print("\nCOBERTURA ESPACIAL:")

        print(
            f"Média: {coverage.mean():.2f}%"
        )

        print(
            f"Mínima: {coverage.min():.2f}%"
        )

        print(
            f"Máxima: {coverage.max():.2f}%"
        )

    # ======================================
    # SALVAR RESULTADO
    # ======================================

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = (
        REPORT_DIR / "gpm_imerg_validation_report.csv"
    )

    import csv

    fields = [
        "file",
        "year",
        "month",
        "status",
        "valid_cells",
        "coverage",
        "minimum_rate",
        "maximum_rate",
        "errors"
    ]

    with open(
        report_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=fields
        )

        writer.writeheader()

        for item in results:

            row = item.copy()

            row["errors"] = "; ".join(
                item["errors"]
            )

            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })

    print("\nRelatório salvo em:")
    print(report_file)

    # ======================================
    # RESULTADO FINAL
    # ======================================

    success = (
        len(files) == len(expected_months)
        and len(valid) == len(expected_months)
        and not missing
        and not duplicates
        and not invalid
    )

    print("\n" + "=" * 60)

    if success:
        print("VALIDAÇÃO CONCLUÍDA: SÉRIE ÍNTEGRA")
    else:
        print("VALIDAÇÃO INCOMPLETA: VERIFICAR RELATÓRIO")

    print("=" * 60)


if __name__ == "__main__":
    main()
