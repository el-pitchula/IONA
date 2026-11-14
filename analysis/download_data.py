
from pathlib import Path
import json
from datetime import datetime, timezone

import earthaccess


# ==========================================
# CONFIGURAÇÕES DO PROJETO IONA
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gpm_imerg"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SHORT_NAME = "GPM_3IMERGM"

START_YEAR = 2001
END_YEAR = 2024

BOUNDING_BOX = (-48.5, -16.5, -46.0, -14.0)

METADATA_FILE = METADATA_DIR / "gpm_imerg_historical_acquisition.json"


# ==========================================
# REGISTRO DE METADADOS
# ==========================================

def save_metadata(metadata):

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata["last_updated_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================
# AQUISIÇÃO DE DADOS
# ==========================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("IONA | HISTORICAL DATA ACQUISITION")
    print("=" * 60)

    # --------------------------------------
    # 1. Autenticação
    # --------------------------------------

    print("\n[1/3] Autenticando na NASA Earthdata...")

    auth = earthaccess.login()

    if not auth:
        raise RuntimeError(
            "Falha na autenticação NASA Earthdata."
        )

    print("Autenticação concluída.")

    # --------------------------------------
    # 2. Inicialização dos metadados
    # --------------------------------------

    metadata = {
        "project": "IONA",
        "dataset": SHORT_NAME,
        "product_version": "V07B",
        "doi": "10.5067/GPM/IMERG/3B-MONTH/07",
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "bounding_box": BOUNDING_BOX,
        "temporal_resolution": "monthly",
        "acquisition_started_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "yearly_batches": [],
        "failures": []
    }

    # --------------------------------------
    # 3. Busca e download por ano
    # --------------------------------------

    print("\n[2/3] Iniciando aquisição histórica...")

    for year in range(START_YEAR, END_YEAR + 1):

        print("\n" + "-" * 60)
        print(f"ANO: {year}")
        print("-" * 60)

        batch_record = {
            "year": year,
            "status": "started",
            "granules_found": 0,
            "files_returned": [],
            "completed_utc": None
        }

        try:

            # Busca os 12 arquivos mensais do ano
            granules = earthaccess.search_data(
                short_name=SHORT_NAME,
                temporal=(
                    f"{year}-01-01",
                    f"{year}-12-31"
                ),
                bounding_box=BOUNDING_BOX,
                count=12
            )

            batch_record["granules_found"] = len(granules)

            print(f"Granulações encontradas: {len(granules)}")

            if not granules:

                batch_record["status"] = "no_data"

                print("Nenhum arquivo encontrado.")

            else:

                # Download incremental.
                # Arquivos existentes não são sobrescritos.
                downloaded_files = earthaccess.download(
                    granules,
                    local_path=str(OUTPUT_DIR),
                    threads=4,
                    force=False
                )

                batch_record["files_returned"] = [
                    str(file) for file in downloaded_files
                ]

                batch_record["status"] = "completed"

                print(
                    f"Arquivos retornados pelo download: "
                    f"{len(downloaded_files)}"
                )

            batch_record["completed_utc"] = datetime.now(
                timezone.utc
            ).isoformat()

        except Exception as error:

            batch_record["status"] = "failed"
            batch_record["error"] = str(error)

            metadata["failures"].append({
                "year": year,
                "error": str(error)
            })

            print(f"ERRO no ano {year}: {error}")

        metadata["yearly_batches"].append(batch_record)

        # Salva o progresso imediatamente.
        save_metadata(metadata)

    # --------------------------------------
    # 4. Verificação final
    # --------------------------------------

    print("\n" + "=" * 60)
    print("IONA | AQUISIÇÃO FINALIZADA")
    print("=" * 60)

    local_files = sorted(OUTPUT_DIR.glob("*.HDF5"))

    completed_years = sum(
        batch["status"] == "completed"
        for batch in metadata["yearly_batches"]
    )

    failed_years = sum(
        batch["status"] == "failed"
        for batch in metadata["yearly_batches"]
    )

    print(f"\nArquivos HDF5 presentes: {len(local_files)}")
    print(f"Lotes concluídos: {completed_years}")
    print(f"Lotes com falhas: {failed_years}")

    print(f"\nDados armazenados em:\n{OUTPUT_DIR}")
    print(f"\nMetadados armazenados em:\n{METADATA_FILE}")

    if metadata["failures"]:

        print("\nATENÇÃO: existem anos com falhas.")

        for failure in metadata["failures"]:
            print(
                f" - {failure['year']}: "
                f"{failure['error']}"
            )

    else:

        print("\nNenhuma falha registrada.")

    print("\nPróxima etapa: validar a integridade da série histórica.")


if __name__ == "__main__":
    main()
