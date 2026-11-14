
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

# Período inicial de teste: três meses
START_DATE = "2024-01-01"
END_DATE = "2024-03-31"

# Área piloto: Distrito Federal e entorno
# Ordem: longitude oeste, latitude sul,
#        longitude leste, latitude norte.
BOUNDING_BOX = (-48.5, -16.5, -46.0, -14.0)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print("          IONA | DATA ACQUISITION")
    print("======================================\n")

    # 1. Autenticação NASA Earthdata
    print("[1/3] Autenticando na NASA Earthdata...")

    auth = earthaccess.login()

    if not auth:
        raise RuntimeError(
            "Não foi possível autenticar. "
            "Verifique sua conta NASA Earthdata."
        )

    print("Autenticação concluída.")

    # 2. Busca dos arquivos
    print("\n[2/3] Pesquisando dados GPM IMERG...")

    granules = earthaccess.search_data(
        short_name=SHORT_NAME,
        temporal=(START_DATE, END_DATE),
        bounding_box=BOUNDING_BOX,
        count=3
    )

    if not granules:
        raise RuntimeError(
            "Nenhum arquivo encontrado. "
            "Verifique o produto, período e região."
        )

    print(f"Arquivos encontrados: {len(granules)}")

    for index, granule in enumerate(granules, start=1):
        print(f"  {index}. {granule}")

    # 3. Download
    print("\n[3/3] Baixando arquivos...")

    downloaded_files = earthaccess.download(
        granules,
        local_path=str(OUTPUT_DIR)
    )

    if not downloaded_files:
        raise RuntimeError("Nenhum arquivo foi baixado.")

    # 4. Registrar metadados da aquisição
    metadata = {
        "project": "IONA",
        "dataset_short_name": SHORT_NAME,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "bounding_box": BOUNDING_BOX,
        "files_downloaded": [
            str(file) for file in downloaded_files
        ],
        "download_date_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    metadata_file = METADATA_DIR / "gpm_imerg_download.json"

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")
    print("          DOWNLOAD CONCLUÍDO")
    print("======================================")

    print(f"\nArquivos salvos em:\n{OUTPUT_DIR}")
    print(f"\nMetadados registrados em:\n{metadata_file}")

    print("\nArquivos recebidos:")

    for file in downloaded_files:
        print(f" - {file}")

    print("\nPróxima etapa: inspecionar os arquivos NetCDF.")


if __name__ == "__main__":
    main()
