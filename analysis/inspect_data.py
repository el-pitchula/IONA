from pathlib import Path
import h5py

# Diretório dos dados GPM IMERG
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "gpm_imerg"

def inspect_hdf5(file_path):

    print("\n" + "=" * 70)
    print(f"ARQUIVO: {file_path.name}")
    print("=" * 70)

    with h5py.File(file_path, "r") as file:

        def inspect_item(name, obj):

            print(f"\nCaminho: {name}")

            if isinstance(obj, h5py.Dataset):

                print("  Tipo: Dataset")
                print(f"  Dimensões: {obj.shape}")
                print(f"  Tipo de dado: {obj.dtype}")

                if obj.attrs:
                    print("  Atributos:")

                    for key, value in obj.attrs.items():
                        print(f"    {key}: {value}")

            elif isinstance(obj, h5py.Group):

                print("  Tipo: Grupo")

                if obj.attrs:
                    print("  Atributos:")

                    for key, value in obj.attrs.items():
                        print(f"    {key}: {value}")

        file.visititems(inspect_item)

        print("\n" + "-" * 70)
        print("ATRIBUTOS GLOBAIS")
        print("-" * 70)

        for key, value in file.attrs.items():
            print(f"{key}: {value}")


def main():

    files = sorted(DATA_DIR.glob("*.HDF5"))

    if not files:
        print("Nenhum arquivo HDF5 encontrado.")
        return

    print(f"Arquivos encontrados: {len(files)}")

    for file_path in files:
        inspect_hdf5(file_path)


if __name__ == "__main__":
    main()