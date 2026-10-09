"""Descarga el split en español de Amazon Reviews Multi (copia de MTEB) a data/raw/.

Uso:
    uv run python -m src.data.download
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset

REPO_ID = "mteb/amazon_reviews_multi"
LANG = "es"
# Commit fijo del repo para que la descarga sea reproducible.
REVISION = "c379a6705fec24a2493fa68e011692605f44e119"
SPLITS = ("train", "validation", "test")
EXPECTED_COLUMNS = ("id", "text", "label", "label_text")
EXPECTED_ROWS = {"train": 200_000, "validation": 5_000, "test": 5_000}
RAW_DIR = Path("data/raw")


def load_raw() -> DatasetDict:
    # El repo define sus configs con un script de carga, que `datasets` ya no ejecuta.
    # Los JSONL traen las mismas columnas, así que se leen directamente.
    data_files = {
        split: f"hf://datasets/{REPO_ID}@{REVISION}/{LANG}/{split}.jsonl" for split in SPLITS
    }
    return load_dataset("json", data_files=data_files)


def validate(ds: DatasetDict, expected_rows: dict[str, int] = EXPECTED_ROWS) -> None:
    """Falla si el dataset no tiene los splits, columnas, filas o etiquetas esperados."""
    missing = set(expected_rows) - set(ds)
    if missing:
        raise ValueError(f"Faltan splits: {sorted(missing)}")
    for split, n_expected in expected_rows.items():
        columns = tuple(ds[split].column_names)
        if columns != EXPECTED_COLUMNS:
            raise ValueError(f"{split}: columnas {columns}, se esperaban {EXPECTED_COLUMNS}")
        if ds[split].num_rows != n_expected:
            raise ValueError(f"{split}: {ds[split].num_rows} filas, se esperaban {n_expected}")
        labels = set(ds[split].unique("label"))
        if not labels <= set(range(5)):
            raise ValueError(f"{split}: etiquetas fuera de 0-4: {sorted(labels)}")


def save(ds: DatasetDict, out_dir: Path = RAW_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ds:
        path = out_dir / f"{split}.parquet"
        ds[split].to_parquet(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    ds = load_raw()
    validate(ds)
    for path in save(ds, args.out_dir):
        print(f"{path}  ({ds[path.stem].num_rows:,} filas)")

    stars = ds["train"].to_pandas()["label"].add(1).value_counts().sort_index()
    print("\nReseñas por estrellas (train):")
    print(stars.to_string())
    print(f"\nQuejas (1-2 estrellas) en train: {int(stars.loc[[1, 2]].sum()):,}")


if __name__ == "__main__":
    main()
