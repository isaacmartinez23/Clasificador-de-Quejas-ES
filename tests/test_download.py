import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from src.data.download import save, validate

EXPECTED = {"train": 3, "validation": 1, "test": 1}


def make_split(n: int, labels: list[int] | None = None) -> Dataset:
    labels = labels or [i % 5 for i in range(n)]
    return Dataset.from_dict(
        {
            "id": [f"es_{i:07d}" for i in range(n)],
            "text": [f"Título {i}\n\nCuerpo de la reseña {i}" for i in range(n)],
            "label": labels,
            "label_text": [str(label) for label in labels],
        }
    )


def make_dataset() -> DatasetDict:
    return DatasetDict({split: make_split(n) for split, n in EXPECTED.items()})


def test_validate_accepts_expected_schema():
    validate(make_dataset(), EXPECTED)


def test_validate_rejects_missing_split():
    ds = make_dataset()
    del ds["test"]
    with pytest.raises(ValueError, match="Faltan splits"):
        validate(ds, EXPECTED)


def test_validate_rejects_wrong_columns():
    ds = make_dataset()
    ds["train"] = ds["train"].remove_columns("label_text")
    with pytest.raises(ValueError, match="columnas"):
        validate(ds, EXPECTED)


def test_validate_rejects_wrong_row_count():
    ds = make_dataset()
    ds["train"] = make_split(2)
    with pytest.raises(ValueError, match="filas"):
        validate(ds, EXPECTED)


def test_validate_rejects_label_out_of_range():
    ds = make_dataset()
    ds["train"] = make_split(3, labels=[0, 1, 5])
    with pytest.raises(ValueError, match="etiquetas"):
        validate(ds, EXPECTED)


def test_save_writes_one_parquet_per_split(tmp_path):
    ds = make_dataset()
    paths = save(ds, tmp_path)
    assert sorted(p.name for p in paths) == ["test.parquet", "train.parquet", "validation.parquet"]
    train = pd.read_parquet(tmp_path / "train.parquet")
    assert list(train.columns) == ["id", "text", "label", "label_text"]
    assert train["text"].iloc[0] == "Título 0\n\nCuerpo de la reseña 0"
