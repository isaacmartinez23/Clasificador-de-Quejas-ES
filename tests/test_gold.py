import pandas as pd
import pytest

from src.labeling.gold import (
    LABEL_COLUMNS,
    agreement,
    build_sample,
    label_loop,
    parse_answer,
    read_labels,
    stats_report,
    stratified_sample,
    write_labels,
)
from src.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


@pytest.fixture
def complaints() -> pd.DataFrame:
    n = 600
    return pd.DataFrame(
        {
            "review_id": [f"es_{i:04d}" for i in range(n)],
            "estrellas": [1 if i % 3 else 2 for i in range(n)],
            "n_tokens": [5 + i % 60 for i in range(n)],
            "titulo": "Título",
            "cuerpo": [f"Cuerpo de la queja {i}" for i in range(n)],
            "texto": [
                f"Queja {i}: pedí el reembolso" if i % 10 == 0 else f"Queja {i}" for i in range(n)
            ],
        }
    )


def test_taxonomy_is_consistent(taxonomy):
    assert taxonomy.tema_ids[-1] == "otro"
    assert taxonomy.urgencia_ids == ["alta", "media", "baja"]
    # El etiquetado usa un dígito por tema y la inicial de cada urgencia.
    assert len(taxonomy.tema_ids) <= 9
    assert len({urgencia[0] for urgencia in taxonomy.urgencia_ids}) == 3


def test_stratified_sample_is_exact_proportional_and_deterministic(complaints):
    sample = stratified_sample(complaints, 90, seed=1)
    assert len(sample) == 90
    assert sample["review_id"].is_unique
    # El reparto es exacto por estrato (estrellas × longitud), no por estrellas sola.
    assert abs(sample["estrellas"].eq(1).sum() - 60) <= 2
    assert sample.index.equals(stratified_sample(complaints, 90, seed=1).index)


def test_build_sample_parts_are_disjoint_and_targeted_matches_pattern(complaints):
    sample = build_sample(complaints, n_random=100, n_targeted=20, seed=1)
    assert sample["review_id"].is_unique
    assert sample["origen"].value_counts().to_dict() == {"aleatorio": 100, "dirigido": 20}
    targeted = sample.loc[sample["origen"] == "dirigido", "review_id"]
    texts = complaints.set_index("review_id").loc[targeted, "texto"]
    assert texts.str.contains("reembolso").all()
    assert list(sample.columns) == ["review_id", "origen"]


def test_parse_answer(taxonomy):
    assert parse_answer("1a", taxonomy) == ("entrega", "alta")
    assert parse_answer(" 2 M ", taxonomy) == (taxonomy.tema_ids[1], "media")
    assert parse_answer("7b", taxonomy) == ("otro", "baja")
    for invalid in ["", "3", "m", "0a", "8a", "3x", "33m", "am"]:
        assert parse_answer(invalid, taxonomy) is None


def test_labels_roundtrip_without_text(tmp_path):
    path = tmp_path / "gold.csv"
    labels = {"es_0001": ("entrega", "alta"), "es_0002": ("otro", "baja")}
    write_labels(labels, path)
    assert read_labels(path) == labels
    assert list(pd.read_csv(path).columns) == LABEL_COLUMNS
    assert read_labels(tmp_path / "no_existe.csv") == {}


def run_loop(answers, review_ids, complaints, taxonomy, labels=None):
    saved = []
    answers = iter(answers)
    result = label_loop(
        review_ids,
        complaints.set_index("review_id"),
        labels or {},
        taxonomy,
        save=lambda current: saved.append(dict(current)),
        input_fn=lambda _: next(answers),
        print_fn=lambda _: None,
    )
    return result, saved


def test_label_loop_saves_after_each_answer_and_ignores_invalid(complaints, taxonomy):
    ids = ["es_0001", "es_0002"]
    labels, saved = run_loop(["zz", "?", "1a", "7b"], ids, complaints, taxonomy)
    assert labels == {"es_0001": ("entrega", "alta"), "es_0002": ("otro", "baja")}
    assert len(saved) == 2


def test_label_loop_resumes_and_quits(complaints, taxonomy):
    ids = ["es_0001", "es_0002", "es_0003"]
    done = {"es_0001": ("entrega", "alta")}
    labels, _ = run_loop(["7b", "q"], ids, complaints, taxonomy, labels=done)
    assert labels == {"es_0001": ("entrega", "alta"), "es_0002": ("otro", "baja")}


def test_label_loop_goes_back_to_fix_previous(complaints, taxonomy):
    ids = ["es_0001", "es_0002"]
    labels, _ = run_loop(["1a", "<", "7b", "1m"], ids, complaints, taxonomy)
    assert labels == {"es_0001": ("otro", "baja"), "es_0002": ("entrega", "media")}


def test_agreement():
    first = {"a": ("entrega", "alta"), "b": ("otro", "baja"), "c": ("calidad", "media")}
    assert agreement(first, dict(first)) == {
        "n": 3,
        "kappa_tema": 1.0,
        "acuerdo_tema": 1.0,
        "kappa_urgencia": 1.0,
        "acuerdo_urgencia": 1.0,
    }
    second = {"a": ("entrega", "baja"), "b": ("otro", "baja"), "z": ("otro", "baja")}
    result = agreement(first, second)
    assert result["n"] == 2
    assert result["acuerdo_tema"] == 1.0
    assert result["acuerdo_urgencia"] == 0.5
    with pytest.raises(ValueError):
        agreement(first, {})


def test_stats_report_flags_small_classes(taxonomy):
    sample = pd.DataFrame(
        {"review_id": ["a", "b", "c"], "origen": ["aleatorio", "aleatorio", "dirigido"]}
    )
    labels = {"a": ("entrega", "alta"), "c": ("postventa", "media")}
    report = stats_report(labels, sample, taxonomy)
    assert "Etiquetadas: 2 de 3" in report
    assert "menos de 20" in report
    assert stats_report({}, sample, taxonomy) == "Etiquetadas: 0 de 3"
