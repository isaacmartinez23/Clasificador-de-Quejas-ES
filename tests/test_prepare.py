import pandas as pd

from src.data.prepare import COLUMNS, build_complaints, join_title_body, split_title_body


def test_split_title_body():
    assert split_title_body("Roto\n\nLlegó roto") == ("Roto", "Llegó roto")
    assert split_title_body("Sin cuerpo") == ("Sin cuerpo", "")
    assert split_title_body("Título\n\nPárrafo 1\n\nPárrafo 2") == (
        "Título",
        "Párrafo 1\n\nPárrafo 2",
    )


def test_join_title_body_adds_period_only_when_needed():
    assert join_title_body("Roto", "Llegó roto") == "Roto. Llegó roto"
    assert join_title_body("¡Roto!", "Llegó roto") == "¡Roto! Llegó roto"


def test_join_title_body_skips_repeated_title():
    assert join_title_body("No funciona", "NO FUNCIONA") == "NO FUNCIONA"
    assert join_title_body("No funciona", "No funciona desde el primer día") == (
        "No funciona desde el primer día"
    )


def test_join_title_body_with_missing_part():
    assert join_title_body("", "Llegó roto") == "Llegó roto"
    assert join_title_body("Roto", "") == "Roto"


def make_raw(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "text", "label"]).assign(label_text="x")


def test_build_complaints_filters_and_reports_funnel():
    raw = make_raw(
        [
            ("es_1", "Roto\n\nEl producto llegó roto por completo", 0),
            ("es_2", "Flojo\n\nLa calidad deja bastante que desear", 1),
            ("es_3", "Bien\n\nEl producto funciona de maravilla siempre", 4),  # 5 estrellas
            ("es_4", "Malo\n\nMuy malo", 0),  # menos de 5 palabras
            ("es_5", "ROTO\n\nEl producto llegó roto por completo", 0),  # duplicado de es_1
        ]
    )
    df, funnel = build_complaints(raw)

    assert list(df.columns) == COLUMNS
    assert df["review_id"].tolist() == ["es_1", "es_2"]
    assert df["estrellas"].tolist() == [1, 2]
    assert df["texto"].tolist() == [
        "Roto. El producto llegó roto por completo",
        "Flojo. La calidad deja bastante que desear",
    ]
    assert df["n_tokens"].tolist() == [7, 7]
    assert list(funnel.values()) == [5, 4, 3, 2]
