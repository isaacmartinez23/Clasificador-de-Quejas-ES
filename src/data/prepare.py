"""Convierte las reseñas crudas en el corpus de quejas (1-2 estrellas) limpio y deduplicado.

Uso:
    uv run python -m src.data.prepare
"""

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.clean_text import clean_heavy_batch, clean_light

RAW_PATH = Path("data/raw/train.parquet")
OUT_PATH = Path("data/processed/quejas.parquet")
COMPLAINT_LABELS = (0, 1)  # label = estrellas - 1
MIN_TOKENS = 5
COLUMNS = ["review_id", "estrellas", "titulo", "cuerpo", "texto", "n_tokens"]


def split_title_body(text: str) -> tuple[str, str]:
    title, _, body = text.partition("\n\n")
    return title, body


def join_title_body(titulo: str, cuerpo: str) -> str:
    """Une título y cuerpo en un solo texto, sin repetir el título si el cuerpo ya lo contiene."""
    if not titulo or cuerpo.casefold().startswith(titulo.casefold()):
        return cuerpo
    if not cuerpo:
        return titulo
    separator = " " if titulo[-1] in ".!?…:;," else ". "
    return titulo + separator + cuerpo


def build_complaints(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Devuelve las quejas y cuántas filas sobreviven a cada filtro."""
    funnel = {"reseñas": len(raw)}

    df = raw[raw["label"].isin(COMPLAINT_LABELS)].copy()
    funnel["1-2 estrellas"] = len(df)

    parts = [split_title_body(text) for text in df["text"]]
    df["titulo"] = [clean_light(title) for title, _ in parts]
    df["cuerpo"] = [clean_light(body) for _, body in parts]
    df["texto"] = [join_title_body(t, c) for t, c in zip(df["titulo"], df["cuerpo"], strict=True)]
    df["n_tokens"] = df["texto"].str.split().str.len()
    df["estrellas"] = df["label"] + 1
    df = df.rename(columns={"id": "review_id"})

    df = df[df["n_tokens"] >= MIN_TOKENS]
    funnel[f"con {MIN_TOKENS}+ palabras"] = len(df)

    df = df[~df["texto"].str.casefold().duplicated()]
    funnel["sin duplicados"] = len(df)

    return df[COLUMNS].reset_index(drop=True), funnel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    df, funnel = build_complaints(pd.read_parquet(args.raw))
    for step, n in funnel.items():
        print(f"{step:>20}: {n:>7,}")

    print("Lematizando para TF-IDF (tarda unos minutos)...")
    df["texto_tfidf"] = clean_heavy_batch(df["texto"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{args.out}  ({len(df):,} quejas)")


if __name__ == "__main__":
    main()
