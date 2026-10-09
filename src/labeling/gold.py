"""Gold set: muestreo, etiquetado manual en la terminal y consistencia.

Uso:
    uv run python -m src.labeling.gold sample     # elige las quejas a etiquetar
    uv run python -m src.labeling.gold label      # etiqueta (se puede salir y retomar)
    uv run python -m src.labeling.gold stats      # avance y reparto por clase
    uv run python -m src.labeling.gold relabel    # segunda pasada, una semana después
    uv run python -m src.labeling.gold kappa      # acuerdo entre las dos pasadas
"""

import argparse
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.taxonomy import Taxonomy, load_taxonomy

COMPLAINTS_PATH = Path("data/processed/quejas.parquet")
SAMPLE_PATH = Path("data/labeled/gold_sample.csv")
GOLD_PATH = Path("data/labeled/gold.csv")
RELABEL_PATH = Path("data/labeled/gold_relabel.csv")

N_RANDOM = 400
N_TARGETED = 100
N_RELABEL = 50
SEED = 42
MIN_PER_CLASS = 20
LABEL_COLUMNS = ["review_id", "tema", "urgencia"]

# Candidatas a "postventa": la clase es demasiado rara para reunir ejemplos al azar.
POSTVENTA_PATTERN = (
    r"reembols|devoluci[oó]n|garant[ií]a|cobr[oóa]|me han cobrado|reclamaci[oó]n"
    r"|no (?:me )?(?:responde|contesta)|sin respuesta|atenci[oó]n al cliente|servicio t[eé]cnico"
)

Labels = dict[str, tuple[str, str]]


# --- Muestreo ---------------------------------------------------------------


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestra n filas repartidas en proporción a estrellas × tercil de longitud."""
    tercil = pd.qcut(df["n_tokens"], 3, labels=False, duplicates="drop")
    groups = df.groupby([df["estrellas"], tercil], sort=True)
    exact = groups.size() / len(df) * n
    quota = exact.astype(int)
    leftovers = (exact - quota).sort_values(ascending=False, kind="stable")
    quota[leftovers.index[: n - quota.sum()]] += 1
    parts = [group.sample(quota[key], random_state=seed) for key, group in groups]
    return pd.concat(parts)


def build_sample(
    df: pd.DataFrame, n_random: int = N_RANDOM, n_targeted: int = N_TARGETED, seed: int = SEED
) -> pd.DataFrame:
    """Elige las quejas del gold: una parte al azar y otra dirigida a la clase rara."""
    random_part = stratified_sample(df, n_random, seed)
    rest = df.drop(random_part.index)
    candidates = rest[rest["texto"].str.contains(POSTVENTA_PATTERN, case=False, regex=True)]
    targeted_part = candidates.sample(min(n_targeted, len(candidates)), random_state=seed)
    sample = pd.concat(
        [
            random_part[["review_id"]].assign(origen="aleatorio"),
            targeted_part[["review_id"]].assign(origen="dirigido"),
        ]
    )
    # Mezclado, para que al etiquetar no se sepa de qué parte viene cada queja.
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


# --- Lectura y escritura de etiquetas ---------------------------------------


def read_labels(path: Path) -> Labels:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return {row.review_id: (row.tema, row.urgencia) for row in df.itertuples()}


def write_labels(labels: Labels, path: Path) -> None:
    rows = [(review_id, tema, urgencia) for review_id, (tema, urgencia) in labels.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(path, index=False)


# --- Etiquetado interactivo -------------------------------------------------


def parse_answer(answer: str, taxonomy: Taxonomy) -> tuple[str, str] | None:
    """Convierte '3m' o '3 m' en (tema, urgencia). Devuelve None si no es válida."""
    compact = answer.replace(" ", "").lower()
    if len(compact) != 2 or not compact[0].isdigit():
        return None
    urgencias = {urgencia[0]: urgencia for urgencia in taxonomy.urgencia_ids}
    index = int(compact[0]) - 1
    if not 0 <= index < len(taxonomy.tema_ids) or compact[1] not in urgencias:
        return None
    return taxonomy.tema_ids[index], urgencias[compact[1]]


def format_menu(taxonomy: Taxonomy) -> str:
    temas = "   ".join(f"{i} {tema}" for i, tema in enumerate(taxonomy.tema_ids, start=1))
    urgencias = "   ".join(f"{urgencia[0]} {urgencia}" for urgencia in taxonomy.urgencia_ids)
    return f"Tema:      {temas}\nUrgencia:  {urgencias}"


def format_help(taxonomy: Taxonomy) -> str:
    lines = [f"Si hay varios motivos, gana el primero de: {', '.join(taxonomy.prioridad)}", ""]
    for i, (tema, spec) in enumerate(taxonomy.temas.items(), start=1):
        lines.append(f"{i} {tema}: {spec['definicion']}")
        lines += [f"      + {item}" for item in spec.get("incluye", [])]
        lines += [f"      - {item}" for item in spec.get("no_incluye", [])]
    lines.append("")
    for urgencia, spec in taxonomy.urgencias.items():
        lines.append(f"{urgencia[0]} {urgencia}: {spec['definicion']}")
        lines += [f"      · {item}" for item in spec["criterios"]]
    return "\n".join(lines)


def label_loop(
    review_ids: list[str],
    texts: pd.DataFrame,
    labels: Labels,
    taxonomy: Taxonomy,
    save: Callable[[Labels], None],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> Labels:
    """Recorre las quejas pendientes. Guarda tras cada respuesta, así que se puede retomar."""
    position = next((i for i, rid in enumerate(review_ids) if rid not in labels), len(review_ids))
    while position < len(review_ids):
        review_id = review_ids[position]
        row = texts.loc[review_id]
        print_fn(f"\n{'─' * 100}\n[{position + 1}/{len(review_ids)}]  {row.estrellas}★")
        print_fn(textwrap.fill(f"TÍTULO: {row.titulo}", width=100))
        print_fn(textwrap.fill(row.cuerpo, width=100))
        print_fn("\n" + format_menu(taxonomy))
        answer = input_fn("Respuesta (p. ej. 3m · ? ayuda · < anterior · q salir) > ").strip()
        if answer == "q":
            break
        if answer == "?":
            print_fn("\n" + format_help(taxonomy))
        elif answer == "<":
            position = max(position - 1, 0)
        elif (parsed := parse_answer(answer, taxonomy)) is None:
            print_fn("No entendí la respuesta: número de tema + letra de urgencia, p. ej. 3m.")
        else:
            labels[review_id] = parsed
            save(labels)
            position += 1
    return labels


# --- Avance y consistencia --------------------------------------------------


def stats_report(labels: Labels, sample: pd.DataFrame, taxonomy: Taxonomy) -> str:
    df = sample.assign(
        tema=sample["review_id"].map(lambda rid: labels.get(rid, (None, None))[0]),
        urgencia=sample["review_id"].map(lambda rid: labels.get(rid, (None, None))[1]),
    ).dropna(subset=["tema"])
    lines = [f"Etiquetadas: {len(df)} de {len(sample)}"]
    if df.empty:
        return lines[0]
    random_part = df[df["origen"] == "aleatorio"]
    lines.append(f"\n{'tema':<14}{'total':>6}{'% en muestra aleatoria':>26}")
    for tema in taxonomy.tema_ids:
        total = int((df["tema"] == tema).sum())
        share = (random_part["tema"] == tema).mean() if len(random_part) else float("nan")
        warning = f"   <- menos de {MIN_PER_CLASS}" if total < MIN_PER_CLASS else ""
        lines.append(f"{tema:<14}{total:>6}{share:>25.1%}{warning}")
    lines.append(f"\n{'urgencia':<14}{'total':>6}")
    for urgencia in taxonomy.urgencia_ids:
        lines.append(f"{urgencia:<14}{int((df['urgencia'] == urgencia).sum()):>6}")
    return "\n".join(lines)


def agreement(first: Labels, second: Labels) -> dict[str, float]:
    """Kappa de Cohen entre dos pasadas, sobre las quejas presentes en ambas."""
    common = sorted(set(first) & set(second))
    if not common:
        raise ValueError("No hay quejas etiquetadas en ambas pasadas")
    result = {"n": len(common)}
    for i, field in enumerate(("tema", "urgencia")):
        a = [first[rid][i] for rid in common]
        b = [second[rid][i] for rid in common]
        result[f"kappa_{field}"] = cohen_kappa_score(a, b)
        result[f"acuerdo_{field}"] = sum(x == y for x, y in zip(a, b, strict=True)) / len(common)
    return result


# --- CLI --------------------------------------------------------------------


def _load_texts() -> pd.DataFrame:
    return pd.read_parquet(COMPLAINTS_PATH).set_index("review_id")


def _load_sample() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        sys.exit(f"Falta {SAMPLE_PATH}. Ejecuta primero: python -m src.labeling.gold sample")
    return pd.read_csv(SAMPLE_PATH, dtype=str)


def cmd_sample(args: argparse.Namespace) -> None:
    if SAMPLE_PATH.exists() and not args.force:
        sys.exit(f"{SAMPLE_PATH} ya existe. Usa --force solo si aún no empezaste a etiquetar.")
    sample = build_sample(pd.read_parquet(COMPLAINTS_PATH))
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"{SAMPLE_PATH}: {len(sample)} quejas")
    print(sample["origen"].value_counts().to_string())


def cmd_label(_: argparse.Namespace) -> None:
    review_ids = _load_sample()["review_id"].tolist()
    labels = label_loop(
        review_ids,
        _load_texts(),
        read_labels(GOLD_PATH),
        load_taxonomy(),
        save=lambda current: write_labels(current, GOLD_PATH),
    )
    print(f"\nEtiquetadas: {len(labels)} de {len(review_ids)}. Guardado en {GOLD_PATH}")


def cmd_relabel(args: argparse.Namespace) -> None:
    gold = read_labels(GOLD_PATH)
    if len(gold) < args.n:
        sys.exit(f"Solo hay {len(gold)} quejas en el gold; hacen falta al menos {args.n}.")
    # Elección fija por semilla: la segunda pasada se puede interrumpir y retomar.
    review_ids = pd.Series(sorted(gold)).sample(args.n, random_state=SEED).tolist()
    labels = label_loop(
        review_ids,
        _load_texts(),
        read_labels(RELABEL_PATH),
        load_taxonomy(),
        save=lambda current: write_labels(current, RELABEL_PATH),
    )
    print(f"\nSegunda pasada: {len(labels)} de {len(review_ids)}. Guardado en {RELABEL_PATH}")


def cmd_stats(_: argparse.Namespace) -> None:
    print(stats_report(read_labels(GOLD_PATH), _load_sample(), load_taxonomy()))


def cmd_kappa(_: argparse.Namespace) -> None:
    result = agreement(read_labels(GOLD_PATH), read_labels(RELABEL_PATH))
    print(f"Quejas en ambas pasadas: {result['n']}")
    for field in ("tema", "urgencia"):
        print(
            f"{field:<10} kappa = {result[f'kappa_{field}']:.2f}"
            f"   acuerdo = {result[f'acuerdo_{field}']:.0%}"
        )


def main() -> None:
    # La consola de Windows no siempre es UTF-8 y las reseñas traen emojis.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="elige las quejas a etiquetar")
    sample.add_argument("--force", action="store_true", help="sobrescribe la muestra existente")
    sample.set_defaults(func=cmd_sample)
    commands.add_parser("label", help="etiqueta las quejas pendientes").set_defaults(func=cmd_label)
    commands.add_parser("stats", help="avance y reparto por clase").set_defaults(func=cmd_stats)
    relabel = commands.add_parser("relabel", help="segunda pasada para medir consistencia")
    relabel.add_argument("--n", type=int, default=N_RELABEL)
    relabel.set_defaults(func=cmd_relabel)
    commands.add_parser("kappa", help="acuerdo entre las dos pasadas").set_defaults(func=cmd_kappa)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
