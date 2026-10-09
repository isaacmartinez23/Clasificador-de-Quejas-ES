"""Limpieza de texto en dos niveles.

- clean_light: conserva tildes, mayúsculas y emojis. Para embeddings, LLM y RAG.
- clean_heavy: minúsculas, lemas, sin palabras funcionales ni tildes. Solo para TF-IDF.
"""

import html
import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from unidecode import unidecode

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
SPACE_RE = re.compile(r"\s+")

# Las palabras funcionales se descartan por categoría gramatical y no con la lista de
# stopwords de spaCy, que incluye términos clave en una queja ("llegó", "tarde", "mal",
# "peor", "diferente", "dejó", "días").
FUNCTION_POS = frozenset({"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PUNCT", "NUM", "SYM"})

# Se conservan aunque sean funcionales: "no llegó" y "llegó" son quejas opuestas.
NEGATIONS = frozenset({"no", "ni", "nunca", "jamás", "sin", "nada", "tampoco"})


def clean_light(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _nlp() -> Language:
    return spacy.load("es_core_news_sm", exclude=["parser", "ner"])


def _heavy_tokens(doc: Doc) -> str:
    tokens = []
    for token in doc:
        if token.lower_ in NEGATIONS:
            tokens.append(unidecode(token.lower_))
        elif token.is_alpha and token.pos_ not in FUNCTION_POS:
            # Con pronombre pegado ("devolverlo") el lema trae dos palabras ("devolver él").
            lemma = unidecode(token.lemma_.lower()).partition(" ")[0]
            if len(lemma) > 1:
                tokens.append(lemma)
    return " ".join(tokens)


def _prepare_for_spacy(text: str) -> str:
    # En minúsculas el lematizador acierta más con los textos escritos todo en mayúsculas.
    return clean_light(text).lower()


def clean_heavy(text: str) -> str:
    return _heavy_tokens(_nlp()(_prepare_for_spacy(text)))


def clean_heavy_batch(texts: Iterable[str], batch_size: int = 512, n_process: int = 1) -> list[str]:
    docs = _nlp().pipe(
        (_prepare_for_spacy(text) for text in texts), batch_size=batch_size, n_process=n_process
    )
    return [_heavy_tokens(doc) for doc in docs]
