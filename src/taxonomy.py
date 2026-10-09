"""Carga y valida la taxonomía de temas y urgencias."""

from dataclasses import dataclass
from pathlib import Path

import yaml

TAXONOMY_PATH = Path("data/labeled/taxonomy.yaml")


@dataclass(frozen=True)
class Taxonomy:
    version: int
    temas: dict[str, dict]
    urgencias: dict[str, dict]
    prioridad: list[str]

    @property
    def tema_ids(self) -> list[str]:
        return list(self.temas)

    @property
    def urgencia_ids(self) -> list[str]:
        return list(self.urgencias)


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    taxonomy = Taxonomy(
        version=data["version"],
        temas=data["temas"],
        urgencias=data["urgencias"],
        prioridad=data["prioridad"],
    )
    if sorted(taxonomy.prioridad) != sorted(taxonomy.temas):
        raise ValueError("'prioridad' debe listar exactamente los mismos temas que 'temas'")
    return taxonomy
