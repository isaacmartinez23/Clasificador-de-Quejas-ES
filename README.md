# spanish-complaints-nlp

Clasificación automática de quejas de consumidores en español por **tema** y **urgencia**, comparando tres enfoques: TF-IDF + regresión logística, embeddings + clasificador y LLM zero/few-shot. Incluye un mini RAG y una API.

> En desarrollo. El plan completo está en [plan.md](plan.md); la tabla comparativa se publicará aquí al cerrar la Fase 3.

## Puesta en marcha

Requiere [uv](https://docs.astral.sh/uv/). La versión de Python (3.12) la instala uv.

```bash
uv sync
uv run python -m src.data.download   # reseñas crudas -> data/raw/
uv run python -m src.data.prepare    # quejas limpias -> data/processed/ (unos 4 minutos)
uv run pytest
```

## Etiquetado del gold set

La taxonomía de temas y urgencias está en [data/labeled/taxonomy.yaml](data/labeled/taxonomy.yaml). El etiquetado se hace en la terminal, guarda tras cada respuesta y se puede retomar:

```bash
uv run python -m src.labeling.gold label     # etiquetar (q para salir, ? para ver las definiciones)
uv run python -m src.labeling.gold stats     # avance y reparto por clase
uv run python -m src.labeling.gold relabel   # segunda pasada de 50 quejas, una semana después
uv run python -m src.labeling.gold kappa     # acuerdo entre las dos pasadas
```

`uv sync` instala lo necesario para datos y modelo clásico. El resto se instala por grupo cuando haga falta:

| Grupo | Para qué | Comando |
| --- | --- | --- |
| `experiments` | MLflow y notebooks | `uv sync --group experiments` |
| `embeddings` | sentence-transformers | `uv sync --group embeddings` |
| `llm` | SDKs de Gemini y OpenAI | `uv sync --group llm` |
| `rag` | ChromaDB y LangChain | `uv sync --group rag` |
| `api` | FastAPI | `uv sync --group api` |
| `demo` | Gradio | `uv sync --group demo` |

Para usar el LLM, copia `.env.example` a `.env` y completa la clave.

## Datos

[Amazon Reviews Multi](https://huggingface.co/datasets/mteb/amazon_reviews_multi) (Keung et al., 2020), split en español, copia publicada por MTEB. Los textos no se redistribuyen en este repositorio: `src/data/download.py` los descarga a `data/raw/` desde un commit fijo.
