# Plan de desarrollo — Clasificador de Quejas en Español

> **Pregunta de negocio:** ¿Se pueden clasificar automáticamente las quejas de los consumidores por tema y urgencia para atenderlas más rápido? ¿Qué enfoque conviene más en costo, precisión y velocidad?

Comparativa de tres generaciones de NLP (TF-IDF, embeddings, LLM) sobre textos en español, más un mini RAG y una API. Repo: `spanish-complaints-nlp`.

---

## 1. Decisiones de diseño a cerrar antes de programar

El brief deja varios puntos abiertos que condicionan todo lo demás. Cada uno trae una propuesta por defecto; si no se cambia, el plan asume esa.

| # | Tema | Problema | Propuesta por defecto |
| --- | --- | --- | --- |
| D1 | **Fuente del dataset** | Amazon retiró el repo oficial de Hugging Face en 2023; se usa la copia de MTEB. | **Resuelto:** `load_dataset("mteb/amazon_reviews_multi", "es")` (ver §2.1). La copia no declara licencia, así que **no se publican los textos crudos en el repo**: solo el script de descarga y los IDs + etiquetas del gold set. |
| D2 | **Reseñas ≠ quejas** | El dataset son reseñas de 1 a 5 estrellas, no quejas. | Usar solo reseñas de **1–2 estrellas** como "quejas": `label ∈ {0, 1}`, unas 80 000 de las 200 000 de train (el corpus está balanceado al 20 % por estrella). |
| D3 | **De dónde salen las etiquetas de entrenamiento** | El gold set (300–500) alcanza para evaluar, no para entrenar TF-IDF con 7–8 clases. | **Gold set = solo evaluación**, etiquetado a mano y nunca usado para entrenar. Para entrenar: **silver set** de 5–10 k textos etiquetados con LLM y auditados por muestreo. Reportar además una curva de aprendizaje entrenando solo con datos humanos. |
| D4 | **Circularidad LLM-etiqueta / LLM-clasifica** | Si el LLM etiqueta el silver set y luego compite como clasificador, la comparación se sesga a su favor. | La métrica oficial se mide **solo contra el gold humano**. Etiquetar el gold a ciegas (sin ver la sugerencia del LLM). Documentar el sesgo residual en el README. |
| D5 | **Taxonomía** | Los ejemplos del brief ("cobro indebido", "aerolíneas") no encajan del todo con reseñas de productos. | Taxonomía v1 adaptada a e-commerce (ver §3), pendiente de confirmar. Esta copia del dataset no trae categoría de producto, así que la pregunta demo del RAG se apoya en búsqueda semántica por tipo de producto: *"¿Cuáles son las quejas más comunes sobre auriculares?"*. |
| D6 | **Urgencia es subjetiva** | Sin rúbrica, el F1 de urgencia mide ruido. | Rúbrica escrita de 3 niveles con ejemplos (ver §3) y medir consistencia: re-etiquetar 50 textos una semana después y reportar kappa de Cohen. |
| D7 | **Alternativas del stack** | El brief lista opciones "A o B". | **ChromaDB** (persistencia simple), **Gradio** (despliegue directo en HF Spaces), **LangChain** solo en la capa RAG, proveedor LLM detrás de una interfaz (`LLM_PROVIDER=gemini\|openai`) para poder comparar ambos sin reescribir. |
| D8 | **Costo de API** | Sin tope, el etiquetado silver y los experimentos pueden dispararse. | Presupuesto máximo definido en `.env` (p. ej. 10 USD), caché en disco de todas las respuestas del LLM por hash de (prompt, texto, modelo), y estimación de tokens antes de cada corrida masiva. |

---

## 2. Arquitectura

```
HF dataset ──> clean_text ──> processed/*.parquet ──┬──> gold set (humano)  ──> evaluación
                                                    └──> silver set (LLM)   ──> entrenamiento
                                                                 │
                    ┌────────────────────────┬───────────────────┴─────────┐
                    v                        v                             v
              tfidf_model            embedding_model                llm_classifier
                    └────────────────────────┴──────────────┬──────────────┘
                                                            v
                                             evaluation/metrics (F1, costo, latencia)
                                                            v
                                         MLflow ──> 05_comparativa ──> README

processed ──> rag/build_index (Chroma) ──> rag/query ──┐
                                                       ├──> api/main.py ──> Docker
mejor clasificador ────────────────────────────────────┘         └──> app/gradio_app.py ──> HF Spaces
```

**Contrato común de los tres clasificadores** (para que la comparación sea justa y la API pueda intercambiarlos):

```python
class Classifier(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


@dataclass
class Prediction:
    tema: str  # una de las clases de la taxonomía
    urgencia: str  # alta | media | baja
    confianza: float | None
    latencia_ms: float
    costo_usd: float  # 0.0 en modelos locales
```

### 2.1 Dataset

```python
from datasets import load_dataset

ds = load_dataset("mteb/amazon_reviews_multi", "es")
```

**Nota:** el repo define la config `es` con un script de carga, y las versiones actuales de `datasets` (5.x, la instalada) ya no ejecutan scripts, así que ese snippet falla tal cual. `src/data/download.py` obtiene los mismos datos leyendo los JSONL del repo (`es/train.jsonl`, etc.) con el cargador `json`, desde el commit fijo `c379a67`. Para descargar: `uv run python -m src.data.download`.

Verificado con la descarga real el 2026-10-08:

| Aspecto | Valor |
| --- | --- |
| Acceso | Público, sin registro ni aprobación |
| Splits | `train` 200 000 · `validation` 5 000 · `test` 5 000 |
| Tamaño | ~26 MB en parquet |
| Columnas | `id`, `text`, `label`, `label_text` |
| `id` | Identificador de la reseña (p. ej. `es_0491108`); es el `review_id` del plan |
| `text` | Título y cuerpo unidos por una línea en blanco (`"título\n\ncuerpo"`) |
| `label` | Estrellas − 1, de 0 a 4 → las quejas son `label` 0 y 1 |

**Lo que esta copia no trae:** categoría de producto, ID de producto ni ID de autor (el dataset original sí los tenía). Consecuencias: no se puede estratificar ni filtrar por categoría, y el título hay que separarlo de `text` si se quiere tratar aparte.

---

## 3. Taxonomía v1

Definida tras el EDA y la lectura de una muestra de quejas. La fuente de verdad es `data/labeled/taxonomy.yaml`, con definiciones, casos que incluye y que no incluye cada clase.

**Tema** (una sola etiqueta: el motivo principal)

| Clase | Qué es |
| --- | --- |
| `entrega` | No llegó, llegó tarde o hubo problemas con el reparto |
| `defectuoso` | Llegó roto o dañado, no funciona, o falló al poco tiempo |
| `no_coincide` | Distinto de lo anunciado: otro modelo, talla, medidas, compatibilidad, falsificación |
| `incompleto` | Faltan unidades, piezas, accesorios o instrucciones |
| `calidad` | Funciona y es el anunciado, pero es de mala calidad o rinde poco |
| `postventa` | Devoluciones, reembolsos, garantía, vendedor que no responde y cobros indebidos |
| `otro` | Gusto personal, insatisfacción genérica o no es una queja |

Si una queja tiene varios motivos, gana el primero de: `entrega`, `postventa`, `incompleto`, `no_coincide`, `defectuoso`, `calidad`, `otro`.

**Cambios frente a la v0**
- `cobro`, `devolucion_garantia` y `atencion` se fusionan en **`postventa`**: en reseñas de producto casi nunca son el motivo principal (lo mencionan el 6,6 % de las quejas, casi siempre junto a otro problema). Separadas no reunirían 20 ejemplos en el gold.
- Nueva clase **`incompleto`**, frecuente en la muestra leída. Solo la menciona el 3,6 %: si tras etiquetar queda por debajo del 5 %, se fusiona con `no_coincide`.
- `calidad` absorbe el bajo rendimiento («funciona, pero mal»), y el gusto personal pasa a `otro`.

**Urgencia**

| Nivel | Criterio |
| --- | --- |
| `alta` | Riesgo de seguridad o salud, daño a otros bienes, o pagó y no tiene ni producto ni reembolso |
| `media` | Se quedó sin un producto utilizable y necesita reemplazo, devolución o reparación |
| `baja` | Producto utilizable pero decepcionante, problema ya resuelto, u opinión sin petición |

---

## 4. Fases

Estimaciones para trabajo a tiempo parcial (~10 h/semana). Son orientativas.

### Fase 0 — Setup (2–3 días)

- [x] `git init`, `.gitignore` (`data/raw/`, `data/processed/`, `mlruns/`, `.env`, índices vectoriales)
- [x] `pyproject.toml` con `uv`. Dependencias base (datos + modelo clásico) y grupos opcionales: `experiments`, `embeddings`, `llm`, `rag`, `api`, `demo`, `dev`. Solo base y `dev` están instalados; el resto está resuelto en `uv.lock` y se instala con `uv sync --group <nombre>`
- [x] Fijar **Python 3.12** en el proyecto (`.python-version`)
- [x] Crear la estructura de carpetas del brief
- [x] `.env.example`: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BUDGET_USD`
- [x] `ruff` + `pytest` + pre-commit (hook instalado)
- [x] `src/data/download.py`: descarga a `data/raw/*.parquet` y valida splits, columnas, filas y etiquetas
- [x] Modelo `es_core_news_sm` 3.8.0 como dependencia en `pyproject.toml` (no hace falta `spacy download`)
- [ ] Primer commit y subir a GitHub; confirmar que la CI (`.github/workflows/ci.yml`) pasa — escrita pero aún sin ejecutar

**Hecho cuando:** `uv run pytest` pasa en limpio y un script descarga el dataset a `data/raw/`. → Cumplido el 2026-10-08: 7 tests pasan; 200 000 / 5 000 / 5 000 filas descargadas, 80 000 quejas en `train`.

### Fase 1 — Datos y etiquetado (1.5–2 semanas)

**Hecho (2026-10-08)**
- [x] `src/data/prepare.py`: filtra 1–2 estrellas del split `train`, separa título y cuerpo, limpia y deduplica → `data/processed/quejas.parquet` con **79 753 quejas** (`validation` y `test` quedan sin tocar como reserva)
- [x] `src/preprocessing/clean_text.py`: `clean_light` (embeddings, LLM, RAG) y `clean_heavy` (TF-IDF). `clean_heavy` descarta palabras funcionales por categoría gramatical en vez de usar la lista de stopwords de spaCy, que eliminaba términos clave («llegó», «tarde», «mal», «dejó»), y conserva las negaciones
- [x] Filtros: menos de 5 palabras (191 descartadas) y duplicados (56). **Sin filtro de idioma**: el detector probado marcaba como portugués o catalán textos en español, y el 99,8 % del corpus ya es español
- [x] `notebooks/01_eda_texto.ipynb`: longitud, términos frecuentes, diferencias entre 1 y 2 estrellas, señales por tema y conclusiones
- [x] Taxonomía v1 y rúbrica de urgencia en `data/labeled/taxonomy.yaml` (ver §3) — **pendiente de tu visto bueno antes de etiquetar**
- [x] `src/labeling/gold.py`: muestreo, etiquetado en la terminal, avance y kappa
- [x] Muestra del gold en `data/labeled/gold_sample.csv`: 400 quejas al azar (estratificadas por estrellas y longitud) + 100 dirigidas a candidatas de `postventa`, mezcladas

**Pendiente — trabajo manual**
- [ ] Revisar `taxonomy.yaml` y ajustarla si hace falta (si cambia después de empezar, hay que re-etiquetar)
- [ ] Etiquetar las 500 quejas: `uv run python -m src.labeling.gold label` (se puede salir con `q` y retomar)
- [ ] Revisar el reparto con `uv run python -m src.labeling.gold stats`; fusionar las clases con menos de 20 ejemplos
- [ ] Una semana después, segunda pasada de 50 quejas (`gold relabel`) y calcular el acuerdo (`gold kappa`)

**Pendiente — requiere clave de API**
- [ ] **Silver set**: etiquetar 5–10 k quejas con LLM (excluyendo las del gold); auditar 100 al azar y reportar el acuerdo con el criterio humano. Necesita `.env` con proveedor, modelo y clave, y la taxonomía ya confirmada
- [ ] Segundo EDA con etiquetas reales: n-gramas por tema y distribución de clases

**Hecho cuando:** existen `gold.csv` (≥ 300 filas, ninguna clase con menos de 20 ejemplos), `silver.parquet`, el kappa está documentado y ningún `review_id` del gold aparece en el silver.

### Fase 2 — Modelo clásico (1 semana)

- [ ] `src/evaluation/metrics.py`: F1 macro, F1 por clase, matriz de confusión, latencia p50/p95, costo por 1 000 textos, intervalo de confianza por bootstrap
- [ ] `src/models/tfidf_model.py`: `Pipeline(TfidfVectorizer, clasificador)` para tema y otro para urgencia
- [ ] Búsqueda con validación cruzada estratificada (5 folds) sobre el silver: n-gramas de palabra (1–2) frente a caracteres (3–5), `min_df`, `C`, `class_weight`, LogisticRegression frente a LinearSVC
- [ ] Registrar cada corrida en MLflow: parámetros, métricas de CV, métricas en gold, artefacto del modelo
- [ ] `notebooks/02_tfidf_baseline.ipynb`: matriz de confusión en gold, términos con más peso por clase, 20 errores leídos a mano y categorizados
- [ ] Baselines triviales como piso: clase mayoritaria y reglas de palabras clave

**Hecho cuando:** hay un modelo TF-IDF serializado, su F1 macro en gold con intervalo de confianza, y una lista escrita de los tipos de error dominantes.

### Fase 3 — Embeddings y LLM (1.5–2 semanas)

**Embeddings**
- [ ] `src/models/embedding_model.py`: `paraphrase-multilingual-MiniLM-L12-v2` + LogisticRegression sobre los vectores
- [ ] Cachear embeddings en disco (`.npy` con el hash del modelo)
- [ ] Curva de aprendizaje: F1 en gold entrenando con 50 / 200 / 1 000 / 5 000 ejemplos, TF-IDF frente a embeddings
- [ ] Opcional: comparar con un modelo de embeddings más grande (`multilingual-e5-base`)

**LLM**
- [ ] `prompts/classifier_v1.txt`: taxonomía inyectada desde `taxonomy.yaml`, definiciones de cada clase, formato de salida
- [ ] `src/models/llm_classifier.py`: salida estructurada (JSON schema del proveedor) validada con Pydantic; reintento con backoff; respuesta inválida → `otro` y se cuenta como fallo de parseo
- [ ] Zero-shot y few-shot (3–5 ejemplos por clase, tomados del silver, **nunca del gold**)
- [ ] Medir por llamada: tokens de entrada y salida, latencia, costo según la tarifa vigente del proveedor
- [ ] Versionar prompts (`classifier_v2.txt`…) y registrar la versión en MLflow
- [ ] Ajustar el prompt contra una porción del silver, no contra el gold, para no sobreajustar la evaluación

**Comparativa**
- [ ] `notebooks/05_comparativa.ipynb`: tabla final + gráfico de F1 frente a costo

| Enfoque | F1 macro tema | F1 macro urgencia | Latencia p50 | Latencia p95 | Costo / 1 000 textos | Datos etiquetados que requiere |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + LR | | | | | | |
| Embeddings + LR | | | | | | |
| LLM zero-shot | | | | | | |
| LLM few-shot | | | | | | |

- [ ] Análisis de desacuerdos: textos donde los tres enfoques fallan frente a donde solo falla uno
- [ ] Punto de equilibrio: a partir de cuántos textos al mes conviene entrenar un modelo propio en vez de pagar el LLM

**Hecho cuando:** la tabla está completa con intervalos de confianza y todas las métricas salen de la misma función de `metrics.py` sobre el mismo gold.

### Fase 4 — RAG y API (1.5 semanas)

**RAG**
- [ ] `src/rag/build_index.py`: indexar las quejas en ChromaDB con metadatos (`review_id`, `estrellas`, `tema` y `urgencia` predichos)
- [ ] `src/rag/query.py`: recuperación top-k con filtro por metadatos → respuesta del LLM que cita `review_id` → validar que todo ID citado esté entre los recuperados
- [ ] Las preguntas de tipo "más comunes" son agregaciones, no búsquedas: recuperar un top-N amplio (p. ej. 200) por similitud con el tipo de producto, contar por `tema` sobre ese conjunto y pasar al LLM el conteo más unos pocos ejemplos, en vez de pedirle que infiera frecuencias a partir de 5 fragmentos
- [ ] Set de 15–20 preguntas de prueba con respuesta esperada; evaluar a mano fidelidad y validez de las citas

**API**
- [ ] `api/main.py`:
  - `POST /classify` → `{tema, urgencia, confianza, modelo, latencia_ms}`; parámetro `model=tfidf|embeddings|llm`
  - `POST /ask` → `{respuesta, fuentes: [{review_id, fragmento, tema}]}`
  - `GET /health`
- [ ] Modelos cargados una sola vez al arrancar; validación con Pydantic; límite de longitud del texto de entrada
- [ ] `api/Dockerfile` multi-stage, imagen `python:3.12-slim`, torch solo CPU para contener el tamaño
- [ ] Tests con `TestClient` y el LLM simulado

**Demo**
- [ ] `app/gradio_app.py`: pestaña "Clasificar" (los tres enfoques lado a lado con latencia y costo) y pestaña "Preguntar" (RAG con fuentes)
- [ ] Desplegar en HF Spaces; clave en *Secrets*; **límite de uso** para las pestañas que llaman al LLM, porque la demo es pública y gasta de la clave propia

**Hecho cuando:** `docker run` levanta la API y responde ambos endpoints, y la demo pública funciona sin exponer la clave.

### Fase 5 — Portafolio (1 semana)

- [ ] README: pregunta de negocio, tabla comparativa, gráfico F1 frente a costo, recomendación de qué enfoque usar y cuándo, limitaciones, cómo reproducir
- [ ] Sección de limitaciones honesta: reseñas usadas como proxy de quejas, gold set pequeño etiquetado por una sola persona, silver etiquetado por LLM
- [ ] Capturas o GIF de la demo
- [ ] Post: *"¿Vale la pena usar un LLM para clasificar texto? Comparé 3 enfoques"*
- [ ] Sección **Inteligencia Artificial** del sitio web con enlace a la demo y al repo

---

## 5. Tests mínimos

| Archivo | Qué cubre |
| --- | --- |
| `tests/test_clean_text.py` | Tildes, emojis, URLs, texto vacío, `clean_light` frente a `clean_heavy` |
| `tests/test_metrics.py` | F1 macro contra un caso calculado a mano; cálculo de costo |
| `tests/test_llm_parsing.py` | JSON válido, JSON roto, clase fuera de la taxonomía (LLM simulado) |
| `tests/test_no_leakage.py` | Ningún `review_id` del gold aparece en silver ni en los ejemplos few-shot |
| `tests/test_api.py` | `/classify` y `/ask` devuelven el esquema esperado; entrada inválida → 422 |
| `tests/test_rag.py` | Toda fuente citada existe en el índice |

---

## 6. Métricas de éxito

| Métrica | Objetivo | Cómo se verifica |
| --- | --- | --- |
| F1 macro (tema) del mejor enfoque en gold | ≥ 0.80 | `05_comparativa.ipynb`, con intervalo de confianza |
| RAG con fuentes citadas | 100 % de las respuestas citan IDs recuperados | `tests/test_rag.py` + set de preguntas |
| Costo por 1 000 clasificaciones | Documentado para los 4 enfoques | Tabla del README |
| Reproducibilidad | Un clon nuevo reproduce la tabla | Sección "Cómo reproducir" probada en limpio |

Si el F1 no llega a 0.80: fusionar clases confundibles, reportar el resultado real y explicar por qué. Para el portafolio vale más una comparativa honesta que un número inflado.

---

## 7. Riesgos

| Riesgo | Señal | Mitigación |
| --- | --- | --- |
| La copia de MTEB desaparece de Hugging Face o cambia | Falla la descarga en un clon nuevo | La revisión ya está fijada en `src/data/download.py`; conservar además el parquet local fuera de git |
| El dataset no declara licencia | — | No redistribuir textos; citar el paper original (Keung et al., 2020) en el README; en la demo pública mostrar solo fragmentos cortos como fuentes |
| Sin categoría de producto, el RAG no puede filtrar por sector | Preguntas por tipo de producto devuelven resultados mezclados | Apoyarse en búsqueda semántica; si no basta, inferir una categoría gruesa con el LLM para el subconjunto indexado |
| Etiquetar a mano toma más de lo previsto | Menos de 50 textos por hora | Bajar el gold a 300 y reducir clases |
| Urgencia poco consistente | Kappa < 0.6 | Pasar a dos niveles (alta / no alta) o reportarla como tarea secundaria |
| Clases muy desbalanceadas | `cobro` o `atencion` casi vacías | Fusionar, o sobremuestrear esas clases al armar el gold |
| El gold es chico para distinguir enfoques | Intervalos de confianza que se solapan | Decirlo explícitamente; la recomendación se apoya entonces en costo y latencia |
| Costo del LLM se dispara | Estimación previa > presupuesto | Caché, modelo más barato para el silver, reducir el silver a 5 k |
| La imagen Docker o el Space quedan pesados | Imagen > 2 GB o Space sin memoria | Torch CPU; servir TF-IDF por defecto y cargar embeddings bajo demanda |

---

## 8. Orden de ejecución sugerido

1. Fase 0 completa, con el dataset descargado en `data/raw/`
2. EDA → taxonomía v1 → gold set (el etiquetado es la ruta crítica; empezarlo cuanto antes)
3. `metrics.py` antes que cualquier modelo, para que todos se midan igual
4. TF-IDF → embeddings → LLM (cada uno deja un piso para el siguiente)
5. RAG y API en paralelo una vez elegido el clasificador por defecto
6. README y post al final, con los números ya congelados

**Duración total estimada:** 6–8 semanas a tiempo parcial.
