# Proyecto Cero — RAG PDF Chat (README TÉCNICO)

Aplicación CLI y API REST en Python que implementa un pipeline RAG sobre PDFs locales. Permite hacer preguntas en lenguaje natural y responde sintetizando contexto recuperado de Hojas de Datos de Seguridad (u otros PDFs) mediante un LLM configurable por `.env`.

## 1) Arquitectura
- Ingesta: extrae texto por página (PyMuPDF) desde `data/raw/*.pdf` → `data/interim/ingest.jsonl`.
- Chunking: trozos solapados (por caracteres) → `data/interim/chunks.jsonl`.
- Indexado: embeddings + vector store persistente (Chroma) → `data/index`.
- Recuperación: búsqueda semántica `top_k` + filtros heurísticos (producto/keywords) → lista de chunks.
- Generación: LLM sintetiza respuesta usando exclusivamente el contexto recuperado.
- Exposición: CLI (`main.py`) y API FastAPI (`/query`, `/answer`, `/healthz`).

## 2) Estructura del proyecto
- `proyecto_cero/main.py`: CLI (recupera + sintetiza y muestra solo la respuesta).
- `proyecto_cero/src/proyecto_cero/rag/ingest.py`: PDFs → JSONL por página.
- `proyecto_cero/src/proyecto_cero/rag/chunk.py`: páginas → chunks solapados.
- `proyecto_cero/src/proyecto_cero/rag/index.py`: embeddings + Chroma (persistencia).
- `proyecto_cero/src/proyecto_cero/rag/retrieve.py`: búsqueda `top_k` con filtros.
- `proyecto_cero/src/proyecto_cero/rag/generate.py`: generación con LLM (proveedor seleccionable).
- `proyecto_cero/src/proyecto_cero/api/server.py`: FastAPI (`/query`, `/answer`, `/healthz`).
- `proyecto_cero/src/proyecto_cero/settings.py`: Settings centralizados (lee `.env`).
- `proyecto_cero/data/`: `raw/`, `interim/`, `index/`.

## 3) Modelos y librerías
- PDFs: `pymupdf` (rápido, sin OCR).
- Embeddings: `sentence-transformers` (default local: `all-MiniLM-L6-v2`); opcional OpenAI (`text-embedding-3-small`).
- Vector store: `chromadb` (persistente en disco).
- LLMs soportados (por `.env`): `xAI Grok` (base_url compatible OpenAI), `Groq` (Llama/Mixtral), `OpenAI`, `Ollama` (local), `AWS Bedrock` (Claude, etc.).
- API: `fastapi` + `uvicorn`.

## 4) Configuración (.env)
Colocar en `proyecto_cero/.env` solo secretos/proveedor. Ejemplos:

OpenAI (pago)
- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=sk-...`
- `OPENAI_MODEL=gpt-4o-mini` (opcional)

Groq (gratis, sin descargas)
- `LLM_PROVIDER=groq`
- `GROQ_API_KEY=gsk_...`
- `GROQ_MODEL=llama3-8b-8192`

xAI Grok
- `LLM_PROVIDER=xai`
- `XAI_API_KEY=xai-...`
- `XAI_BASE_URL=https://api.x.ai/v1`
- `XAI_MODEL=grok-2-latest`

Ollama (local)
- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=llama3.2:3b`

Bedrock (AWS)
- `LLM_PROVIDER=bedrock`
- `BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0`
- `BEDROCK_REGION=us-east-1`

Parámetros RAG opcionales (no sensibles)
- `EMBEDDINGS_PROVIDER=sentence-transformers|openai`
- `EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small`
- `RAG_CHUNK_CHARS=2000`, `RAG_CHUNK_OVERLAP=200`, `RAG_TOP_K=6`

## 5) Pipeline (local)
- Instalar deps: `python -m pip install -r proyecto_cero/requirements.txt`
- Preparar índice:
  - `$env:PYTHONPATH="src"; python -m proyecto_cero.rag.ingest`
  - `$env:PYTHONPATH="src"; python -m proyecto_cero.rag.chunk`
  - `$env:PYTHONPATH="src"; python -m proyecto_cero.rag.index`
- CLI: `python proyecto_cero/main.py`
- API: `$env:PYTHONPATH="src"; uvicorn proyecto_cero.api.server:app --host 0.0.0.0 --port 8000`

## 6) Docker
- Build: `docker build -t chatbot-rag:latest .`
- Ingest/Chunk/Index: montar `proyecto_cero/data` y ejecutar módulos en el contenedor.
- API: `docker run --rm -it --env-file proyecto_cero/.env -v "${PWD}/proyecto_cero/data:/app/proyecto_cero/data" -p 8000:8000 chatbot-rag:latest uvicorn proyecto_cero.api.server:app --host 0.0.0.0 --port 8000`

## 7) Exposición a n8n (ngrok)
- `ngrok http 8000` → usar URL pública.
- Endpoints:
  - GET `/healthz` (health)
  - GET `/docs` (Swagger)
  - POST `/answer` (solo texto): body `{"question":"...","top_k":8}`
  - POST `/query` (JSON completo): body `{ "question":"...","top_k":8, "only_retrieve": false }`
- n8n (solo texto): HTTP Request → POST `/answer` → Response format: String.

## 8) Heurísticas de recuperación
- Filtros por producto (amistar/acelepryn/abofol) con tolerancia de typo (p.ej. "abofoll"→"abofol").
- Keywords de dominio: "primeros auxilios", "ojos/ocular", "incendio".
- Fallbacks progresivos: con filtros → sólo contenido → sin filtros.

## 9) Por qué es un buen proyecto
- Modular, configurable y portable (local ↔ cloud) sin cambios de código.
- Index persistente y datos intermedios reproducibles.
- API y CLI listas para integraciones.
- Contenedorizado; fácil de llevar a ECS/Fargate.

## 10) Limitaciones actuales
- Sin reranking por cross-encoder (potencial mejora de precisión).
- Sin autenticación nativa (opcional añadir `X-API-Key` y rate limiting).
- Sin evaluación QA automática ni métricas de latencia/tokens.
- Sin endpoint de subida de PDFs y reindexado on-the-fly.

## 11) Próximos pasos
- Reranking (cross-encoder) o BM25+vectores.
- Auth por token en FastAPI, CORS restringido y despliegue tras reverse proxy con TLS.
- Métricas/observabilidad: tiempo de recuperación, uso LLM, logs estructurados.
- Endpoint para subir/actualizar PDFs y reindexar incrementalmente.
- Cache de respuestas por `(query, index_version)`.
- Streaming de respuestas (SSE) para mejor UX.
- Integración Bedrock completa (incl. embeddings Titan) y Secrets Manager para claves.

## 12) Checklist rápido
- `.env` con LLM_PROVIDER y su API key (o Ollama local).
- PDFs en `data/raw`.
- Índice construido (ingest → chunk → index).
- API en `0.0.0.0:8000` y ngrok apuntando a ese puerto si se comparte.

