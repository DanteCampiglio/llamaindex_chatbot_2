# Proyecto Cero — RAG PDF Chat (Técnico)

Aplicación CLI en Python que implementa un pipeline RAG sobre PDFs locales. Permite hacer preguntas en lenguaje natural y responde sintetizando contexto recuperado de las Hojas de Datos de Seguridad (u otros PDFs) mediante un LLM configurable.

## Qué hace
- Ingesta: extrae texto por página de PDFs en `data/raw` usando PyMuPDF.
- Chunking: divide cada página en fragmentos solapados (caracteres) y guarda JSONL.
- Indexado: crea embeddings y persiste un índice vectorial con Chroma en `data/index`.
- Recuperación: dado un query, busca los `top_k` fragmentos más relevantes.
- Generación: sintetiza una respuesta breve y accionable con un LLM (OpenAI, Groq, xAI Grok, Ollama o Bedrock) usando únicamente el contexto recuperado.

## Estructura
- `proyecto_cero/main.py`: CLI interactivo (lee índice, recupera y sintetiza respuesta).
- `proyecto_cero/src/proyecto_cero/rag/ingest.py`: ingesta de PDFs → `interim/ingest.jsonl`.
- `proyecto_cero/src/proyecto_cero/rag/chunk.py`: chunking → `interim/chunks.jsonl`.
- `proyecto_cero/src/proyecto_cero/rag/index.py`: indexado (embeddings + Chroma) → `data/index`.
- `proyecto_cero/src/proyecto_cero/rag/retrieve.py`: búsqueda (`top_k`, filtros por producto/keywords).
- `proyecto_cero/src/proyecto_cero/rag/generate.py`: generación con LLM (proveedor seleccionable).
- `proyecto_cero/src/proyecto_cero/settings.py`: configuración centralizada (lee `.env`).
- `proyecto_cero/data/raw`: PDFs de entrada.
- `proyecto_cero/data/interim`: artefactos intermedios (jsonl).
- `proyecto_cero/data/index`: índice Chroma persistente.

## Requisitos
- Python 3.11+
- Instalar dependencias:
  - `python -m pip install -r proyecto_cero/requirements.txt`

## Configuración (.env)
Coloca un `.env` en `proyecto_cero/.env`. Solo secretos/proveedor son necesarios. Ejemplos de proveedores:

OpenAI (de pago)
- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=sk-...`
- `OPENAI_MODEL=gpt-4o-mini` (opcional)

Groq (gratis, sin descarga de modelo; sirve Llama/Mixtral)
- `LLM_PROVIDER=groq`
- `GROQ_API_KEY=gsk_...`
- `GROQ_MODEL=llama3-8b-8192`

xAI Grok (modelo Grok)
- `LLM_PROVIDER=xai`
- `XAI_API_KEY=xai-...`
- `XAI_BASE_URL=https://api.x.ai/v1`
- `XAI_MODEL=grok-2-latest`

Ollama (local, requiere instalar y descargar modelo)
- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=llama3.2:3b`

Bedrock (AWS, para despliegue)
- `LLM_PROVIDER=bedrock`
- `BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0`
- `BEDROCK_REGION=us-east-1`
(Credenciales por IAM Role/`aws configure`)

Parámetros RAG opcionales (no sensibles)
- `EMBEDDINGS_PROVIDER=sentence-transformers` | `openai`
- `EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small`
- `RAG_CHUNK_CHARS=2000`
- `RAG_CHUNK_OVERLAP=200`
- `RAG_TOP_K=6`

## Pipeline — cómo correr local
Desde la raíz del repo:

1) Ingesta
- PowerShell: `setx PYTHONPATH "src"` (o por sesión: `$env:PYTHONPATH="src"`)
- `python -m proyecto_cero.rag.ingest`

2) Chunking
- `python -m proyecto_cero.rag.chunk`

3) Indexado
- `python -m proyecto_cero.rag.index`

4) CLI (preguntar)
- `python proyecto_cero/main.py`
- Escribe la pregunta; el CLI imprime solo la respuesta del LLM basada en los chunks recuperados.

Notas
- Si cambias los PDFs o el modelo de embeddings, repite: ingest → chunk → index.
- El recuperador tolera algunos typos en productos y prioriza keywords como “primeros auxilios”, “ojos”, “incendio”.

## Docker
Construir imagen
- `docker build -t chatbot-rag:latest .`

Generar índice (monta `data/` para persistir)
- `docker run --rm -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest python -m proyecto_cero.rag.ingest`
- `docker run --rm -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest python -m proyecto_cero.rag.chunk`
- `docker run --rm -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest python -m proyecto_cero.rag.index`

Ejecutar CLI
- `docker run --rm -it --env-file proyecto_cero/.env -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest`

AWS (ECS/Fargate)
- Define secretos/vars (p. ej. `XAI_API_KEY`, `LLM_PROVIDER`) en el Task.
- Monta EFS en `/app/proyecto_cero/data` para persistir el índice.
- Para Bedrock, usa `LLM_PROVIDER=bedrock` + `BEDROCK_MODEL/REGION` y credenciales vía IAM Role.

## Tuning y recomendaciones
- Español: considera `EMBEDDINGS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` y reindexa.
- Ajusta `RAG_TOP_K`, `RAG_CHUNK_CHARS`, `RAG_CHUNK_OVERLAP` según tamaño de documentos y modelo del LLM.
- El prompt del sistema y reglas de respuesta se definen en `rag/generate.py` (`build_prompt` + mensajes system por proveedor).

## Limitaciones conocidas
- Si el LLM no tiene cuota/API válida, el CLI mostrará un aviso y no imprimirá respuesta generada.
- La recuperación usa similitud semántica; para consultas muy específicas, conviene guiar con términos de la HDS (ej. “Sección 4 primeros auxilios ojos”).

