# Proyecto Cero — RAG PDF Chat (Technical)

A minimal Python CLI application that implements a RAG pipeline over local PDFs. It allows natural-language questions and responds by synthesizing context retrieved from Safety Data Sheets (or other PDFs) using a configurable LLM.

## What it does
- Ingest: extracts text per page from PDFs in `data/raw` using PyMuPDF.
- Chunking: splits each page into overlapping character chunks and saves JSONL.
- Indexing: creates embeddings and persists a vector index with Chroma in `data/index`.
- Retrieval: for a query, finds the top_k most relevant chunks.
- Generation: synthesizes a short, actionable answer with an LLM (OpenAI, Groq, xAI Grok, Ollama or Bedrock) using only the retrieved context.

## Structure
- `proyecto_cero/main.py`: interactive CLI (loads index, retrieves, synthesizes answer).
- `proyecto_cero/src/proyecto_cero/rag/ingest.py`: PDF ingestion → `interim/ingest.jsonl`.
- `proyecto_cero/src/proyecto_cero/rag/chunk.py`: chunking → `interim/chunks.jsonl`.
- `proyecto_cero/src/proyecto_cero/rag/index.py`: indexing (embeddings + Chroma) → `data/index`.
- `proyecto_cero/src/proyecto_cero/rag/retrieve.py`: search (`top_k`, product/keyword filters).
- `proyecto_cero/src/proyecto_cero/rag/generate.py`: generation with LLM (selectable provider).
- `proyecto_cero/src/proyecto_cero/settings.py`: centralized configuration (reads `.env`).
- `proyecto_cero/data/raw`: input PDFs.
- `proyecto_cero/data/interim`: intermediate artifacts (jsonl).
- `proyecto_cero/data/index`: persistent Chroma index.

## Requirements
- Python 3.11+
- Install dependencies:
  - `python -m pip install -r proyecto_cero/requirements.txt`

## Configuration (.env)
Place a `.env` file at `proyecto_cero/.env`. Only secrets/provider credentials are required. Example providers:

OpenAI (paid)
- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=sk-...`
- `OPENAI_MODEL=gpt-4o-mini` (optional)

Groq (free, no model download; serves Llama/Mixtral)
- `LLM_PROVIDER=groq`
- `GROQ_API_KEY=gsk_...`
- `GROQ_MODEL=llama3-8b-8192`

xAI Grok (Grok model)
- `LLM_PROVIDER=xai`
- `XAI_API_KEY=xai-...`
- `XAI_BASE_URL=https://api.x.ai/v1`
- `XAI_MODEL=grok-2-latest`

Ollama (local, requires installing and downloading model)
- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=llama3.2:3b`

Bedrock (AWS, for deployment)
- `LLM_PROVIDER=bedrock`
- `BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0`
- `BEDROCK_REGION=us-east-1`
(Credentials via IAM Role / `aws configure`)

Optional RAG parameters (non-sensitive)
- `EMBEDDINGS_PROVIDER=sentence-transformers` | `openai`
- `EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small`
- `RAG_CHUNK_CHARS=2000`
- `RAG_CHUNK_OVERLAP=200`
- `RAG_TOP_K=6`

## Pipeline — how to run locally
From the repo root:

1) Ingest
- PowerShell: `setx PYTHONPATH "src"` (or for session: `$env:PYTHONPATH="src"`)
- `python -m proyecto_cero.rag.ingest`

2) Chunking
- `python -m proyecto_cero.rag.chunk`

3) Indexing
- `python -m proyecto_cero.rag.index`

4) CLI (ask questions)
- `python proyecto_cero/main.py`
- Type your question; the CLI prints the LLM response based only on retrieved chunks.

Notes
- If you change PDFs or the embeddings model, re-run: ingest → chunk → index.
- The retriever tolerates some typos in product names and prioritizes keywords like “first aid”, “eyes”, “fire”.

## Docker
Build image
- `docker build -t chatbot-rag:latest .`

Generate index (mount `data/` to persist)
- `docker run --rm -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest python -m proyecto_cero.rag.ingest`
- `docker run --rm -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest python -m proyecto_cero.rag.chunk`
- `docker run --rm -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest python -m proyecto_cero.rag.index`

Run CLI
- `docker run --rm -it --env-file proyecto_cero/.env -v ${PWD}/proyecto_cero/data:/app/proyecto_cero/data chatbot-rag:latest`

AWS (ECS/Fargate)
- Define secrets/vars (e.g. `XAI_API_KEY`, `LLM_PROVIDER`) in the Task.
- Mount EFS at `/app/proyecto_cero/data` to persist the index.
- For Bedrock, use `LLM_PROVIDER=bedrock` + `BEDROCK_MODEL/REGION` and supply IAM Role credentials.

## Tuning and recommendations
- Spanish: consider `EMBEDDINGS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and reindex.
- Adjust `RAG_TOP_K`, `RAG_CHUNK_CHARS`, `RAG_CHUNK_OVERLAP` based on document size and LLM.
- The system prompt and response rules live in `rag/generate.py` (`build_prompt` + system messages per provider).

## Known limitations
- If the LLM has no quota or invalid API key, the CLI will show a warning and will not print a generated response.
- Retrieval uses semantic similarity; for very specific queries, guide the query with terms from the SDS (e.g. “Section 4 first aid eyes”).