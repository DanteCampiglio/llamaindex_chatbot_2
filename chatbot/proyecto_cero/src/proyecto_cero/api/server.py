from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path

from proyecto_cero.settings import Settings
from proyecto_cero.rag.retrieve import search as rag_search
from proyecto_cero.rag.generate import answer_with_llm


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    only_retrieve: bool = False
    max_context: int = 6


class SourceItem(BaseModel):
    filename: str
    page: int
    chunk_index: int
    preview: str


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    provider: Optional[str] = None
    sources: List[SourceItem] = []


app = FastAPI(title="Proyecto Cero - RAG API", version="1.0.0")

# Libre por defecto para facilitar consumo desde N8N
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    s = Settings.from_env()
    persist_dir = s.index_dir
    if not persist_dir.exists():
        raise HTTPException(status_code=500, detail=f"No existe el índice en '{persist_dir}'. Ejecuta ingest -> chunk -> index")

    top_k = req.top_k or s.top_k
    res = rag_search(persist_dir, req.question, top_k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    if not docs:
        return QueryResponse(answer=None, provider=None, sources=[])

    # Preparar fuentes
    max_ctx = max(1, min(req.max_context, len(docs)))
    sources: List[SourceItem] = []
    for d, m in list(zip(docs, metas))[:max_ctx]:
        preview = (d[:300].replace("\n", " ") + ("..." if len(d) > 300 else ""))
        sources.append(
            SourceItem(
                filename=str(m.get("filename", "")),
                page=int(m.get("page", 0)),
                chunk_index=int(m.get("chunk_index", 0)),
                preview=preview,
            )
        )

    if req.only_retrieve:
        return QueryResponse(answer=None, provider=None, sources=sources)

    # Generación con LLM
    answer, provider = answer_with_llm(req.question, docs[:max_ctx], metas[:max_ctx], settings=s)
    return QueryResponse(answer=answer, provider=provider, sources=sources)


# Comando para correr con: uvicorn proyecto_cero.api.server:app --host 0.0.0.0 --port 8000


@app.post("/answer", response_class=PlainTextResponse)
def answer_only(req: QueryRequest) -> str:
    """Devuelve solo el texto de la respuesta (sin sources ni provider).

    Útil para integraciones tipo n8n que esperan una cadena simple.
    """
    s = Settings.from_env()
    persist_dir = s.index_dir
    if not persist_dir.exists():
        raise HTTPException(status_code=500, detail=f"No existe el índice en '{persist_dir}'. Ejecuta ingest -> chunk -> index")

    top_k = req.top_k or s.top_k
    res = rag_search(persist_dir, req.question, top_k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    if not docs:
        return ""  # vacío si no hubo recuperación

    max_ctx = max(1, min(req.max_context, len(docs)))
    if req.only_retrieve:
        # Si piden solo retrieve, devolvemos previews concatenadas como texto
        previews = []
        for d in list(docs)[:max_ctx]:
            previews.append((d[:300].replace("\n", " ") + ("..." if len(d) > 300 else "")))
        return "\n".join(previews)

    answer, _provider = answer_with_llm(req.question, docs[:max_ctx], metas[:max_ctx], settings=s)
    return answer or ""
