from __future__ import annotations

import logging
from pathlib import Path
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from proyecto_cero.settings import Settings


logger = logging.getLogger("proyecto_cero.rag.retrieve")


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "pdf_chunks"


def get_collection(persist_dir: Path | None, collection: str | None = None):
    s = Settings.from_env()
    persist_dir = persist_dir or s.index_dir
    collection = collection or s.collection_name
    client = chromadb.PersistentClient(path=str(persist_dir))
    if s.embeddings_provider == "openai":
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=s.openai_embeddings_model,
        )
    else:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=s.embeddings_model
        )
    return client.get_or_create_collection(name=collection, embedding_function=embed_fn)


def _detect_where_filter(query: str):
    q = query.lower()
    if "acelepryn" in q:
        return {"filename": {"$eq": "ficha-seguridad-acelepryn.pdf"}}
    if "amistar" in q:
        return {"filename": {"$eq": "AMISTAR XTRA_hoja_de_seguridad (1).pdf"}}
    if "abofol" in q:
        return {"filename": {"$eq": "ficha-seguridad-abofol-l (1).pdf"}}
    return None


def search(persist_dir: Path, query: str, top_k: int = 4):
    coll = get_collection(persist_dir)
    where = _detect_where_filter(query)
    # Tolerancia a typo del producto (p.ej., "abofoll" -> abofol)
    if where is None and "abofoll" in query.lower():
        where = {"filename": {"$eq": "ficha-seguridad-abofol-l (1).pdf"}}
    # Filtrado por contenido del documento si hay pistas en la consulta
    q = query.lower()
    where_document = None
    for kw in ("quemadura", "primeros auxilios", "incendio", "contacto con la piel", "ojos", "ocular"):
        if kw in q:
            where_document = {"$contains": kw}
            break

    def run_query(_where, _where_doc):
        if _where is not None and _where_doc is not None:
            return coll.query(query_texts=[query], n_results=top_k, where=_where, where_document=_where_doc)
        if _where is not None:
            return coll.query(query_texts=[query], n_results=top_k, where=_where)
        if _where_doc is not None:
            return coll.query(query_texts=[query], n_results=top_k, where_document=_where_doc)
        return coll.query(query_texts=[query], n_results=top_k)

    # intentamos con where_document si lo hay y luego hacemos fallback si viene vacío
    result = run_query(where, where_document)
    docs = result.get("documents", [[]])[0] if isinstance(result, dict) else []
    if not docs and where_document is not None:
        result = run_query(where, None)
        docs = result.get("documents", [[]])[0] if isinstance(result, dict) else []
    # Fallback final: sin filtros si seguimos sin resultados
    if not docs:
        result = coll.query(query_texts=[query], n_results=top_k)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    persist_dir = root / "data" / "index"

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
        )

    print("Escribe tu consulta (ENTER para salir):")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        res = search(persist_dir, q, top_k=4)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        print("\nResultados:")
        for i, (d, m) in enumerate(zip(docs, metas), start=1):
            src = f"{m.get('filename','')}#p{m.get('page','')} c{m.get('chunk_index','')}"
            preview = d[:300].replace("\n", " ") + ("..." if len(d) > 300 else "")
            print(f"{i}. {src}\n   {preview}\n")


if __name__ == "__main__":
    main()
