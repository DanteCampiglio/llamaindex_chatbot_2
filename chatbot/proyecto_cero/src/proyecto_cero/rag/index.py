from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, Iterable, List, Tuple
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from proyecto_cero.settings import Settings


logger = logging.getLogger("proyecto_cero.rag.index")


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "pdf_chunks"


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_id(meta: Dict) -> str:
    key = f"{meta.get('filename','')}|p{meta.get('page','')}|c{meta.get('chunk_index','')}|{meta.get('start_char','')}|{meta.get('end_char','')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def build_index(
    chunks_path: Path,
    persist_dir: Path,
    model_name: str = DEFAULT_MODEL,
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = 128,
) -> Tuple[int, str]:
    """Construye un índice Chroma persistente a partir de chunks.jsonl.

    Devuelve (num_insertados, collection_name).
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    s = Settings.from_env()
    if s.embeddings_provider == "openai":
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=s.openai_embeddings_model,
        )
    else:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    # Chromadb >=1.x usa parámetro 'name' (antes 'collection_name')
    coll = client.get_or_create_collection(name=collection, embedding_function=embed_fn)

    docs: List[str] = []
    metas: List[Dict] = []
    ids: List[str] = []
    total = 0

    def flush_batch():
        nonlocal docs, metas, ids, total
        if not docs:
            return
        coll.upsert(documents=docs, metadatas=metas, ids=ids)
        total += len(docs)
        docs, metas, ids = [], [], []

    for rec in load_jsonl(chunks_path):
        content = (rec.get("content") or "").strip()
        meta = rec.get("metadata") or {}
        if not content:
            continue
        docs.append(content)
        metas.append(meta)
        ids.append(make_id(meta))
        if len(docs) >= batch_size:
            flush_batch()
    flush_batch()

    return total, collection


def main() -> None:
    s = Settings.from_env()
    chunks_path = s.interim_dir / "chunks.jsonl"
    persist_dir = s.index_dir

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
        )

    if not chunks_path.exists():
        logger.error("No existe el archivo de chunks: %s", chunks_path)
        return

    logger.info("Construyendo índice con %s", DEFAULT_MODEL)
    n, coll = build_index(chunks_path, persist_dir)
    logger.info("Insertados %d documentos en la colección '%s' -> %s", n, coll, persist_dir)


if __name__ == "__main__":
    main()
