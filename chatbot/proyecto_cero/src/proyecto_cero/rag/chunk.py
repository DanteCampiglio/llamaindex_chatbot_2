from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


logger = logging.getLogger("proyecto_cero.rag.chunk")


@dataclass
class ChunkRecord:
    content: str
    metadata: Dict


def chunk_text(
    text: str,
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> Iterator[tuple[str, int, int]]:
    """Divide texto largo en trozos por caracteres con solape.

    - Intenta cortar en límite de palabra (espacio) cerca del tope.
    - Devuelve (chunk, start_idx, end_idx) con índices 0-based sobre el texto original.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars debe ser > 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars debe estar en [0, chunk_chars)")

    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_chars, n)
        if end < n:
            window = text[i:end]
            cut = window.rfind(" ")
            if cut != -1 and cut > chunk_chars - 300:  # evita recortes muy tempranos
                end = i + cut
        chunk = text[i:end].strip()
        start_idx = i
        end_idx = end
        if chunk:
            yield chunk, start_idx, end_idx
        if end == n:
            break
        i = max(0, end - overlap_chars)


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunk_ingest_jsonl(
    ingest_path: Path,
    out_path: Path,
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> int:
    """Lee ingest.jsonl y escribe chunks.jsonl. Devuelve #chunks generados."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in load_jsonl(ingest_path):
            content = (rec.get("content") or "").strip()
            meta = rec.get("metadata") or {}
            if not content:
                continue
            idx = 0
            for chunk, start, end in chunk_text(
                content, chunk_chars=chunk_chars, overlap_chars=overlap_chars
            ):
                count += 1
                chunk_meta = {
                    **meta,
                    "chunk_index": idx,
                    "start_char": start,
                    "end_char": end,
                    "chunk_chars": chunk_chars,
                    "overlap_chars": overlap_chars,
                }
                obj = {"content": chunk, "metadata": chunk_meta}
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                idx += 1
    return count


def main() -> None:
    from proyecto_cero.settings import Settings
    s = Settings.from_env()
    ingest_path = s.interim_dir / "ingest.jsonl"
    out_path = s.interim_dir / "chunks.jsonl"

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
        )

    if not ingest_path.exists():
        logger.error("No existe el archivo de ingesta: %s", ingest_path)
        return

    logger.info("Chunking desde %s", ingest_path)
    total = chunk_ingest_jsonl(
        ingest_path,
        out_path,
        chunk_chars=s.chunk_chars,
        overlap_chars=s.overlap_chars,
    )
    logger.info("Generados %d chunks -> %s", total, out_path)


if __name__ == "__main__":
    main()
