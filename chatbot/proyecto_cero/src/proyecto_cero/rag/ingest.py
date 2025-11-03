from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF
from proyecto_cero.settings import Settings


logger = logging.getLogger("proyecto_cero.rag.ingest")


@dataclass
class IngestRecord:
    """Unidad mínima de ingesta: texto + metadatos.

    - content: texto plano (por página en este paso)
    - metadata: incluye al menos: source (ruta), filename, page (1-based), n_pages, mtime
    """

    content: str
    metadata: dict


def extract_pdf_pages(pdf_path: Path) -> List[IngestRecord]:
    """Extrae texto por página desde un PDF, sin OCR.

    Se omiten páginas sin texto (vacías o con solo espacios).
    """
    records: List[IngestRecord] = []
    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count
        for i in range(n_pages):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue
            records.append(
                IngestRecord(
                    content=text,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": i + 1,
                        "n_pages": n_pages,
                        "mtime": int(pdf_path.stat().st_mtime),
                    },
                )
            )
    return records


def iter_pdfs(raw_dir: Path) -> Iterable[Path]:
    """Itera rutas de PDFs dentro de un directorio (no recursivo)."""
    yield from sorted(raw_dir.glob("*.pdf"))


def ingest_directory(raw_dir: Path) -> List[IngestRecord]:
    """Ingesta todos los PDFs del directorio dado, devolviendo una lista de registros."""
    all_records: List[IngestRecord] = []
    for pdf_path in iter_pdfs(raw_dir):
        logger.info("Extrayendo: %s", pdf_path)
        try:
            records = extract_pdf_pages(pdf_path)
            all_records.extend(records)
            logger.info(
                "Listo %s: %d páginas con texto", pdf_path.name, len(records)
            )
        except Exception as exc:  # pragma: no cover (defensivo)
            logger.exception("Error extrayendo %s: %s", pdf_path, exc)
    return all_records


def save_jsonl(records: List[IngestRecord], out_path: Path) -> None:
    """Guarda los registros en JSONL (una línea por registro)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            obj = {"content": rec.content, "metadata": rec.metadata}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    s = Settings.from_env()
    raw_dir = s.raw_dir
    out_path = s.interim_dir / "ingest.jsonl"

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
        )

    if not raw_dir.exists():
        logger.error("No existe el directorio de entrada: %s", raw_dir)
        return

    logger.info("Iniciando ingesta desde: %s", raw_dir)
    records = ingest_directory(raw_dir)
    logger.info("Registros totales con texto: %d", len(records))

    save_jsonl(records, out_path)
    logger.info("Guardado JSONL: %s", out_path)


if __name__ == "__main__":  # Permite ejecutar como script si se desea
    main()
