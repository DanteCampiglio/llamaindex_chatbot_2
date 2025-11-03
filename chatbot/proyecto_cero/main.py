import sys
from pathlib import Path

# Añade ./proyecto_cero/src al sys.path para imports locales
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from proyecto_cero.rag.retrieve import search as rag_search  # noqa: E402
from proyecto_cero.rag.generate import answer_with_llm  # noqa: E402
from proyecto_cero.settings import Settings  # noqa: E402


def main() -> None:
    print("=" * 50)
    print("Proyecto Cero - RAG CLI")
    print("Escribe 'salir' para terminar.")
    print("=" * 50)

    s = Settings.from_env()
    persist_dir = s.index_dir
    if not persist_dir.exists():
        print(
            f"[!] No existe el índice en '{persist_dir}'.\n"
            "    Corre antes: ingest -> chunk -> index"
        )
        return

    while True:
        try:
            texto = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if texto.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego!")
            break
        if not texto:
            continue

        try:
            res = rag_search(persist_dir, texto, top_k=s.top_k)
        except Exception as exc:
            print(f"[!] Error al consultar el índice: {exc}")
            continue

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not docs:
            print("Sin resultados relevantes.")
            continue

        # Solo mostrar la respuesta sintetizada a partir de los chunks recuperados
        try:
            answer, provider = answer_with_llm(texto, docs[:6], metas[:6], settings=s)
            print(answer + "\n")
        except Exception as e:
            print("[i] No se pudo sintetizar con LLM (" + str(e) + ").")


if __name__ == "__main__":
    main()

