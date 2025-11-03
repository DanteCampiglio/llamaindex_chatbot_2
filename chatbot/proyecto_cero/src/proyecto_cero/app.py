import logging


class App:
    """Aplicación mínima con loop de consola.

    100% independiente del código existente del repo.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("proyecto_cero")
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
            )

    def run(self) -> None:
        print("=" * 50)
        print("Proyecto Cero - CLI Minimal")
        print("Escribe 'salir' para terminar.")
        print("=" * 50)
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

            # Por ahora, sólo eco. Aquí podremos enchufar LLM/RAG.
            self.logger.info("Echo: %s", texto)
            print(f"Echo: {texto}")

