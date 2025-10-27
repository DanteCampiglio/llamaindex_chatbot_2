from pathlib import Path
from pydantic_settings import BaseSettings
from typing import ClassVar, Dict, List

class Config(BaseSettings):
    """Configuración de la aplicación"""

    # ========== RUTAS BASE ==========
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"

    # ========== DIRECTORIOS DE DATOS ==========
    PDF_DIR: Path = DATA_DIR / "raw/pdfs"
    EMBEDDING_CACHE_DIR: Path = DATA_DIR / "embedding_cache"
    CHROMA_DB_PATH: Path = DATA_DIR / "chroma_db"

    # ========== CHROMADB ==========
    CHROMA_COLLECTION_NAME: str = "syngenta_docs"
    CHROMA_DISTANCE_FUNCTION: str = "cosine"

    # ========== MODELO LLAMA ==========
    LLAMA_MODEL_PATH: Path = MODELS_DIR / "llama32-3b" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    LLAMA_CONTEXT_SIZE: int = 8192
    LLAMA_MAX_TOKENS: int = 140
    LLAMA_TEMPERATURE: float = 0.7
    LLAMA_N_GPU_LAYERS: int = -1
    LLAMA_N_THREADS: int = 8
    LLAMA_N_BATCH: int = 512

    # ========== PARÁMETROS ANTI-LOOP ==========
    LLAMA_STOP_SEQUENCES: List[str] = [
        "¿Quieres", "\n\n\n", "PREGUNTA:", "CONTEXTO:", "###",
        "Usuario:", "Asistente:", "RESPUESTA:", "\n\nPREGUNTA"
    ]
    LLAMA_REPEAT_PENALTY: float = 1.5
    LLAMA_TOP_P: float = 0.85
    LLAMA_TOP_K: int = 30

    # ========== EMBEDDINGS ==========
    EMBEDDING_DIMENSIONS: int = 512
    EMBEDDING_MODEL: str = "llama"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # ========== CHUNKING ==========
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128

    # ========== RETRIEVAL ==========
    SIMILARITY_TOP_K: int = 5
    RESPONSE_MODE: str = "compact"
    RETRIEVER_MODE: str = "similarity"
   # BM25_K1: float = 1.2
   # BM25_B: float = 0.75
    QUERY_FUSION_NUM_QUERIES: int = 1
    QUERY_FUSION_MODE: str = "reciprocal_rerank"

    # ========== LIMPIEZA DE RESPUESTAS ==========
    RESPONSE_METADATA_KEYWORDS: List[str] = [
        'page_label:', 'file_path:', 'FICHA DE DATOS DE SEGURIDAD',
        'según el Reglamento', 'Versión', 'Fecha de revisión:',
        'Número SDS:', 'Fecha de la última expedición:',
        'Fecha de la primera expedición:'
    ]
    RESPONSE_TEXT_PREVIEW_LENGTH: int = 200
    RESPONSE_DEBUG_PREVIEW_LENGTH: int = 50

    # ========== PROMPTS ==========
    PROMPTS: ClassVar[Dict[str, str]] = {
        "qa_template": """
CONTEXTO: {context_str}

PREGUNTA: {query_str}

Si la respuesta está en el contexto, responde con precisión.
Si NO está en el contexto, di: "No encuentro esa información en los documentos"

RESPUESTA:""",
        "refine_template": """
Pregunta: {query_str}
Respuesta actual: {existing_answer}
Nuevo contexto: {context_msg}

Mejora la respuesta solo si el nuevo contexto aporta información relevante.
""",
        "system": "Responde solo con información del documento."
    }

    # ========== LOGGING ==========
    LOG_LEVEL: str = "INFO"

    # ========== LÍMITES DE CONTEXTO ==========
    MAX_CONTEXT_TOKENS: int = 6000
    RESERVED_TOKENS_FOR_RESPONSE: int = 2048

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self):
        """Crea todos los directorios necesarios"""
        for dir_path in [
            self.PDF_DIR,
            self.CHROMA_DB_PATH,
            self.EMBEDDING_CACHE_DIR,
            self.LOG_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def llama_model_kwargs(self) -> Dict:
        """Genera model_kwargs para LlamaCPP dinámicamente"""
        return {
            "n_gpu_layers": self.LLAMA_N_GPU_LAYERS,
            "n_threads": self.LLAMA_N_THREADS,
            "n_batch": self.LLAMA_N_BATCH,
            "n_ctx": self.LLAMA_CONTEXT_SIZE,
            "stop": self.LLAMA_STOP_SEQUENCES,
            "repeat_penalty": self.LLAMA_REPEAT_PENALTY,
            "top_p": self.LLAMA_TOP_P,
            "top_k": self.LLAMA_TOP_K,
        }

    @property
    def response_cleaning(self) -> Dict:
        """Configuración de limpieza de respuestas"""
        return {
            "metadata_keywords": self.RESPONSE_METADATA_KEYWORDS,
            "text_preview_length": self.RESPONSE_TEXT_PREVIEW_LENGTH,
            "debug_preview_length": self.RESPONSE_DEBUG_PREVIEW_LENGTH
        }

    def validate_context_size(self) -> bool:
        """Valida coherencia de configuración de contexto"""
        estimated_context = (self.CHUNK_SIZE * self.SIMILARITY_TOP_K) + 1000
        available = self.LLAMA_CONTEXT_SIZE - self.LLAMA_MAX_TOKENS - 500

        if estimated_context > available:
            from loguru import logger
            logger.warning("⚠️ Configuración de contexto puede causar overflow:")
            logger.warning(f"   Estimado: {estimated_context} tokens")
            logger.warning(f"   Disponible: {available} tokens")
            logger.warning("   Recomendación: Reducir CHUNK_SIZE o SIMILARITY_TOP_K")
            return False
        return True

# Instancia global
settings = Config()


def setup_llama_index():
    """Configura LLM de LlamaIndex con parámetros anti-loop"""
    from llama_index.core import Settings
    from llama_index.llms.llama_cpp import LlamaCPP
    from loguru import logger

    if not settings.validate_context_size():
        logger.warning("⚠️ Continuando con configuración que puede causar problemas...")

    llm = LlamaCPP(
        model_path=str(settings.LLAMA_MODEL_PATH),
        temperature=settings.LLAMA_TEMPERATURE,
        max_new_tokens=settings.LLAMA_MAX_TOKENS,
        context_window=settings.LLAMA_CONTEXT_SIZE,
        model_kwargs=settings.llama_model_kwargs,  # ✅ Uso de property
        verbose=False,
    )

    logger.info("🤖 LLM Configurado:")
    logger.info(f"   ├─ Model: {settings.LLAMA_MODEL_PATH.name}")
    logger.info(f"   ├─ Context Window: {settings.LLAMA_CONTEXT_SIZE}")
    logger.info(f"   ├─ Max Tokens: {settings.LLAMA_MAX_TOKENS}")
    logger.info(f"   ├─ Temperature: {settings.LLAMA_TEMPERATURE}")
    logger.info(f"   ├─ GPU Layers: {settings.LLAMA_N_GPU_LAYERS}")
    logger.info(f"   ├─ Threads: {settings.LLAMA_N_THREADS}")
    logger.info(f"   ├─ Repeat Penalty: {settings.LLAMA_REPEAT_PENALTY}")
    logger.info(f"   └─ Stop Sequences: {len(settings.LLAMA_STOP_SEQUENCES)} configuradas")

    Settings.llm = llm
    Settings.chunk_size = settings.CHUNK_SIZE
    Settings.chunk_overlap = settings.CHUNK_OVERLAP

    return llm, None, None