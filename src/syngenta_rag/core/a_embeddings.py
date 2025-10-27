"""
embeddings.py - Gestión de embeddings para Syngenta RAG
"""
from pathlib import Path
from typing import List, Any, Optional
import logging
import sys
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """Embedding usando Sentence-Transformers"""
    
    _model: Any = PrivateAttr()
    _model_name: str = PrivateAttr()
    _dimensions: int = PrivateAttr()
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: Any):
        super().__init__(**kwargs)
        self._model_name = model_name
        
        logger.info(f"🔄 Cargando modelo: {model_name}")
        
        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        
        logger.info(f"✅ Modelo cargado ({self._dimensions} dimensiones)")
    
    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"
    
    def _get_embedding(self, text: str) -> List[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    def get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


class EmbeddingManager:
    """Gestiona embeddings usando Sentence-Transformers"""
    
    MODELS = {
        "fast": "all-MiniLM-L6-v2",
        "balanced": "all-mpnet-base-v2",
        "multilingual": "paraphrase-multilingual-mpnet-base-v2",
    }
    
    def __init__(
        self, 
        model_type: str = "multilingual", 
        cache_folder: Optional[str] = None
    ):
        # ✅ FIX: Importación dinámica con manejo de rutas
        try:
            from config.settings import settings
            default_cache = settings.EMBEDDING_CACHE_DIR
        except ModuleNotFoundError:
            # ✅ Fallback: Calcular ruta manualmente
            logger.warning("⚠️ No se encontró config.settings, usando ruta por defecto")
            project_root = Path(__file__).parent.parent.parent.parent
            default_cache = project_root / "data" / "embedding_cache"
        
        self.model_type = model_type
        self.cache_folder = Path(cache_folder or default_cache)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        
        # Obtener nombre completo del modelo
        if model_type in self.MODELS:
            self.model_name = f"sentence-transformers/{self.MODELS[model_type]}"
        else:
            self.model_name = model_type
        
        self.embed_model = None
        
        logger.info(f"🎯 EmbeddingManager inicializado")
        logger.info(f"📦 Modelo: {self.model_name}")
        logger.info(f"📁 Cache: {self.cache_folder}")
    
    def get_embedding_model(self) -> BaseEmbedding:
        if self.embed_model is None:
            logger.info(f"🔄 Creando modelo de embeddings...")
            self.embed_model = SentenceTransformerEmbedding(
                model_name=self.model_name
            )
        
        return self.embed_model
    
    def get_dimension(self) -> int:
        model = self.get_embedding_model()
        return model._dimensions
    
    def get_model_info(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "cache_folder": str(self.cache_folder),
            "dimension": self.get_dimension() if self.embed_model else "N/A",
            "loaded": self.embed_model is not None,
        }
    
    @classmethod
    def list_models(cls) -> dict:
        return cls.MODELS


# ========================================================================
# 🧪 TESTING
# ========================================================================

def test_embeddings():
    """Prueba los embeddings"""
    print("=" * 70)
    print("🧪 TEST DE EMBEDDINGS")
    print("=" * 70)
    
    print("\n📋 Modelos disponibles:")
    for key, name in EmbeddingManager.list_models().items():
        print(f"  - {key}: {name}")
    
    print("\n" + "=" * 70)
    print("🔧 Probando modelo: multilingual")
    print("=" * 70)
    
    # ✅ FIX: No requiere config.settings para test
    manager = EmbeddingManager(
        model_type="multilingual",
        cache_folder="./test_cache"  # ✅ Cache temporal para test
    )
    embed_model = manager.get_embedding_model()
    
    test_text = "¿Cuál es un buen consejo de prudencia para ACELEPRYN?"
    
    print(f"\n🔄 Generando embedding...")
    embedding = embed_model.get_text_embedding(test_text)
    
    print(f"\n✅ Embedding generado:")
    print(f"   - Texto: '{test_text}'")
    print(f"   - Dimensión: {len(embedding)}")
    print(f"   - Primeros 5 valores: {[f'{v:.4f}' for v in embedding[:5]]}")
    
    norma = sum(x*x for x in embedding) ** 0.5
    print(f"   - Norma L2: {norma:.4f}")
    
    print(f"\n📊 Info del modelo:")
    for key, value in manager.get_model_info().items():
        print(f"   - {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    test_embeddings()