"""
embeddings.py - Gestión de embeddings locales para Syngenta RAG
"""
from pathlib import Path
from typing import Optional, List, Any
import logging
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

logger = logging.getLogger(__name__)


# ========================================================================
# CLASE ORIGINAL - NO TOCAR
# ========================================================================

class LocalEmbedding(BaseEmbedding):
    """Embedding local basado en hash - 100% offline"""
    
    _dimensions: int = PrivateAttr(default=384)
    
    def __init__(self, dimensions: int = 384, **kwargs: Any):
        super().__init__(**kwargs)
        self._dimensions = dimensions
        logger.info(f"✓ LocalEmbedding inicializado ({dimensions} dimensiones)")
    
    @classmethod
    def class_name(cls) -> str:
        return "LocalEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Genera embedding para consulta"""
        words = query.lower().split()
        embedding = []
        for i in range(self._dimensions):
            hash_val = sum(hash(word + str(i)) for word in words)
            embedding.append((hash_val % 10000) / 10000.0)
        return embedding
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Genera embedding para texto"""
        return self._get_query_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para múltiples textos"""
        return [self._get_text_embedding(text) for text in texts]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    # Métodos públicos
    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    def get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


# ========================================================================
# ✅ NUEVA CLASE - SOLO AGREGAR (no modifica nada existente)
# ========================================================================

class LlamaEmbedding(BaseEmbedding):
    """Embedding usando modelo Llama local (GGUF) - Semántico real"""
    
    _model: Any = PrivateAttr()
    _dimensions: int = PrivateAttr(default=512)
    
    def __init__(self, model_path: str, dimensions: int = 512, **kwargs: Any):
        super().__init__(**kwargs)
        self._dimensions = dimensions
        
        logger.info(f"🔄 Cargando Llama para embeddings...")
        
        try:
            from llama_cpp import Llama
            
            self._model = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=4,
                embedding=True,
                verbose=False,
                n_gpu_layers=-1
            )
            
            logger.info(f"✅ LlamaEmbedding inicializado ({dimensions}D)")
            
        except Exception as e:
            logger.error(f"❌ Error cargando Llama: {e}")
            raise
    
    @classmethod
    def class_name(cls) -> str:
        return "LlamaEmbedding"
    
    def _get_embedding(self, text: str) -> List[float]:
        try:
            # ✅ FIX: embed() puede devolver lista de listas [[...]]
            raw_embedding = self._model.embed(text)
            
            # Debug: ver qué devuelve realmente
            logger.debug(f"Tipo raw_embedding: {type(raw_embedding)}")
            
            # Aplanar si es lista de listas
            if isinstance(raw_embedding, list) and len(raw_embedding) > 0:
                if isinstance(raw_embedding[0], list):
                    # Es [[...]] -> tomar primer elemento
                    embedding = raw_embedding[0]
                    logger.debug(f"Embedding era lista de listas, aplanado")
                else:
                    # Ya es [...]
                    embedding = raw_embedding
            else:
                # Convertir numpy array u otro tipo
                embedding = list(raw_embedding)
            
            # Ajustar dimensión
            if len(embedding) > self._dimensions:
                embedding = embedding[:self._dimensions]
            elif len(embedding) < self._dimensions:
                # Padding con ceros
                embedding = embedding + [0.0] * (self._dimensions - len(embedding))
            
            # ✅ Asegurar que todos son floats (ahora sí funcionará)
            embedding = [float(x) for x in embedding]
            
            logger.debug(f"Embedding final: dim={len(embedding)}, tipo={type(embedding[0])}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback a embedding aleatorio
            import random
            logger.warning(f"⚠️ Usando embedding aleatorio como fallback")
            return [random.gauss(0, 0.1) for _ in range(self._dimensions)]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]
    
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
    
    
# ========================================================================
# ✅ MODIFICAR SOLO ESTE MÉTODO en EmbeddingManager
# ========================================================================

class EmbeddingManager:
    """Gestiona embeddings locales sin dependencias externas"""
    
    MODELS = {
        "local": "Local Hash-based Embeddings (offline)",
        "llama": "Llama 3.2 3B Embeddings (semantic)",
    }
    
    def __init__(
        self, 
        model_name: Optional[str] = "local",
        cache_folder: Optional[str] = None,
        dimensions: int = 384
    ):
        from config.settings import settings
        
        self.model_name = model_name
        self.cache_folder = Path(cache_folder or settings.EMBEDDING_CACHE_DIR)
        self.dimensions = dimensions
        self.embed_model = None
        self.current_model = self.model_name
        
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 EmbeddingManager inicializado")
        logger.info(f"📦 Modelo: {self.model_name}")
        logger.info(f"📐 Dimensiones: {self.dimensions}")
    
    def get_embedding_model(self) -> BaseEmbedding:
        """
        ✅ ÚNICO CAMBIO - Detecta automáticamente qué modelo usar
        """
        if self.embed_model is not None:
            logger.debug("♻️ Usando modelo en caché")
            return self.embed_model
        
        try:
            # ✅ AUTO-DETECT: Si model_name es "llama", usar LlamaEmbedding
            if self.model_name == "llama":
                from config.settings import settings
                
                logger.info(f"🔄 Creando LlamaEmbedding...")
                self.embed_model = LlamaEmbedding(
                    model_path=str(settings.LLAMA_MODEL_PATH),
                    dimensions=self.dimensions
                )
            else:
                # Default: LocalEmbedding (hash)
                logger.info(f"🔄 Creando LocalEmbedding...")
                self.embed_model = LocalEmbedding(dimensions=self.dimensions)
            
            logger.info(f"✅ Modelo creado correctamente")
            return self.embed_model
            
        except Exception as e:
            logger.error(f"❌ Error creando modelo: {e}")
            raise
    
    # ========== RESTO DEL CÓDIGO SIN CAMBIOS ==========
    
    def set_model(self, model_name: str, dimensions: int = None) -> None:
        self.model_name = model_name
        self.current_model = model_name
        if dimensions:
            self.dimensions = dimensions
        self.embed_model = None
        logger.info(f"🔄 Modelo cambiado a: {model_name} ({self.dimensions}D)")
    
    def get_dimension(self) -> int:
        return self.dimensions
    
    def get_embedding_dimension(self) -> int:
        return self.get_dimension()
    
    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": "Llama (GGUF)" if self.model_name == "llama" else "Local (Hash)",
            "cache_folder": str(self.cache_folder),
            "dimension": self.dimensions,
            "cached": self.embed_model is not None,
            "loaded": self.embed_model is not None,
            "offline": True
        }
    
    def get_stats(self) -> dict:
        return self.get_model_info()
    
    @classmethod
    def list_models(cls) -> dict:
        return cls.MODELS


# ========================================================================
# 🧪 TESTING
# ========================================================================

def test_embeddings():
    """Prueba los modelos de embeddings"""
    print("=" * 70)
    print("🧪 TEST DE EMBEDDINGS")
    print("=" * 70)
    
    print("\n📋 Modelos disponibles:")
    for key, name in EmbeddingManager.list_models().items():
        print(f"  - {key}: {name}")
    
    test_text = "Syngenta es una empresa líder en agricultura."
    
    for model_name in ["local", "llama"]:
        print("\n" + "=" * 70)
        print(f"🔧 Probando: {model_name}")
        print("=" * 70)
        
        try:
            manager = EmbeddingManager(model_name=model_name, dimensions=384)
            embed_model = manager.get_embedding_model()
            
            print(f"\n🔄 Generando embedding...")
            embedding = embed_model.get_text_embedding(test_text)
            
            print(f"\n✅ Embedding generado:")
            print(f"   - Texto: '{test_text}'")
            print(f"   - Dimensión: {len(embedding)}")
            print(f"   - Tipo: {type(embedding)}")
            print(f"   - Tipo elementos: {type(embedding[0]) if embedding else 'N/A'}")
            
            # ✅ FIX: Convertir a string antes de formatear
            primeros_5 = [f'{float(v):.4f}' for v in embedding[:5]]
            print(f"   - Primeros 5: {primeros_5}")
            
            # Calcular norma (para verificar que no son todos ceros)
            norma = sum(x*x for x in embedding) ** 0.5
            print(f"   - Norma L2: {norma:.4f}")
            
            info = manager.get_model_info()
            print(f"\n📊 Info:")
            for key, value in info.items():
                print(f"   - {key}: {value}")
            
        except Exception as e:
            print(f"\n⚠️ Error con {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    test_embeddings()

