"""
Gestión de retrievers (similarity, hybrid, BM25) parametrizada
"""
from typing import Optional
from loguru import logger
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever

# Asegúrate de importar tu objeto settings global
from config import settings  # Ajusta el import según tu estructura

class RetrieverFactory:
    """Factory para crear diferentes tipos de retrievers parametrizados"""
    
    SUPPORTED_MODES = ["similarity", "hybrid", "bm25"]
    
    @staticmethod
    def create_retriever(
        index: VectorStoreIndex,
        mode: Optional[str] = None,
        similarity_top_k: Optional[int] = None,
        bm25_k1: Optional[float] = None,   # Ya no se usará
        bm25_b: Optional[float] = None,    # Ya no se usará
        query_fusion_num_queries: Optional[int] = None,
        query_fusion_mode: Optional[str] = None,
        use_async: Optional[bool] = None,
    ):
        """
        Crea un retriever según el modo especificado y los parámetros de settings
        
        Args:
            index: Índice de vectores
            mode: Modo de retrieval ("similarity", "hybrid", "bm25")
            similarity_top_k: Número de documentos a recuperar
            bm25_k1, bm25_b: Hiperparámetros opcionales para BM25 (no usados en from_defaults)
            query_fusion_num_queries, query_fusion_mode: Parámetros para QueryFusionRetriever
            use_async: Si usar async en QueryFusionRetriever
            
        Returns:
            Retriever configurado
        """
        # Usa settings por defecto si no se pasan parámetros
        mode = mode or settings.RETRIEVER_MODE
        similarity_top_k = similarity_top_k or settings.SIMILARITY_TOP_K
        query_fusion_num_queries = query_fusion_num_queries if query_fusion_num_queries is not None else getattr(settings, "QUERY_FUSION_NUM_QUERIES", 1)
        query_fusion_mode = query_fusion_mode or getattr(settings, "QUERY_FUSION_MODE", "reciprocal_rerank")
        use_async = use_async if use_async is not None else getattr(settings, "QUERY_FUSION_USE_ASYNC", False)

        if mode not in RetrieverFactory.SUPPORTED_MODES:
            logger.warning(f"⚠️ Modo '{mode}' no soportado. Usando 'similarity'")
            mode = "similarity"
        
        if mode == "similarity":
            return RetrieverFactory._create_similarity_retriever(index, similarity_top_k)
        
        elif mode == "hybrid":
            return RetrieverFactory._create_hybrid_retriever(
                index, 
                similarity_top_k,
                query_fusion_num_queries,
                query_fusion_mode,
                use_async
            )
        
        elif mode == "bm25":
            return RetrieverFactory._create_bm25_retriever(index, similarity_top_k)
    
    @staticmethod
    def _create_similarity_retriever(index: VectorStoreIndex, top_k: int):
        """Crea retriever basado en similitud (embeddings)"""
        logger.info(f"🔍 Creando SIMILARITY retriever (top_k={top_k})")
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k
        )
    
    @staticmethod
    def _create_hybrid_retriever(
        index: VectorStoreIndex, 
        top_k: int,
        query_fusion_num_queries: int,
        query_fusion_mode: str,
        use_async: bool
    ):
        """Crea retriever híbrido (embeddings + BM25) parametrizado"""
        logger.info(f"🔍 Creando HYBRID retriever (top_k={top_k}, fusion_num_queries={query_fusion_num_queries}, fusion_mode={query_fusion_mode})")
        
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            
            # Vector retriever
            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k
            )
            
            # BM25 retriever SIN k1, b
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=index.docstore,
                similarity_top_k=top_k
            )
            
            # Fusión
            return QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=top_k,
                num_queries=query_fusion_num_queries,
                mode=query_fusion_mode,
                use_async=use_async
            )
            
        except ImportError:
            logger.error("❌ BM25Retriever no disponible. Instala: pip install llama-index-retrievers-bm25")
            logger.warning("   Fallback a similarity retriever")
            return RetrieverFactory._create_similarity_retriever(index, top_k)
        
        except Exception as e:
            logger.error(f"❌ Error creando hybrid retriever: {e}")
            logger.warning("   Fallback a similarity retriever")
            return RetrieverFactory._create_similarity_retriever(index, top_k)
    
    @staticmethod
    def _create_bm25_retriever(index: VectorStoreIndex, top_k: int):
        """Crea retriever basado en BM25 (keywords)"""
        logger.info(f"🔍 Creando BM25 retriever (top_k={top_k})")
        
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            
            return BM25Retriever.from_defaults(
                docstore=index.docstore,
                similarity_top_k=top_k
            )
            
        except ImportError:
            logger.error("❌ BM25Retriever no disponible. Instala: pip install llama-index-retrievers-bm25")
            logger.warning("   Fallback a similarity retriever")
            return RetrieverFactory._create_similarity_retriever(index, top_k)
        
        except Exception as e:
            logger.error(f"❌ Error creando BM25 retriever: {e}")
            logger.warning("   Fallback a similarity retriever")
            return RetrieverFactory._create_similarity_retriever(index, top_k)