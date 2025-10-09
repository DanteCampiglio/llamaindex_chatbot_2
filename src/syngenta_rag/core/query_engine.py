"""
quick_test.py - Test rápido del sistema RAG
"""
import sys
from pathlib import Path
from loguru import logger

# Configurar logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

from config.settings import settings, setup_llama_index
from src.syngenta_rag.core import (
    EmbeddingManager,
    IndexManager,
    RetrieverFactory,
    PromptManager,
    ResponseBuilder
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

def test_rag_pipeline():
    """Test completo del pipeline RAG"""
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO TEST DEL SISTEMA RAG")
    logger.info("=" * 80)
    
    try:
        # 1. Setup LLM
        logger.info("\n📦 PASO 1: Configurando LLM...")
        llm, _, _ = setup_llama_index()
        logger.info("✅ LLM configurado")
        
        # 2. Embeddings
        logger.info("\n📦 PASO 2: Inicializando Embeddings...")
        embed_manager = EmbeddingManager()
        embed_model = embed_manager.get_embed_model()
        logger.info("✅ Embeddings listos")
        
        # 3. Index Manager
        logger.info("\n📦 PASO 3: Cargando/Creando índice...")
        index_manager = IndexManager(
            persist_dir=settings.CHROMA_DB_PATH,
            embed_model=embed_model
        )
        
        # Verificar si existe índice
        if index_manager.index_exists():
            logger.info("📂 Cargando índice existente...")
            index = index_manager.load_index()
        else:
            logger.info("🆕 Creando nuevo índice desde PDFs...")
            pdf_files = list(settings.PDF_DIR.glob("*.pdf"))
            
            if not pdf_files:
                logger.error(f"❌ No se encontraron PDFs en: {settings.PDF_DIR}")
                logger.info(f"💡 Copia archivos PDF a: {settings.PDF_DIR.absolute()}")
                return
            
            logger.info(f"📄 Encontrados {len(pdf_files)} PDFs")
            index = index_manager.create_index_from_pdfs(pdf_files)
        
        logger.info("✅ Índice listo")
        
        # 4. Retriever
        logger.info("\n📦 PASO 4: Creando Retriever...")
        retriever = RetrieverFactory.create_retriever(
            index=index,
            mode=settings.RETRIEVER_MODE,
            similarity_top_k=settings.SIMILARITY_TOP_K
        )
        logger.info("✅ Retriever creado")
        
        # 5. Prompts
        logger.info("\n📦 PASO 5: Configurando Prompts...")
        prompt_manager = PromptManager()
        qa_prompt = prompt_manager.get_qa_prompt()
        logger.info("✅ Prompts configurados")
        
        # 6. Response Synthesizer
        logger.info("\n📦 PASO 6: Creando Response Synthesizer...")
        response_synthesizer = get_response_synthesizer(
            response_mode=settings.RESPONSE_MODE,
            text_qa_template=qa_prompt,
            use_async=False,
            streaming=False
        )
        logger.info("✅ Response Synthesizer listo")
        
        # 7. Query Engine
        logger.info("\n📦 PASO 7: Creando Query Engine...")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,  # ✅ SOLO UNA VEZ
            response_synthesizer=response_synthesizer
        )
        logger.info("✅ Query Engine creado")
        
        # 8. TEST QUERY
        logger.info("\n" + "=" * 80)
        logger.info("🧪 EJECUTANDO QUERY DE PRUEBA")
        logger.info("=" * 80)
        
        test_query = "¿Qué es Syngenta?"
        logger.info(f"\n❓ Pregunta: {test_query}")
        
        response = query_engine.query(test_query)
        
        logger.info("\n" + "=" * 80)
        logger.info("📋 RESPUESTA:")
        logger.info("=" * 80)
        logger.info(f"\n{response.response}\n")
        
        # Mostrar fuentes
        if hasattr(response, 'source_nodes') and response.source_nodes:
            logger.info("=" * 80)
            logger.info("📚 FUENTES UTILIZADAS:")
            logger.info("=" * 80)
            for i, node in enumerate(response.source_nodes, 1):
                score = node.score if hasattr(node, 'score') else 'N/A'
                logger.info(f"\n[{i}] Score: {score}")
                logger.info(f"Texto: {node.text[:200]}...")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Error en test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    test_rag_pipeline()