"""
quick_test.py - Test completo del flujo RAG con Llama local
"""
import sys
from pathlib import Path

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings, setup_llama_index
from src.syngenta_rag.core import (
    IndexManager,
    RetrieverFactory,
    PromptManager,
    ResponseBuilder
)

def query_with_sources(query_engine, question: str) -> dict:
    """
    Ejecuta consulta y retorna respuesta estructurada
    
    Args:
        query_engine: Query engine de LlamaIndex
        question: Pregunta del usuario
        
    Returns:
        Dict con response, sources y metadata
    """
    from loguru import logger
    
    logger.info(f"🔍 Consultando: {question}")
    
    try:
        # Ejecutar query
        response = query_engine.query(question)
        
        # Limpiar respuesta

        rb = ResponseBuilder()
        clean_text = rb.clean_response(str(response))
        # Extraer fuentes
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            sources = ResponseBuilder.extract_sources(response.source_nodes)
        
        # Construir respuesta
        return ResponseBuilder.build_response_dict(
            response_text=clean_text,
            sources=sources,
            metadata={
                "top_k": settings.SIMILARITY_TOP_K,
                "retriever_mode": getattr(settings, 'RETRIEVER_MODE', 'similarity'),
                "response_mode": settings.RESPONSE_MODE
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error en consulta: {e}")
        return {
            "response": f"Error al procesar la consulta: {str(e)}",
            "sources": [],
            "metadata": {"error": str(e)}
        }

def main():
    print("=" * 70)
    print("🧪 TEST COMPLETO DEL FLUJO RAG DE SYNGENTA")
    print("=" * 70)
    
    # ========== 1. VERIFICAR ÍNDICE ==========
    print("\n📂 Verificando índice...")
    if not settings.CHROMA_DB_PATH.exists():
        print(f"❌ No existe índice en: {settings.CHROMA_DB_PATH}")
        print("   Ejecuta primero: python main.py --reindex")
        return
    
    print(f"✅ Índice encontrado: {settings.CHROMA_DB_PATH}")
    
    # ========== 2. VERIFICAR MODELO LLAMA ==========
    print("\n🔍 Verificando modelo Llama...")
    if not settings.LLAMA_MODEL_PATH.exists():
        print(f"❌ Modelo no encontrado: {settings.LLAMA_MODEL_PATH}")
        print("   Descarga el modelo GGUF y colócalo en models/llama32-3b/")
        return
    
    print(f"✅ Modelo encontrado: {settings.LLAMA_MODEL_PATH.name}")
    
    # ========== 3. CONFIGURAR LLM ==========
    print("\n⚙️ Configurando LLM...")
    try:
        llm, _, _ = setup_llama_index()
        print(f"✅ LLM: Llama 3.2 3B (GGUF)")
        print(f"   - Context: {settings.LLAMA_CONTEXT_SIZE} tokens")
        print(f"   - Max tokens: {settings.LLAMA_MAX_TOKENS}")
        print(f"   - Temperature: {settings.LLAMA_TEMPERATURE}")
        print(f"   - GPU layers: {settings.LLAMA_N_GPU_LAYERS}")
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        return
    
    # ========== 4. CARGAR ÍNDICE ==========
    print("\n📦 Cargando índice con IndexManager...")
    try:
        index_manager = IndexManager()
        index = index_manager.load_index()
        
        if index is None:
            print("❌ No se pudo cargar el índice")
            print("   Ejecuta: python main.py --reindex")
            return
        
        print("✅ Índice cargado correctamente")
        
    except Exception as e:
        print(f"❌ Error cargando índice: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        return
    
    # ========== 5. ESTADÍSTICAS ==========
    print("\n📊 Estadísticas del índice:")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        collection = client.get_collection("syngenta_docs")
        count = collection.count()
        print(f"   - Chunks indexados: {count}")
        print(f"   - Colección: syngenta_docs")
        print(f"   - Dimensión embeddings: {settings.EMBEDDING_DIMENSIONS}")
    except Exception as e:
        print(f"⚠️ No se pudieron obtener estadísticas: {e}")
    
    # ========== 6. CREAR QUERY ENGINE ==========
    print("\n🤖 Creando QueryEngine...")
    try:
        # Crear retriever
        retriever = RetrieverFactory.create_retriever(
            index=index,
            mode=getattr(settings, 'RETRIEVER_MODE', 'similarity'),
            similarity_top_k=settings.SIMILARITY_TOP_K
        )
        
        # Crear prompt manager
        prompt_manager = PromptManager()
        
        # ✅ CORRECCIÓN: Usar from_args con response_synthesizer
        from llama_index.core.response_synthesizers import get_response_synthesizer
        from llama_index.core.query_engine import RetrieverQueryEngine
        
        # Crear response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=settings.RESPONSE_MODE,
            text_qa_template=prompt_manager.get_qa_prompt(),
            refine_template=prompt_manager.get_refine_prompt(),
            use_async=False,
            streaming=False
        )
        
        # Crear query engine SIN DUPLICAR retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,  # ✅ SOLO UNA VEZ
            response_synthesizer=response_synthesizer
        )
        
        print(f"✅ QueryEngine listo")
        print(f"   - Retriever: {getattr(settings, 'RETRIEVER_MODE', 'similarity')}")
        print(f"   - Top K: {settings.SIMILARITY_TOP_K}")
        print(f"   - Response mode: {settings.RESPONSE_MODE}")
        
    except Exception as e:
        print(f"❌ Error creando QueryEngine: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        return
    
    # ========== 7. CONSULTA DE PRUEBA ==========
    test_query = "¿Que hacer en caso de incendio con AMISTAR?"
    print(f"\n💬 Consulta de prueba: '{test_query}'")
    print("⏳ Procesando (puede tardar con Llama local)...")
    
    try:
        result = query_with_sources(query_engine, test_query)
        
        response = result.get('response', 'Sin respuesta')
        sources = result.get('sources', [])
        metadata = result.get('metadata', {})
        
        print("\n" + "=" * 70)
        print("📝 RESPUESTA:")
        print("=" * 70)
        print(response)
        
        if metadata:
            print(f"\n📊 Metadata:")
            print(f"   - Fuentes recuperadas: {metadata.get('num_sources', 0)}")
            print(f"   - Top K: {metadata.get('top_k', 'N/A')}")
            print(f"   - Modo respuesta: {metadata.get('response_mode', 'N/A')}")
        
        if sources:
            print("\n📚 Fuentes utilizadas:")
            for i, source in enumerate(sources[:3], 1):
                score = source.get('score', 0.0)
                text = source.get('text', '')[:150]
                print(f"   {i}. Score: {score:.3f}")
                print(f"      {text}...")
        else:
            print("\n⚠️ No se encontraron fuentes")
            
    except Exception as e:
        print(f"❌ Error en consulta: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
    
    # ========== 8. MODO INTERACTIVO ==========
    if "--interactive" in sys.argv:
        print("\n" + "=" * 70)
        print("💬 MODO INTERACTIVO")
        print("=" * 70)
        print("Comandos:")
        print("  - Escribe tu pregunta y presiona Enter")
        print("  - 'salir' / 'exit' / 'quit' para terminar")
        print("  - '--sources' para toggle mostrar fuentes")
        print("  - '--stats' para ver estadísticas")
        print("=" * 70)
        
        show_sources = "--sources" in sys.argv
        
        while True:
            try:
                user_query = input("\n🔍 Tu pregunta: ").strip()
                
                # Comandos especiales
                if user_query.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if user_query.lower() == '--stats':
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
                        collection = client.get_collection("syngenta_docs")
                        count = collection.count()
                        print(f"\n📊 Estadísticas:")
                        print(f"   - Chunks indexados: {count}")
                        print(f"   - Colección: syngenta_docs")
                        print(f"   - Base de datos: {settings.CHROMA_DB_PATH}")
                    except Exception as e:
                        print(f"❌ Error obteniendo stats: {e}")
                    continue
                
                if user_query.lower() == '--sources':
                    show_sources = not show_sources
                    print(f"   Mostrar fuentes: {'✅ ON' if show_sources else '❌ OFF'}")
                    continue
                
                # Ignorar vacío
                if not user_query:
                    continue
                
                # Procesar consulta
                print("⏳ Procesando...")
                result = query_with_sources(query_engine, user_query)
                
                # Extraer valores
                response = result.get('response', 'Sin respuesta')
                sources = result.get('sources', [])
                
                # Mostrar respuesta
                print("\n📝 Respuesta:")
                print("-" * 70)
                print(response)
                
                # Mostrar fuentes si está activado
                if show_sources and sources:
                    print("\n📚 Fuentes:")
                    for i, source in enumerate(sources[:3], 1):
                        score = source.get('score', 0.0)
                        text = source.get('text', '')[:150]
                        print(f"   {i}. Score: {score:.3f}")
                        print(f"      {text}...")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
    
    # ========== 9. CLEANUP ==========
    print("\n🧹 Limpiando recursos...")
    try:
        if hasattr(index_manager, 'close'):
            index_manager.close()
        print("✅ Test completado")
    except Exception as e:
        print(f"⚠️ Error en cleanup: {e}")

if __name__ == "__main__":
    main()