"""
main.py - Sistema RAG de Syngenta
"""
import os
os.environ["POSTHOG_DISABLED"] = "1"

from config.settings import settings
from src.syngenta_rag.core.index_manager import IndexManager
from src.syngenta_rag.core.query_engine import QueryEngine


def main():
    print("=" * 60)
    print("🌱 SYNGENTA RAG - Sistema de Consultas")
    print("=" * 60)
    
    # 1. Inicializar IndexManager
    print("\n📂 Inicializando sistema de indexación...")
    index_manager = IndexManager()
    
    # 2. Indexar documentos
    print(f"\n📥 Indexando documentos desde: {settings.PDF_DIR}")  # ← CAMBIO
    index, msg = index_manager.load_and_index_documents(
        pdf_directory=settings.PDF_DIR,  # ← CAMBIO
        force_reindex=False
    )
    
    if index is None:
        print(f"❌ {msg}")
        return
    
    print(f"✅ {msg}")
    
    # 3. Inicializar QueryEngine
    print("\n🔍 Inicializando motor de consultas...")
    query_engine = QueryEngine(index)
    
    # 4. Loop de consultas
    print("\n" + "=" * 60)
    print("💬 Sistema listo. Escribe 'salir' para terminar.")
    print("=" * 60)
    
    while True:
        query = input("\n🌱 Pregunta: ").strip()
        
        if query.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!")
            break
            
        if not query:
            continue
            
        print("\n🔎 Buscando respuesta...")
        response = query_engine.query(query)
        
        print("\n" + "=" * 60)
        print("💡 RESPUESTA:")
        print("=" * 60)
        print(response.response)
        
        print("\n" + "=" * 60)
        print("📚 FUENTES:")
        print("=" * 60)
        for i, node in enumerate(response.source_nodes, 1):
            metadata = node.node.metadata
            score = node.score if hasattr(node, 'score') else 'N/A'
            print(f"\n{i}. {metadata.get('file_name', 'Unknown')} - "
                  f"Página {metadata.get('page_label', 'N/A')} "
                  f"(Score: {score})")
            print(f"   {node.node.get_content()[:200]}...")


if __name__ == "__main__":
    main()