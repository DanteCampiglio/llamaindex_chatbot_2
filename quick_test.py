#!/usr/bin/env python3
"""
⚡ Test rápido de SynGPT
"""

import sys
from pathlib import Path

# 🔥 Agregar path padre
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# ✅ IMPORTS CORRECTOS
from config.settings import setup_llama_index, settings
from src.syngenta_rag.core.query_engine import QueryEngine

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

def quick_test(question: str = None):
    if not question:
        question = sys.argv[1] if len(sys.argv) > 1 else "¿Cual es la capital de Escocia?"
    
    print(f"🧪 TEST: {question}")
    print("=" * 40)
    
    try:
        # 🔍 PASO 1: Setup
        print("📝 1. Configurando LlamaIndex...")
        llm, embed_model, vector_store = setup_llama_index()
        print("✅ Setup completado")
        
        # 🔍 PASO 2: Verificar directorio de índice
        print(f"📝 2. Verificando índice en: {settings.INDEX_DIR}")
        if not settings.INDEX_DIR.exists():
            print(f"❌ Directorio no existe: {settings.INDEX_DIR}")
            return
        
        # 🔍 PASO 3: Cargar StorageContext CON vector_store
        print("📝 3. Cargando StorageContext con vector_store...")
        storage_context = StorageContext.from_defaults(
            persist_dir=str(settings.INDEX_DIR),
            vector_store=vector_store  # 🔥 AGREGAR VECTOR STORE
        )
        print("✅ StorageContext cargado con vector store")
        
        # 🔍 PASO 4: Cargar índice
        print("📝 4. Cargando índice...")
        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model  # 🔥 AGREGAR EMBED MODEL
        )
        print("✅ Índice cargado")
        
        # 🔍 PASO 5: Crear QueryEngine
        print("📝 5. Creando QueryEngine...")
        query_engine = QueryEngine(index)
        print("✅ QueryEngine creado")
        
        # 🔍 PASO 6: Ejecutar query
        print("📝 6. Ejecutando query...")
        response = query_engine.query(question)
        print("✅ Query ejecutada")
        
        print("\n" + "="*50)
        print("RESPUESTA:")
        print(response)
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print("\n📋 TRACEBACK COMPLETO:")
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()