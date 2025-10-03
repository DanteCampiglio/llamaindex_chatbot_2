import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import streamlit as st
from pathlib import Path

# LLMs y embeddings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

class Config:
    # 🔥 FORZAR MODO LOCAL (ignorar .env)
    MODE = "local"  # ✅ HARDCODEADO - NO usar os.getenv()

    # Local model
    PDF_DIR = os.getenv("PDF_DIR", "./data/raw/pdfs")  

    # 🔥 NUEVO MODELO LLAMA 3.2 3B - MUCHO MÁS RÁPIDO
    MISTRAL_MODEL_PATH = Path(r"./models/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

    INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data/processed/index")) 
    
    # Vector DB
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")  
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/processed/chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "safety_sheets")
    
    # 🔥 CHUNKING - MANTENER CONFIGURACIÓN ORIGINAL
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))  # ✅ ORIGINAL
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # ✅ ORIGINAL
    
    # 🔥 QUERY - MANTENER CONFIGURACIÓN ORIGINAL
    TOP_K = int(os.getenv("TOP_K", 5))  # ✅ ORIGINAL
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

    # 🔥 PROMPTS CENTRALIZADOS - MÁS CONCISOS
    PROMPTS = {
        "system_prompt": """Eres un experto asistente de IA de Syngenta especializado en agricultura. 
Responde siempre en español con tono profesional pero accesible. 
Enfócate en soluciones prácticas para agricultores.""",

        "qa_template": """Eres un experto asistente de IA de Syngenta especializado en agricultura.

Contexto relevante:
{context_str}

Pregunta del usuario: {query_str}

Instrucciones:
- Responde en español como experto en agricultura de Syngenta
- Usa SOLO información del contexto proporcionado
- Si no tienes información suficiente, dilo claramente
- Incluye referencias específicas cuando sea posible
- Enfócate en soluciones prácticas para agricultores

Respuesta:"""
    }

    # 🔥 SYSTEM PROMPT OPTIMIZADO PARA LLAMA 3.2
    LLM_SYSTEM_PROMPT = """Eres un asistente especializado en seguridad de productos Syngenta. 
SIEMPRE responde en ESPAÑOL. 
Proporciona respuestas concisas y específicas basadas únicamente en la información de las fichas de seguridad proporcionadas.
Si no encuentras información específica, di "No encuentro esa información en las fichas disponibles"."""

    # 🔥 CONFIGURACIÓN QUERY ENGINE - MANTENER ORIGINAL
    QUERY_ENGINE_CONFIG = {
        "similarity_top_k": 3,  # ✅ ORIGINAL
        "response_mode": "compact",
        "streaming": False
    }

# Instancia global
settings = Config()

def setup_llama_index():
    """Configura LlamaIndex FORZANDO modo local con Llama 3.2 3B"""
    
    # 🔥 LOGS DETALLADOS
    print(f"🔧 MODO CONFIGURADO: {settings.MODE}")
    print(f"🤖 MODELO PATH: {settings.MISTRAL_MODEL_PATH}")
    print(f"📁 ARCHIVO EXISTE: {settings.MISTRAL_MODEL_PATH.exists()}")
    print(f"📊 TAMAÑO ARCHIVO: {settings.MISTRAL_MODEL_PATH.stat().st_size / (1024**3):.2f} GB" if settings.MISTRAL_MODEL_PATH.exists() else "❌ NO EXISTE")

    # 🔥 VERIFICAR QUE EL ARCHIVO EXISTE
    if not settings.MISTRAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ MODELO NO ENCONTRADO: {settings.MISTRAL_MODEL_PATH}")
    
    print("🚀 Inicializando LlamaCPP con Llama 3.2 3B LOCAL...")
    
    # 🔥 LLM LOCAL - PARÁMETROS CORREGIDOS
    llm = LlamaCPP(
        model_path=str(settings.MISTRAL_MODEL_PATH),
        temperature=settings.TEMPERATURE,
        max_new_tokens=512,  # ✅ MANTENER ORIGINAL
        context_window=4096,  # ✅ MANTENER ORIGINAL
        verbose=True,
        system_prompt=settings.PROMPTS["system_prompt"],  # 🔥 USAR PROMPT NUEVO Y CONCISO
        # 🔥 TODOS LOS PARÁMETROS ESPECÍFICOS VAN EN model_kwargs
        model_kwargs={
            #"n_ctx": 1024,  # ✅ MOVIDO A model_kwargs
            "n_batch": 256,  # ✅ MOVIDO A model_kwargs
            "n_threads": None,  # ✅ MOVIDO A model_kwargs (auto-detect)
            "use_mmap": True,  # ✅ MOVIDO A model_kwargs
            "use_mlock": False,  # ✅ MOVIDO A model_kwargs
            "n_gpu_layers": 0,  # Sin GPU para CPU puro
            "f16_kv": True,  # Float16 para KV cache
        }
    )
    
    print("✅ LlamaCPP inicializado correctamente con Llama 3.2 3B")
    print("🚀 Inicializando embeddings locales...")
    
    # 🔥 EMBEDDINGS LOCALES
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./model/embeddings"
    )
    
    print("✅ Embeddings locales inicializados")
    
    # Vector store
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Settings globales - MANTENER CONFIGURACIÓN ORIGINAL
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = settings.CHUNK_SIZE  # ✅ 1024 ORIGINAL
    Settings.chunk_overlap = settings.CHUNK_OVERLAP  # ✅ 200 ORIGINAL
    
    print("🎯 CONFIGURACIÓN COMPLETADA - LLAMA 3.2 3B CON CONFIGURACIÓN ORIGINAL")
    
    return llm, embed_model, vector_store

def get_chroma_collection():
    """Obtiene la colección de Chroma"""
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return chroma_client.get_or_create_collection(settings.COLLECTION_NAME)

# Test directo
if __name__ == "__main__":
    print("🔥 PROBANDO CONFIGURACIÓN LOCAL CON LLAMA 3.2 3B...")
    llm, embed_model, vector_store = setup_llama_index()
    print(f"✅ LlamaIndex configurado en modo {settings.MODE}")
    print(f"🤖 LLM tipo: {type(llm)}")
    print(f"📊 Embed tipo: {type(embed_model)}")
    
    # 🔥 TEST DEL SYSTEM PROMPT
    print("\n🧪 PROBANDO SYSTEM PROMPT...")
    try:
        response = llm.complete("¿Qué es Syngenta?")
        print(f"✅ RESPUESTA: {response}")
    except Exception as e:
        print(f"❌ ERROR: {e}")