"""
api_n8n.py - API optimizada para n8n
Ejecutar: python api_n8n.py
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, setup_llama_index
from src.syngenta_rag.core.index_manager import IndexManager
from src.syngenta_rag.core.query_engine import QueryEngine

# ========== MODELOS ==========
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3  # Default para n8n

class QueryResponse(BaseModel):
    success: bool
    response: str
    sources: List[Dict[str, Any]]
    num_sources: int

# ========== APP ==========
app = FastAPI(
    title="Syngenta RAG API",
    description="API para n8n - Consultas sobre documentos Syngenta",
    version="1.0.0"
)

# CORS - IMPORTANTE para n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especifica el dominio de n8n
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global
query_engine = None

# ========== STARTUP ==========
@app.on_event("startup")
async def startup():
    global query_engine
    
    print("=" * 60)
    print("🚀 INICIANDO SYNGENTA RAG API PARA N8N")
    print("=" * 60)
    
    try:
        # Setup LLM
        print("⚙️  Configurando LLM...")
        setup_llama_index()
        
        # Cargar índice
        print("📦 Cargando índice...")
        index_manager = IndexManager()
        index = index_manager.load_index()
        
        if not index:
            raise RuntimeError("No se pudo cargar índice")
        
        # Query engine
        print("🤖 Creando QueryEngine...")
        query_engine = QueryEngine(
            index=index,
            similarity_top_k=settings.SIMILARITY_TOP_K
        )
        
        print("=" * 60)
        print("✅ API LISTA")
        print(f"   📍 URL: http://localhost:8000")
        print(f"   📚 Docs: http://localhost:8000/docs")
        print(f"   🔗 n8n: Usa POST http://localhost:8000/query")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR EN STARTUP: {e}")
        raise

# ========== ENDPOINTS ==========

@app.get("/")
def root():
    """Endpoint raíz - Info básica"""
    return {
        "service": "Syngenta RAG API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check para n8n - Verifica que la API esté lista"""
    is_ready = query_engine is not None
    
    return {
        "status": "healthy" if is_ready else "initializing",
        "ready": is_ready,
        "model": settings.LLAMA_MODEL_PATH.name if is_ready else None
    }

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Endpoint principal para n8n
    
    Ejemplo de uso en n8n:
    - Method: POST
    - URL: http://localhost:8000/query
    - Body JSON: {"question": "¿Qué hacer en caso de incendio con Abofol?"}
    """
    # Validar que el sistema esté listo
    if query_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Sistema inicializando. Intenta en unos segundos."
        )
    
    try:
        print(f"\n🔍 Query desde n8n: {request.question}")
        
        # Actualizar top_k si se especifica
        if request.top_k:
            query_engine.update_config(similarity_top_k=request.top_k)
        
        # Ejecutar query
        result = query_engine.query_with_sources(request.question)
        
        # Formatear respuesta para n8n
        response = QueryResponse(
            success=True,
            response=result.get('response', ''),
            sources=result.get('sources', []),
            num_sources=len(result.get('sources', []))
        )
        
        print(f"✅ Respuesta enviada ({response.num_sources} fuentes)")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}"
        )

# ========== RUN ==========
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Permite conexiones externas
        port=8000,
        log_level="info"
    )