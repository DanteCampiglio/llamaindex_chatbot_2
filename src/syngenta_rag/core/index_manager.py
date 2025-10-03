from pathlib import Path
from llama_index.core import VectorStoreIndex
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from loguru import logger
import streamlit as st

class IndexManager:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.document_processor = None
        self.vector_store = None
        self.index = None
    
    @st.cache_resource
    def load_and_index_documents(_self, pdf_directory: Path):
        """Carga documentos y crea índice (con caché de Streamlit)"""
        try:
            # Configurar vector store
            _self.vector_store = _self.vector_store_manager.get_chroma_vector_store()
            
            # Configurar document processor con vector store
            _self.document_processor = DocumentProcessor(vector_store=_self.vector_store)
            
            # Verificar PDFs
            if not pdf_directory.exists() or not list(pdf_directory.glob("*.pdf")):
                return None, "❌ No se encontraron PDFs en data/raw/pdfs/"
            
            # Cargar documentos
            documents = _self.document_processor.load_pdfs(pdf_directory)
            
            if not documents:
                return None, "❌ No se pudieron cargar documentos"
            
            # Crear índice
            _self.index = _self.document_processor.create_index(documents)
            
            return _self.index, f"✅ Indexados {len(documents)} documentos"
            
        except Exception as e:
            logger.error(f"❌ Error en indexación: {e}")
            return None, f"❌ Error al indexar: {str(e)}"
    
    def get_query_engine(self, similarity_top_k: int = 5):
        """Crea query engine desde el índice"""
        if not self.index:
            raise ValueError("❌ Índice no inicializado")
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )