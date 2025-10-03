from typing import Optional
from loguru import logger
from llama_index.core import PromptTemplate, VectorStoreIndex
from  config.settings import settings  # 🔥 IMPORTAR SETTINGS

class QueryEngine:
    """Motor de consultas con prompt personalizado de Syngenta"""
    
    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 5):
        self.index = index
        self.similarity_top_k = similarity_top_k
        
        # 🔥 USAR PROMPT DESDE SETTINGS
        qa_template = PromptTemplate(settings.PROMPTS["qa_template"])
        
        self.query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )
        
        logger.info(f"🤖 QueryEngine inicializado con prompt centralizado (top_k={similarity_top_k})")
    
 
    def query(self, question: str) -> str:
        """
        Realiza una consulta usando el prompt de Syngenta
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Respuesta generada por el modelo
        """
        logger.info(f"🔍 Consultando: {question}")
        
        try:
            response = self.query_engine.query(question)
            logger.info("✅ Respuesta generada exitosamente")
            return str(response)
            
        except Exception as e:
            logger.error(f"❌ Error en consulta: {e}")
            return f"Error al procesar la consulta: {str(e)}"
    
    def get_source_documents(self, question: str) -> list:
        """
        Obtiene los documentos fuente utilizados para responder
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Lista de documentos fuente
        """
        try:
            response = self.query_engine.query(question)
            return response.source_nodes if hasattr(response, 'source_nodes') else []
        except Exception as e:
            logger.error(f"❌ Error obteniendo fuentes: {e}")
            return []