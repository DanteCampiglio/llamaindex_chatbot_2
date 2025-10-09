"""
Construcción y limpieza de respuestas
"""
from typing import List, Dict, Any
from loguru import logger
from llama_index.core.schema import NodeWithScore

class ResponseBuilder:
    """Constructor de respuestas con limpieza y formateo"""
    
    # Palabras clave a filtrar de las respuestas
    METADATA_KEYWORDS = [
        'page_label:',
        'file_path:',
        'FICHA DE DATOS DE SEGURIDAD',
        'según el Reglamento',
        'Versión',
        'Fecha de revisión:',
        'Número SDS:',
        'Fecha de la última expedición:',
        'Fecha de la primera expedición:'
    ]
    
    @staticmethod
    def clean_response(response_text: str) -> str:
        """
        Limpia la respuesta removiendo metadata no deseada
        
        Args:
            response_text: Texto de respuesta crudo
            
        Returns:
            Texto limpio
        """
        response_text = str(response_text).strip()
        
        # Filtrar líneas con metadata
        lines = response_text.split('\n')
        clean_lines = [
            line for line in lines 
            if not any(kw in line for kw in ResponseBuilder.METADATA_KEYWORDS)
        ]
        
        return '\n'.join(clean_lines).strip()
    
    @staticmethod
    def extract_sources(source_nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """
        Extrae información de las fuentes utilizadas
        
        Args:
            source_nodes: Nodos fuente de LlamaIndex
            
        Returns:
            Lista de diccionarios con información de fuentes
        """
        sources = []
        
        for i, node in enumerate(source_nodes, 1):
            score = node.score if hasattr(node, 'score') else 0.0
            text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            
            sources.append({
                "rank": i,
                "score": float(score),
                "text": text_preview,
                "metadata": node.metadata if hasattr(node, 'metadata') else {}
            })
            
            logger.debug(f"   [{i}] Score: {score:.4f} | {text_preview[:50]}...")
        
        return sources
    
    @staticmethod
    def build_response_dict(
        response_text: str,
        sources: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construye el diccionario de respuesta completo
        
        Args:
            response_text: Texto de la respuesta
            sources: Lista de fuentes
            metadata: Metadata adicional
            
        Returns:
            Diccionario estructurado con la respuesta
        """
        return {
            "response": response_text,
            "sources": sources,
            "metadata": {
                "num_sources": len(sources),
                **metadata
            }
        }