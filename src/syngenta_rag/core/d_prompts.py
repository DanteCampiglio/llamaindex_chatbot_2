from typing import Dict
from llama_index.core import PromptTemplate
from loguru import logger
from config import settings  # Ajusta el import según tu estructura

class PromptManager:
    """Gestor de prompts del sistema"""

    def __init__(self, custom_prompts: Dict[str, str] = None):
        """
        Inicializa el gestor de prompts
        
        Args:
            custom_prompts: Diccionario con prompts personalizados
        """
        # Orden de prioridad: custom_prompts > settings.PROMPTS > defaults hardcodeados
        self.prompts = (
            custom_prompts
            or getattr(settings, "PROMPTS", None)
            or {
                "system": "Responde solo con información del documento.",
                "qa_template": "CONTEXTO: {context_str}\n\nPREGUNTA: {query_str}\n\nSi la respuesta está en el contexto, responde con precisión.\nSi NO está en el contexto, di: \"No encuentro esa información en los documentos\"\n\nRESPUESTA:",
                "refine_template": "Pregunta: {query_str}\nRespuesta actual: {existing_answer}\nNuevo contexto: {context_msg}\n\nMejora la respuesta solo si el nuevo contexto aporta información relevante."
            }
        )
        logger.info(f"📝 PromptManager inicializado con {'custom' if custom_prompts else 'settings' if hasattr(settings, 'PROMPTS') else 'default'} prompts")
    
    def get_qa_template(self) -> PromptTemplate:
        return PromptTemplate(self.prompts.get("qa_template"))

    def get_refine_template(self) -> PromptTemplate:
        return PromptTemplate(self.prompts.get("refine_template"))

    def get_qa_prompt(self) -> PromptTemplate:
        return self.get_qa_template()
    
    def get_refine_prompt(self) -> PromptTemplate:
        return self.get_refine_template()
    
    def get_system_prompt(self) -> str:
        return self.prompts.get("system")
    
    def update_prompt(self, prompt_type: str, new_prompt: str):
        if prompt_type in ["qa_template", "refine_template", "system"]:
            self.prompts[prompt_type] = new_prompt
            logger.info(f"✅ Prompt '{prompt_type}' actualizado")
        else:
            logger.warning(f"⚠️ Tipo de prompt desconocido: {prompt_type}")