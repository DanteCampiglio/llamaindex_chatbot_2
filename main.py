"""
main.py - Orquestador principal del chatbot Syngenta
"""
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Setup paths - CORREGIDO para tu estructura
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src" / "syngenta_rag" / "core"))
sys.path.append(str(project_root / "config"))

# Imports corregidos
from document_processor import DocumentProcessor
from settings import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyngentaChatbot:
    """Clase principal que maneja todo el sistema"""
    
    def __init__(self):
        self.document_processor = None
        self.query_engine = None
        self._setup_directories()
    
    def _setup_directories(self):
        """Crear directorios necesarios - CORREGIDO"""
        # Convertir strings a Path objects
        directories = [
            Path(settings.PDF_DIR),                    # ✅ Convertir a Path
            Path(settings.INDEX_DIR),                  # ✅ Convertir a Path
            Path(settings.CHROMA_DB_PATH).parent,      # ✅ Convertir a Path primero
            Path("logs"),
            Path("model")  # Para el modelo Mistral
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Directorio creado/verificado: {dir_path}")
    
    def setup_system(self, force_reindex=False):
        """Configurar todo el sistema"""
        logger.info("🚀 Inicializando sistema Syngenta Chatbot...")
        
        try:
            # 1. Verificar modelo LLM
            model_path = Path("model") / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            if not model_path.exists():
                logger.error(f"❌ Modelo no encontrado: {model_path}")
                logger.info("📁 Descarga el modelo desde:")
                logger.info("   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
                logger.info("   Y colócalo en la carpeta 'model/'")
                return False
            
            # 2. Inicializar componentes
            logger.info("🔧 Inicializando componentes...")
            self.document_processor = DocumentProcessor()
            
            # 3. Procesar documentos si es necesario
            if force_reindex or not self._index_exists():
                logger.info("📚 Procesando documentos...")
                success = self._process_documents()
                if not success:
                    logger.warning("⚠️ No se procesaron documentos, pero continuando...")
            
            logger.info("✅ Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            logger.error(f"   Tipo de error: {type(e).__name__}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _index_exists(self):
        """Verificar si existe índice persistido"""
        index_dir = Path(settings.INDEX_DIR)
        if not index_dir.exists():
            logger.debug(f"📂 Directorio de índice no existe: {index_dir}")
            return False
        
        # Verificar archivos típicos de LlamaIndex
        required_files = ["docstore.json", "index_store.json", "vector_store.json"]
        existing_files = [f for f in required_files if (index_dir / f).exists()]
        
        logger.debug(f"📋 Archivos de índice encontrados: {existing_files}")
        return len(existing_files) > 0
    
    def _process_documents(self):
        """Procesar documentos y crear índice"""
        try:
            # Verificar PDFs disponibles
            available_pdfs = self.document_processor.list_available_pdfs()
            
            if not available_pdfs:
                pdf_dir = Path(settings.PDF_DIR)
                logger.warning(f"⚠️ No hay PDFs en {pdf_dir.absolute()}")
                logger.info(f"💡 Coloca archivos PDF en: {pdf_dir.absolute()}")
                
                # Crear archivo de ejemplo
                example_file = pdf_dir / "README.txt"
                if not example_file.exists():
                    example_file.write_text(
                        "Coloca aquí tus archivos PDF de fichas de seguridad.\n"
                        "Ejemplo: ficha_producto_1.pdf, ficha_producto_2.pdf"
                    )
                    logger.info(f"📝 Creado archivo de ejemplo: {example_file}")
                
                return False
            
            logger.info(f"📄 Encontrados {len(available_pdfs)} PDFs:")
            for pdf in available_pdfs[:5]:  # Mostrar primeros 5
                logger.info(f"  - {pdf}")
            if len(available_pdfs) > 5:
                logger.info(f"  ... y {len(available_pdfs) - 5} más")
            
            # Cargar documentos
            logger.info("📖 Cargando documentos...")
            documents = self.document_processor.load_pdfs()
            
            if not documents:
                logger.warning("⚠️ No se pudieron cargar documentos")
                return False
            
            # Crear índice
            logger.info("🔍 Creando índice vectorial...")
            index = self.document_processor.create_index(documents)
            
            # Guardar índice
            logger.info("💾 Guardando índice...")
            self.document_processor.save_index(index)
            
            logger.info("✅ Documentos procesados e indexados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error procesando documentos: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def run_ui(self, port=8501):
        """Lanzar interfaz Streamlit"""
        logger.info(f"🌐 Iniciando interfaz web en puerto {port}")
        logger.info(f"🔗 URL: http://localhost:{port}")
        
        # Verificar que streamlit_app.py existe
        streamlit_app = Path("app/streamlit_app.py")
        if not streamlit_app.exists():
            logger.error("❌ No se encontró streamlit_app.py")
            logger.info("💡 Necesitas crear el archivo streamlit_app.py en la raíz del proyecto")
            logger.info("   ¿Quieres que te ayude a crearlo?")
            return
        
        cmd = [
            "streamlit", "run", "app/streamlit_app.py",
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error ejecutando Streamlit: {e}")
        except KeyboardInterrupt:
            logger.info("👋 Cerrando aplicación...")
        except FileNotFoundError:
            logger.error("❌ Streamlit no encontrado")
            logger.info("💡 Instala con: pip install streamlit")

def main():
    """Entry point principal"""
    parser = argparse.ArgumentParser(description="Syngenta Chatbot")
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Puerto para la interfaz web (default: 8501)"
    )
    parser.add_argument(
        "--reindex", 
        action="store_true",
        help="Forzar reindexación de documentos"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Ejecutar en modo debug"
    )
    
    args = parser.parse_args()
    
    # Debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Mostrar configuración actual
    logger.info("=" * 50)
    logger.info("🔬 SYNGENTA CHATBOT")
    logger.info("=" * 50)
    logger.info(f"📁 PDF Directory: {settings.PDF_DIR}")
    logger.info(f"📊 Index Directory: {settings.INDEX_DIR}")
    logger.info(f"🔧 Mode: {settings.MODE}")
    logger.info(f"🗄️ ChromaDB: {settings.CHROMA_DB_PATH}")
    
    # Crear y ejecutar chatbot
    try:
        chatbot = SyngentaChatbot()
        
        if chatbot.setup_system(force_reindex=args.reindex):
            chatbot.run_ui(port=args.port)
        else:
            logger.error("❌ No se pudo inicializar el sistema completamente")
            logger.info("💡 Revisa los logs anteriores para más detalles")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()