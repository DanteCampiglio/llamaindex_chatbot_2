# src/syngenta_rag/core/document_processor.py
import sys
from pathlib import Path
from typing import List, Optional
from loguru import logger

# 🔥 IMPORTAR desde TU config en la raíz (no duplicar)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "config"))
from settings import settings, setup_llama_index  

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document

class DocumentProcessor:
    def __init__(self, vector_store=None, data_dir: Optional[str] = None):
        # 🔥 INICIALIZAR LLAMAINDEX PRIMERO (solo una vez)
        if vector_store is None:
            _, _, vector_store = setup_llama_index()  
        
        self.vector_store = vector_store
        self.pdf_reader = PyMuPDFReader()
        
        # 👇 USA TU CONFIGURACIÓN EXISTENTE
        self.node_parser = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # 👇 Usar tu directorio de PDFs
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(settings.PDF_DIR)  # ✅ Convertir a Path
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Modo: {settings.MODE}")
        logger.info(f"📊 ChromaDB: {settings.CHROMA_DB_PATH}")
    
    def list_available_pdfs(self) -> List[str]:
        """Lista todos los PDFs disponibles en el directorio"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        return [pdf.name for pdf in pdf_files]
    
    def load_pdfs(self, pdf_files: Optional[List[str]] = None) -> List[Document]:
        """Carga PDFs específicos o todos los disponibles"""
        if pdf_files is None:
            pdf_files = self.list_available_pdfs()
        
        if not pdf_files:
            logger.warning("No se encontraron archivos PDF para cargar")
            return []
        
        documents = []
        for pdf_file in pdf_files:
            pdf_path = self.data_dir / pdf_file
            if pdf_path.exists():
                try:
                    docs = self.pdf_reader.load_data(file_path=pdf_path)
                    documents.extend(docs)
                    logger.info(f"✅ Cargado: {pdf_file}")
                except Exception as e:
                    logger.error(f"❌ Error cargando {pdf_file}: {e}")
            else:
                logger.warning(f"⚠️ Archivo no encontrado: {pdf_path}")
        
        logger.info(f"📊 Total documentos cargados: {len(documents)}")
        return documents
    
    def load_documents(self) -> List[Document]:
        """Alias para compatibilidad con main.py"""
        return self.load_pdfs()
    
    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Crea un índice vectorial a partir de los documentos"""
        if not documents:
            raise ValueError("No hay documentos para indexar")
        
        # Parsear documentos en nodos
        nodes = self.node_parser.get_nodes_from_documents(documents)
        logger.info(f"📝 Nodos creados: {len(nodes)}")
        
        # Crear contexto de almacenamiento
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Crear índice
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        logger.info("🔍 Índice vectorial creado")
        
        return index
    
    def save_index(self, index: VectorStoreIndex, persist_dir: Optional[str] = None) -> bool:
        """Guarda el índice en disco"""
        try:
            if persist_dir is None:
                persist_dir = str(settings.INDEX_DIR)
            
            # Crear directorio si no existe
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            
            # Persistir índice
            index.storage_context.persist(persist_dir=persist_dir)
            logger.info(f"💾 Índice guardado en: {persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando índice: {e}")
            return False

    def load_index(self, persist_dir: Optional[str] = None) -> Optional[VectorStoreIndex]:
        """Carga un índice existente desde disco"""
        if persist_dir is None:
            persist_dir = str(settings.INDEX_DIR)
        
        persist_path = Path(persist_dir)
        
        if not persist_path.exists():
            logger.warning(f"📁 Directorio de índice no existe: {persist_dir}")
            return None
        
        try:
            # ✅ MÉTODO CORRECTO para LlamaIndex v0.10+
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=persist_dir
            )
            
            index = load_index_from_storage(storage_context)
            logger.info(f"✅ Índice cargado desde: {persist_dir}")
            return index
            
        except Exception as e:
            logger.error(f"❌ Error cargando índice: {e}")
            logger.info("🔄 Intentando recrear índice...")
            
            try:
                # Recrear índice como fallback
                documents = self.load_pdfs()
                if documents:
                    new_index = self.create_index(documents)
                    self.save_index(new_index, persist_dir)
                    return new_index
                else:
                    logger.error("❌ No hay documentos para recrear índice")
                    return None
                    
            except Exception as recreate_error:
                logger.error(f"❌ Error recreando índice: {recreate_error}")
                return None