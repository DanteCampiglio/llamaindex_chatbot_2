"""
index_manager.py - Gestión centralizada de índices vectoriales
"""
from pathlib import Path
from typing import Optional, Tuple, List
import shutil
from datetime import datetime
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.syngenta_rag.core.a_embeddings import EmbeddingManager
from loguru import logger


class IndexManager:
    """
    Gestor centralizado de índices vectoriales con ChromaDB
    """
    
    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        distance_function: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Inicializa el IndexManager
        
        Args:
            chroma_path: Ruta a la base de datos ChromaDB
            collection_name: Nombre de la colección
            distance_function: Función de distancia ('cosine', 'l2', 'ip')
            embedding_model: Modelo de embeddings a usar
            chunk_size: Tamaño de los chunks
            chunk_overlap: Solapamiento entre chunks
        """
        # Importar settings aquí para evitar circular imports
        from config.settings import settings
        
        self.chroma_path = chroma_path or settings.CHROMA_DB_PATH
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.distance_function = distance_function or getattr(
            settings, 'CHROMA_DISTANCE_FUNCTION', 'cosine'
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Crear directorio si no existe
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar embeddings
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model or "local",
            dimensions=getattr(settings, 'EMBEDDING_DIMENSIONS', 512)
        )

        # ✅ FORZAR uso de LocalEmbedding
        local_embed = self.embedding_manager.get_embedding_model()
        Settings.embed_model = local_embed

        # 🔍 Verificar que se usa LocalEmbedding
        logger.info(f"✅ Embedding: {type(local_embed).__name__} ({local_embed._dimensions}D)")
        
        # Configurar settings globales de LlamaIndex
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Inicializar variables
        self._vector_store = None
        self._storage_context = None
        
        logger.info(f"🔧 IndexManager inicializado")
        logger.info(f"   Path: {self.chroma_path}")
        logger.info(f"   Colección: {self.collection_name}")
        logger.info(f"   Función de distancia: {self.distance_function}")
        logger.info(f"   Modelo embeddings: {self.embedding_manager.current_model}")
    
    def _create_vector_store(self) -> ChromaVectorStore:
        """Crea el vector store con función de distancia correcta"""
        from chromadb.config import Settings as ChromaSettings
        
        logger.info(f"🔧 Creando ChromaDB en: {self.chroma_path}")
        
        chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        try:
            # Intentar obtener colección existente
            collection = chroma_client.get_collection(name=self.collection_name)
            logger.info(f"✅ Colección existente: {self.collection_name}")
            
            # Verificar configuración
            metadata = collection.metadata or {}
            current_distance = metadata.get("hnsw:space", "unknown")
            
            logger.info(f"   Función de distancia actual: {current_distance}")
            
            if current_distance != self.distance_function:
                logger.warning(
                    f"⚠️ Función de distancia diferente: "
                    f"actual={current_distance}, esperada={self.distance_function}"
                )
                logger.warning("   Considera reindexar con --reindex")
            
        except Exception as e:
            # Crear nueva colección
            logger.info(f"📦 Creando nueva colección: {self.collection_name}")
            logger.info(f"   Función de distancia: {self.distance_function}")
            
            collection = chroma_client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_function,
                    "hnsw:construction_ef": 100,
                    "hnsw:M": 16
                }
            )
            logger.info(f"   ✅ Colección creada con éxito")
        
        return ChromaVectorStore(chroma_collection=collection)
    
    def _initialize_chroma(self, force_reset: bool = False) -> None:
        """
        Inicializa o reinicia ChromaDB
        
        Args:
            force_reset: Si True, elimina y recrea la colección
        """
        from chromadb.config import Settings as ChromaSettings
        
        # Crear directorio si no existe
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar cliente ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Resetear colección si se solicita
        if force_reset:
            try:
                chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"Colección '{self.collection_name}' eliminada")
            except Exception as e:
                logger.debug(f"No se pudo eliminar colección (puede no existir): {e}")
        
        # Crear o obtener colección con configuración explícita
        try:
            chroma_collection = chroma_client.get_collection(name=self.collection_name)
            logger.info(f"✅ Colección existente: {self.collection_name}")
        except Exception:
            logger.info(f"📦 Creando colección: {self.collection_name}")
            chroma_collection = chroma_client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_function,
                    "hnsw:construction_ef": 100,
                    "hnsw:M": 16
                }
            )
        
        # Crear vector store
        self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Crear storage context
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        
        logger.info(f"ChromaDB inicializado en: {self.chroma_path}")
    
    def load_and_index_documents(
        self,
        pdf_directory: Optional[Path] = None,
        force_reindex: bool = False
    ) -> Tuple[Optional[VectorStoreIndex], str]:
        """
        Carga e indexa documentos PDF
        
        Args:
            pdf_directory: Directorio con PDFs
            force_reindex: Si True, reindexar aunque ya exista índice
            
        Returns:
            Tupla (índice, mensaje)
        """
        from config.settings import settings
        pdf_directory = pdf_directory or settings.PDF_DIR
        
        # Verificar si ya existe índice
        if not force_reindex and self._index_exists():
            logger.info("Índice existente encontrado, cargando...")
            index = self.load_index()
            if index:
                return index, "✅ Índice cargado desde almacenamiento existente"
        
        # Cargar documentos con SimpleDirectoryReader
        logger.info(f"Cargando PDFs desde: {pdf_directory}")
        
        reader = SimpleDirectoryReader(
            input_dir=str(pdf_directory),
            required_exts=[".pdf"],
            recursive=False
        )
        
        documents = reader.load_data()
        
        if not documents:
            return None, f"❌ No se encontraron documentos en {pdf_directory}"
        
        logger.info(f"Documentos cargados: {len(documents)}")
        
        # Inicializar ChromaDB (forzar reset si reindexar)
        self._initialize_chroma(force_reset=force_reindex)
        
        # Crear parser de nodos
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Crear índice
        logger.info("Creando índice vectorial...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context,
            transformations=[node_parser],
            show_progress=True
        )
        
        # Persistir índice
        index.storage_context.persist(persist_dir=str(self.chroma_path))
        
        # Guardar metadatos
        self._save_metadata(len(documents))
        
        msg = f"✅ Índice creado: {len(documents)} documentos procesados"
        logger.success(msg)
        
        return index, msg
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Carga índice existente desde ChromaDB
        
        Returns:
            Índice cargado o None si no existe
        """
        try:
            if not self._index_exists():
                logger.warning("No existe índice para cargar")
                return None
            
            # Inicializar ChromaDB sin reset
            self._initialize_chroma(force_reset=False)
            
            # Cargar índice
            index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                storage_context=self._storage_context
            )
            
            logger.info("Índice cargado exitosamente")
            return index
            
        except Exception as e:
            logger.error(f"Error cargando índice: {e}")
            return None
    
    def _index_exists(self) -> bool:
        """
        Verifica si existe un índice
        
        Returns:
            True si existe índice
        """
        return self.chroma_path.exists() and any(self.chroma_path.iterdir())
    
    def _save_metadata(self, doc_count: int) -> None:
        """
        Guarda metadatos del índice
        
        Args:
            doc_count: Número de documentos indexados
        """
        metadata = {
            "created_at": datetime.now().isoformat(),
            "document_count": doc_count,
            "embedding_model": self.embedding_manager.current_model,
            "embedding_dimension": self.embedding_manager.get_dimension(),
            "distance_function": self.distance_function,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        
        metadata_file = self.chroma_path / "metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadatos guardados en: {metadata_file}")
    
    def delete_index(self) -> bool:
        """
        Elimina el índice completo
        
        Returns:
            True si se eliminó correctamente
        """
        try:
            if self.chroma_path.exists():
                shutil.rmtree(self.chroma_path)
                logger.info(f"Índice eliminado: {self.chroma_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando índice: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del índice
        
        Returns:
            Dict con estadísticas
        """
        stats = {
            "index_exists": self._index_exists(),
            "chroma_db_path": str(self.chroma_path),
            "collection_name": self.collection_name,
            "distance_function": self.distance_function,
            "embeddings": {
                "model_name": self.embedding_manager.current_model,
                "dimension": self.embedding_manager.get_dimension()
            }
        }
        
        # Estadísticas de ChromaDB
        if self._index_exists():
            try:
                self._initialize_chroma(force_reset=False)
                
                # Obtener colección
                from chromadb.config import Settings as ChromaSettings
                chroma_client = chromadb.PersistentClient(
                    path=str(self.chroma_path),
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                collection = chroma_client.get_collection(name=self.collection_name)
                
                stats["vector_store"] = {
                    "document_count": collection.count(),
                    "collection_name": self.collection_name
                }
            except Exception as e:
                logger.warning(f"No se pudieron obtener estadísticas de ChromaDB: {e}")
        
        return stats
    
    def get_available_models(self) -> dict:
        """
        Lista modelos de embeddings disponibles
        
        Returns:
            Dict con modelos disponibles
        """
        return EmbeddingManager.list_models()


# ========================================================================
# 🚀 SCRIPT DE INDEXACIÓN
# ========================================================================

def main():
    """Script para indexar documentos desde línea de comandos"""
    from config.settings import settings
    
    print("=" * 70)
    print("🚀 SYNGENTA RAG - INDEXACIÓN DE DOCUMENTOS")
    print("=" * 70)
    
    # 1. Verificar PDFs
    pdf_dir = settings.PDF_DIR
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠️  NO SE ENCONTRARON PDFs")
        print(f"   Directorio: {pdf_dir.absolute()}")
        return
    
    print(f"\n📄 PDFs encontrados: {len(pdf_files)}")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   {i}. {pdf.name} ({size_mb:.2f} MB)")

    # 2. Preguntar si reindexar
    print(f"\n📦 Directorio de indexación: {settings.CHROMA_DB_PATH}")
    
    force_reindex = False
    if settings.CHROMA_DB_PATH.exists():
        response = input("\n⚠️  Ya existe un índice. ¿Reindexar todo? (s/N): ").strip().lower()
        force_reindex = response == 's'
    
    # 3. Crear IndexManager
    print("\n🔧 Inicializando IndexManager...")
    
    embedding_model = getattr(settings, 'EMBEDDING_MODEL', None)
    index_manager = IndexManager(embedding_model=embedding_model)
    
    # 4. Indexar
    print("\n🔄 Procesando documentos...")
    print(f"   Chunk size: {settings.CHUNK_SIZE}")
    print(f"   Chunk overlap: {settings.CHUNK_OVERLAP}")
    
    try:
        index, message = index_manager.load_and_index_documents(
            pdf_directory=pdf_dir,
            force_reindex=force_reindex
        )
        
        if index is None:
            print(f"\n❌ ERROR: {message}")
            return
        
        print(f"\n{message}")
        
        # 5. Estadísticas
        print("\n" + "=" * 70)
        print("📊 ESTADÍSTICAS")
        print("=" * 70)
        
        stats = index_manager.get_stats()
        
        vs_stats = stats.get('vector_store', {})
        print(f"\n🗄️  Vector Store:")
        print(f"   - Chunks: {vs_stats.get('document_count', 'N/A')}")
        print(f"   - Colección: {vs_stats.get('collection_name', 'N/A')}")
        
        emb_stats = stats.get('embeddings', {})
        print(f"\n🧠 Embeddings:")
        print(f"   - Modelo: {emb_stats.get('model_name', 'N/A')}")
        print(f"   - Dimensión: {emb_stats.get('dimension', 'N/A')}")
        
        print(f"\n🔧 Configuración:")
        print(f"   - Función de distancia: {stats.get('distance_function', 'N/A')}")
        
        print("\n" + "=" * 70)
        print("✅ INDEXACIÓN COMPLETADA")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
    