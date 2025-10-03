# tests/test_document_processor.py
#!/usr/bin/env python3
"""Script para probar la versión refactorizada"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 🔥 IMPORTAR desde TU config original
sys.path.append(str(project_root / "config"))
from settings import settings

from syngenta_rag.core.document_processor import DocumentProcessor

def main():
    print("🚀 Probando DocumentProcessor refactorizado...")
    
    print(f"📁 PDF Directory: {settings.PDF_DIR}")
    print(f"📁 Index Directory: {settings.INDEX_DIR}")
    
    # 🔥 CREAR PROCESSOR UNA SOLA VEZ (se auto-inicializa)
    processor = DocumentProcessor()
    
    # Listar PDFs disponibles
    pdfs = processor.list_available_pdfs()
    print(f"📄 PDFs disponibles: {pdfs}")
    
    if pdfs:
        # Cargar y procesar documentos
        documents = processor.load_pdfs()
        print(f"📊 Documentos cargados: {len(documents)}")
        
        # Crear índice
        index = processor.create_index(documents)
        
        # Guardar índice
        processor.save_index(index)
        
        print("✅ ¡Todo funcionando correctamente!")
    else:
        print("⚠️ No hay PDFs para procesar. Agrega algunos PDFs a data/raw/pdfs/")

if __name__ == "__main__":
    main()