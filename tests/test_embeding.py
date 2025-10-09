# test_embeding.py
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config.settings import settings
from src.syngenta_rag.core.embeddings import EmbeddingManager

def test_embeddings():
    print("🧪 Test de Embeddings\n")
    
    # 1. Cargar modelo
    print("1️⃣ Cargando modelo...")
    manager = EmbeddingManager()
    embed_model = manager.get_embedding_model()
    print(f"   ✅ Modelo: {type(embed_model).__name__}")
    print(f"   📐 Dimensiones: {manager.get_dimension()}")
    
    # 2. Generar embeddings de prueba
    print("\n2️⃣ Generando embeddings...")
    texts = [
        "ACELEPRYN es un insecticida",
        "En caso de contacto con la piel, lavar con agua",
        "Producto químico de Syngenta"
    ]
    
    embeddings = []
    for text in texts:
        emb = embed_model.get_text_embedding(text)
        embeddings.append(emb)
        print(f"   - Texto: '{text[:40]}...'")
        print(f"     Dimensión: {len(emb)}")
        print(f"     Primeros valores: {[f'{v:.4f}' for v in emb[:5]]}")
    
    # 3. Calcular similitud
    print("\n3️⃣ Calculando similitudes...")
    
    def cosine_similarity(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(x*x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
    
    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    sim_12 = cosine_similarity(embeddings[1], embeddings[2])
    
    print(f"   - Similitud [0-1]: {sim_01:.4f}")
    print(f"   - Similitud [0-2]: {sim_02:.4f}")
    print(f"   - Similitud [1-2]: {sim_12:.4f}")
    

        # 4. Verificar ChromaDB
    print("\n4️⃣ Verificando ChromaDB...")
    try:
        import chromadb
        
        db_path = settings.CHROMA_DB_PATH
        print(f"   📂 Ruta DB: {db_path}")
        
        if not db_path.exists():
            print(f"   ⚠️ Base de datos no existe")
            print(f"   💡 Ejecuta: python -m src.syngenta_rag.indexing.index_documents")
            return
        
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("syngenta_docs")
        
        print(f"   ✅ Colección encontrada")
        count = collection.count()
        print(f"   - Chunks indexados: {count}")
        
        # Obtener un sample
        if count > 0:
            print(f"   🔍 Obteniendo muestra...")
            try:
                results = collection.peek(limit=1)
                print(f"   - Claves en results: {list(results.keys())}")
                
                # Verificar embeddings
                if 'embeddings' in results:
                    embeddings_data = results['embeddings']
                    print(f"   - Tipo de embeddings: {type(embeddings_data)}")
                    print(f"   - Cantidad de embeddings: {len(embeddings_data) if embeddings_data else 0}")
                    
                    if embeddings_data and len(embeddings_data) > 0:
                        sample_emb = embeddings_data[0]
                        print(f"   - Dimensión en DB: {len(sample_emb)}")
                        print(f"   - Primeros valores: {[f'{v:.4f}' for v in sample_emb[:5]]}")
                    else:
                        print(f"   ⚠️ embeddings está vacío")
                else:
                    print(f"   ⚠️ No hay clave 'embeddings' en results")
                    
            except Exception as e:
                print(f"   ⚠️ Error al obtener muestra: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️ La colección está vacía")
            return
        
        # 5. Test de búsqueda
        print("\n5️⃣ Test de búsqueda...")
        query = "contacto con la piel ACELEPRYN"
        print(f"   🔍 Generando embedding para query...")
        query_emb = embed_model.get_text_embedding(query)
        print(f"   - Dimensión query: {len(query_emb)}")
        
        print(f"   🔍 Ejecutando búsqueda...")
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        
        print(f"   Query: '{query}'")
        print(f"   - Claves en results: {list(results.keys())}")
        
        if 'ids' in results and results['ids']:
            print(f"   Resultados encontrados: {len(results['ids'][0])}")
            
            if 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    print(f"\n   [{i+1}] Distance: {dist:.4f}")
                    print(f"       Texto: {doc[:150]}...")
            else:
                print(f"   ⚠️ No hay documentos en los resultados")
        else:
            print(f"   ⚠️ No se encontraron resultados")
            
    except ImportError:
        print(f"   ❌ ChromaDB no instalado")
        print(f"   💡 Instala: pip install chromadb")
    except Exception as e:
        print(f"   ❌ Error con ChromaDB: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()