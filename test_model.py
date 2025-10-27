import warnings
warnings.filterwarnings('ignore')

# Parchear requests ANTES de importar cualquier otra cosa
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import ssl

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', SSLAdapter())

# Reemplazar la sesión global de requests
import requests.sessions
requests.sessions.Session = lambda: session

# Ahora sí, importar sentence_transformers
from sentence_transformers import SentenceTransformer

try:
    print("Descargando modelo...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    print("✅ Modelo descargado exitosamente")
    
    # Guardar localmente
    model.save('./local_model')
    print("✅ Modelo guardado en ./local_model")
    
except Exception as e:
    print(f"❌ Error: {e}")