# start_with_ngrok.py
from pyngrok import ngrok, conf
import uvicorn
import sys

# Tu authtoken de ngrok
NGROK_TOKEN = "2z65xwF3j3E8ksJ2yRFZk1uVRVS_87QkK4WmmZKdgrEM8ZfVX"  # Obtener de: https://dashboard.ngrok.com/get-started/your-authtoken

def main():
    try:
        # Configurar ngrok
        conf.get_default().auth_token = NGROK_TOKEN
        
        # Crear túnel
        print("🔄 Creando túnel público...")
        public_url = ngrok.connect(8000, bind_tls=True)
        
        print("\n" + "="*70)
        print("✅ TÚNEL PÚBLICO CREADO")
        print("="*70)
        print(f"🌐 URL pública: {public_url}")
        print(f"📋 Comparte esta URL con tu compañero para n8n")
        print(f"📊 Dashboard: http://localhost:4040")
        print("="*70 + "\n")
        
        # Importar y ejecutar API
        from api_n8n import app
        print("🚀 Iniciando API...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n👋 Cerrando túnel...")
        ngrok.kill()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()