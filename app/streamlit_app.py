"""
Streamlit interface - Syngenta Safety Chatbot
"""
import streamlit as st
import sys
from pathlib import Path
import logging

# ✅ CONFIGURAR PATHS CORRECTAMENTE
project_root = Path(__file__).parent.parent  # Subir un nivel desde app/
sys.path.append(str(project_root / "src" / "syngenta_rag" / "core"))
sys.path.append(str(project_root / "config"))

# ✅ IMPORTS CORRECTOS
from document_processor import DocumentProcessor
from settings import settings

# Configurar página
st.set_page_config(
    page_title="🔬 Syngenta Safety Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #00A651 0%, #007A3D 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #00A651;
    }
    .bot-message {
        background-color: #e8f5e8;
        border-left: 4px solid #007A3D;
    }
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Cargar el sistema de chatbot (cached)"""
    try:
        # Inicializar processor
        processor = DocumentProcessor()
        
        # Intentar cargar índice existente
        index = processor.load_index()
        
        if index is None:
            st.warning("📚 Índice no encontrado. Creando uno nuevo...")
            
            # Cargar documentos
            documents = processor.load_documents()
            if not documents:
                st.error("❌ No se encontraron documentos PDF en el directorio")
                st.info(f"📁 Directorio esperado: {settings.PDF_DIR}")
                return None
            
            # Crear índice
            index = processor.create_index(documents)
            
            # Guardar índice
            if processor.save_index(index):
                st.success("✅ Índice creado y guardado correctamente")
            else:
                st.warning("⚠️ Índice creado pero no se pudo guardar")
        
        # Crear query engine
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        st.success("✅ Sistema RAG inicializado correctamente")
        return query_engine
        
    except Exception as e:
        st.error(f"❌ Error inicializando chatbot: {e}")
        st.error("🔧 Soluciones posibles:")
        st.error("1. Ejecuta: `python main.py --reindex`")
        st.error("2. Verifica que hay PDFs en el directorio")
        st.error("3. Revisa los logs para más detalles")
        return None
    

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Syngenta Safety Chatbot</h1>
        <p>Consulta información de fichas de seguridad de productos Syngenta</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Información del Sistema")
        
        # Cargar chatbot
        query_engine = load_chatbot()
        
        if query_engine:
            st.success("✅ Sistema cargado correctamente")
            
            # Mostrar estadísticas
            st.markdown("""
            <div class="sidebar-info">
                <h4>📈 Estadísticas</h4>
                <ul>
                    <li><strong>Estado:</strong> ✅ Activo</li>
                    <li><strong>Modelo:</strong> Mistral 7B</li>
                    <li><strong>Documentos:</strong> Fichas de seguridad</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Ejemplos de preguntas
            st.markdown("### 💡 Preguntas de Ejemplo")
            example_questions = [
                "¿Cuáles son los riesgos de AMISTAR XTRA?",
                "¿Cómo debo almacenar Acelepryn?",
                "¿Qué hacer en caso de contacto con la piel?",
                "¿Cuál es la composición de Abofol L?",
                "¿Qué equipos de protección necesito?"
            ]
            
            for i, question in enumerate(example_questions):
                if st.button(f"📝 {question}", key=f"example_{i}"):
                    st.session_state.example_question = question
        else:
            st.error("❌ Sistema no disponible")
            st.markdown("""
            **Para solucionar:**
            1. Ejecuta: `python main.py --reindex`
            2. Espera que termine el procesamiento
            3. Recarga esta página
            """)
    
    # Main chat area
    if query_engine:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "¡Hola! Soy tu asistente de seguridad de Syngenta. Puedo ayudarte con información sobre fichas de seguridad de nuestros productos. ¿En qué puedo ayudarte?"
                }
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            css_class = "user-message" if message["role"] == "user" else "bot-message"
            icon = "👤" if message["role"] == "user" else "🔬"
            
            st.markdown(f"""
            <div class="chat-message {css_class}">
                <strong>{icon} {message["role"].title()}:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        
        # Handle example question
        if "example_question" in st.session_state:
            user_input = st.session_state.example_question
            del st.session_state.example_question
        else:
            user_input = st.chat_input("Escribe tu pregunta sobre seguridad...")
        
        # Process user input
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Show user message immediately
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 Usuario:</strong><br>
                {user_input}
            </div>
            """, unsafe_allow_html=True)
            
            # Generate response
            with st.spinner("🤔 Buscando información..."):
                try:
                    response = query_engine.query(user_input)
                    bot_response = str(response)
                    
                    # Add bot response
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                    # Show bot response
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🔬 Asistente:</strong><br>
                        {bot_response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f"❌ Error procesando consulta: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Rerun to update display
            st.rerun()
    
    else:
        st.warning("⚠️ Sistema no disponible. Revisa la configuración en el sidebar.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        🔬 Syngenta Safety Chatbot | Powered by LlamaIndex & Mistral
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()