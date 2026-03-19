import streamlit as st
import os
from PIL import Image

from src.models.model_loader import ModelLoader
from src.services.prediction_service import PredictionService

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Imágenes Keras",
    page_icon="🤖",
    layout="centered"
)

# Inicializar servicios de forma global o en session_state para no recargar el modelo en cada interacción
@st.cache_resource
def load_services():
    try:
        loader = ModelLoader(
            model_path="converted_keras/keras_model.h5", 
            labels_path="converted_keras/labels.txt"
        )
        service = PredictionService(loader)
        return service
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def main():
    st.title("Clasificador Keras + Streamlit 🚀")
    st.markdown("Sube una imagen y el modelo predecirá si se trata de **Feibert** o **Star Wars**.")

    # Cargar el servicio
    service = load_services()
    if not service:
        st.stop()

    # Opción para subir la imagen (Frontend)
    uploaded_file = st.file_uploader("Elige una imagen para clasificar...", type=["jpg", "jpeg", "png"])
    
    # También podemos habilitar entrada de cámara
    # camera_file = st.camera_input("... o tómate una foto")
    # image_data = uploaded_file or camera_file

    if uploaded_file is not None:
        # Mostrar la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        # Botón para predecir
        if st.button("Predecir Imagen", use_container_width=True, type="primary"):
            with st.spinner("Analizando imagen..."):
                try:
                    class_name, confidence = service.predict(uploaded_file)
                    
                    st.success("¡Predicción completada!")
                    
                    # Mostrar resultados de forma presentable
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicción (Clase)", class_name)
                    with col2:
                        # Mostrar el porcentaje de forma aproximada al 2 decimal
                        conf_percentage = f"{confidence * 100:.2f}%"
                        st.metric("Confianza", conf_percentage)
                except Exception as e:
                    st.error(f"Ocurrió un error al predecir: {e}")

if __name__ == "__main__":
    main()
