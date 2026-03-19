# Clasificador de Imágenes con Keras y Streamlit

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

Este es un proyecto básico de Machine Learning que utiliza un modelo de clasificación pre-entrenado exportado de formato Keras (Teachable Machine/TensorFlow H5) con una aplicación web frontend sencilla construida sobre [Streamlit](https://streamlit.io/).

## Arquitectura del Proyecto

El proyecto está diseñado siguiendo una arquitectura limpia (por capas) para asegurar su mantenibilidad y escabilidad:

```text
├── app.py                      # (Frontend / UX Layer) Interfaz web Streamlit interactiva
├── requirements.txt            # Dependencias del proyecto
├── converted_keras/
│   ├── keras_model.h5          # Modelo entrenado 
│   └── labels.txt              # Etiquetas del clasificador
└── src/
    ├── models/
    │   └── model_loader.py     # (Data Access Layer) Envuelve la carga del modelo 
    └── services/
        └── prediction_service.py # (Business Logic Layer) Preprocesamiento, predicciones
```

## Requisitos

- Python 3.9 o superior.

Clona este repositorio o asegúrate de tener todos los archivos localmente. Para instalar todas las dependencias ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

## Uso

Una vez las dependencias estén listas, despliega el servidor local de Streamlit con:

```bash
streamlit run app.py
```

Abrirá una ventana en tu navegador por defecto (generalmente en `http://localhost:8501`).
Podrás subir archivos `.jpg`, `.jpeg` o `.png` para que el modelo identifique de qué se trata, retornando el porcentaje de confianza.

---

> Aplicación desarrollada mediante arquitectura por capas para fines de organización de Machine Learning en Python.
