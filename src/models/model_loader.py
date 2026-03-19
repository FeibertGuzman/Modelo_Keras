import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow.keras.models import load_model

class ModelLoader:
    def __init__(self, model_path="converted_keras/keras_model.h5", labels_path="converted_keras/labels.txt"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_names = []

    def load(self):
        """Carga el modelo H5 y las etiquetas en memoria."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en la ruta especificada: {self.model_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"No se encontró el archivo de etiquetas en: {self.labels_path}")
        
        # Deshabilitamos el modo de compilación si solo queremos hacer inferencia
        self.model = load_model(self.model_path, compile=False)
        
        # Leemos las etiquetas y limpiamos los retornos de carro (saltos de línea)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Asumimos que el formato es "0 Etiqueta" (ej. "0 Feibert")
            self.class_names = [line.strip().split(" ", 1)[1] if " " in line else line.strip() for line in lines]
            
        return self.model, self.class_names

    def get_model(self):
        return self.model

    def get_class_names(self):
        return self.class_names
