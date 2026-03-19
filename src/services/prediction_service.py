import numpy as np
from PIL import Image, ImageOps

class PredictionService:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.model = None
        self.class_names = []
        self._initialize()

    def _initialize(self):
        """Inicializa el modelo usando el loader inyectado."""
        self.model, self.class_names = self.model_loader.load()

    def predict(self, image_file):
        """
        Recibe un archivo de imagen, lo preprocesa para que se ajuste a 
        las expectativas del modelo Keras (Teachable Machine: 224x224, norm a [-1, 1])
        y devuelve la clase y la confianza.
        """
        # Desactivamos notación científica para más fácil visualización
        np.set_printoptions(suppress=True)
        
        # Cargar imagen
        image = Image.open(image_file).convert("RGB")
        
        # Redimensionar la imagen para que sea 224x224 (esto suele usar Keras Teachable Machine)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Convertir a arreglo numpy
        image_array = np.asarray(image)
        
        # Normalizar imagen (convirtiendo el rango 0-255 a -1 a 1 o 0 a 1 dependiendo 
        # Pero el estándar de resnet/TM suele ser ((x / 127.5) - 1))
        # Si observas problemas de confianza baja, esto puede cambiarse a / 255.0
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Añadir la dimensión de lote (batch dimension) -> (1, 224, 224, 3)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Ejecutar predicción
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        
        return class_name, confidence_score
