#!/usr/bin/env python3
"""
Test directo del modelo para verificar su comportamiento
"""

import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesar imagen para el modelo"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    return img_array

def test_model():
    """Test directo del modelo"""
    print("🔍 Cargando modelo...")
    model = tf.keras.models.load_model('/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5')
    
    print(f"📊 Modelo cargado - Output shape: {model.output_shape}")
    print(f"📊 Número de clases: {model.output_shape[1]}")
    
    # Probar con imagen
    image_path = '/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg'
    print(f"🖼️  Procesando imagen: {image_path}")
    
    image = preprocess_image(image_path)
    image_batch = np.expand_dims(image, axis=0)
    
    # Predicción
    prediction = model.predict(image_batch, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    print(f"🎯 Predicción completa: {prediction}")
    print(f"🎯 Clase predicha: {predicted_class}")
    print(f"🎯 Confianza: {confidence:.4f} ({confidence*100:.1f}%)")
    
    # Mostrar todas las probabilidades
    print("\n📊 Probabilidades por clase:")
    for i, prob in enumerate(prediction):
        print(f"  Clase {i}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Verificar si la predicción es razonable
    if confidence < 0.5:
        print("⚠️  ADVERTENCIA: Confianza muy baja, el modelo no está seguro")
    
    if predicted_class < 0 or predicted_class > 10:
        print(f"⚠️  ADVERTENCIA: Clase predicha fuera del rango esperado: {predicted_class}")

if __name__ == "__main__":
    test_model()


