#!/usr/bin/env python3
"""
Script de prueba simple para diagnosticar el problema
"""

import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

def test_simple():
    """Prueba simple del pipeline"""
    print("🧪 Prueba simple del pipeline...")
    
    # 1. Cargar imagen
    image_path = "/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg"
    img = Image.open(image_path)
    img = img.convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    print(f"✅ Imagen preprocesada: {img_array.shape}")
    
    # 2. Cargar modelo
    model = tf.keras.models.load_model('/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5')
    print(f"✅ Modelo cargado: {model.output_shape}")
    
    # 3. Predecir
    image_batch = np.expand_dims(img_array, axis=0)
    embedding = model.predict(image_batch, verbose=0)[0]
    print(f"✅ Embedding: {embedding.shape}, tipo: {embedding.dtype}")
    
    # 4. Cargar K-means
    with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    print(f"✅ K-means cargado: {kmeans.n_clusters} clusters")
    
    # 5. Predecir cluster
    embedding_float64 = embedding.astype(np.float64)
    cluster = kmeans.predict([embedding_float64])[0]
    print(f"✅ Cluster predicho: {cluster}")
    
    # 6. Calcular similitud
    center = kmeans.cluster_centers_[cluster]
    distance = np.linalg.norm(embedding_float64 - center)
    similarity = max(0, 100 - (distance / 10.0) * 100)
    print(f"✅ Similitud: {similarity:.1f}%")
    
    print("🎉 ¡Prueba exitosa!")

if __name__ == "__main__":
    test_simple()


