#!/usr/bin/env python3
"""
Prueba final del sistema completo
"""

import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

def main():
    print("🧪 Prueba final del sistema...")
    
    # 1. Cargar y preprocesar imagen
    print("1. Cargando imagen...")
    img = Image.open("/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg")
    img = img.convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    print(f"   ✅ Imagen: {img_array.shape}")
    
    # 2. Cargar modelo
    print("2. Cargando modelo...")
    model = tf.keras.models.load_model('/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5')
    print(f"   ✅ Modelo: {model.output_shape}")
    
    # 3. Extraer embedding
    print("3. Extrayendo embedding...")
    image_batch = np.expand_dims(img_array, axis=0)
    embedding = model.predict(image_batch, verbose=0)[0]
    print(f"   ✅ Embedding: {embedding.shape}, {embedding.dtype}")
    
    # 4. Cargar K-means
    print("4. Cargando K-means...")
    with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    print(f"   ✅ K-means: {kmeans.n_clusters} clusters")
    print(f"   ✅ Centros dtype: {kmeans.cluster_centers_.dtype}")
    
    # 5. Corregir tipos
    print("5. Corrigiendo tipos...")
    embedding_fixed = embedding.astype(np.float64)
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)
    print(f"   ✅ Embedding: {embedding_fixed.dtype}")
    print(f"   ✅ Centros: {kmeans.cluster_centers_.dtype}")
    
    # 6. Predecir cluster
    print("6. Prediciendo cluster...")
    try:
        cluster = kmeans.predict([embedding_fixed])[0]
        print(f"   ✅ Cluster: {cluster}")
        
        # 7. Calcular similitud
        center = kmeans.cluster_centers_[cluster]
        distance = np.linalg.norm(embedding_fixed - center)
        similarity = max(0, 100 - (distance / 10.0) * 100)
        print(f"   ✅ Similitud: {similarity:.1f}%")
        
        print("\n🎉 ¡SISTEMA FUNCIONANDO CORRECTAMENTE!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()


