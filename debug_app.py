#!/usr/bin/env python3
"""
Script de debug para diagnosticar problemas en la aplicación
"""

import sys
import os
sys.path.append('/home/jose/PreditorIA2025')

import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

def test_preprocessing():
    """Probar preprocesamiento de imagen"""
    print("🧪 Probando preprocesamiento...")
    
    image_path = "/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg"
    
    try:
        # Cargar imagen
        img = Image.open(image_path)
        print(f"✅ Imagen cargada: {img.size}, modo: {img.mode}")
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"✅ Convertida a RGB: {img.mode}")
        
        # Redimensionar
        img = img.resize((224, 224))
        print(f"✅ Redimensionada: {img.size}")
        
        # Convertir a array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        print(f"✅ Array creado: {img_array.shape}")
        
        # Normalizar (ImageNet)
        img_array = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        print(f"✅ Normalizada: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return None

def test_model_loading():
    """Probar carga del modelo"""
    print("\n🧪 Probando carga del modelo...")
    
    try:
        model_path = '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5'
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado: {model.input_shape}")
        
        # El modelo ya tiene la salida correcta de 11 dimensiones
        print("✅ El modelo ya tiene la salida correcta de 11 dimensiones")
        feature_extractor = model  # Usar el modelo completo
        print(f"✅ Modelo completo usado: {feature_extractor.output_shape}")
        
        return model, feature_extractor
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None

def test_clustering():
    """Probar clustering"""
    print("\n🧪 Probando clustering...")
    
    try:
        # Cargar datos de clustering
        with open('/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl', 'rb') as f:
            clustering_results = pickle.load(f)
        
        # Cargar modelo K-means
        with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        
        print(f"✅ Datos de clustering cargados")
        print(f"   - Clusters: {clustering_results['n_clusters']}")
        print(f"   - Imágenes: {clustering_results['total_images']}")
        print(f"   - Modelo K-means: {kmeans_model.n_clusters} clusters")
        
        return clustering_results, kmeans_model
        
    except Exception as e:
        print(f"❌ Error cargando clustering: {e}")
        return None, None

def test_full_pipeline():
    """Probar pipeline completo"""
    print("\n🧪 Probando pipeline completo...")
    
    # 1. Preprocesar imagen
    img_array = test_preprocessing()
    if img_array is None:
        return
    
    # 2. Cargar modelo
    model, feature_extractor = test_model_loading()
    if model is None or feature_extractor is None:
        return
    
    # 3. Cargar clustering
    clustering_results, kmeans_model = test_clustering()
    if clustering_results is None or kmeans_model is None:
        return
    
        # 4. Extraer embedding
        try:
            print("\n🔄 Extrayendo embedding...")
            image_batch = np.expand_dims(img_array, axis=0)
            embedding = feature_extractor.predict(image_batch, verbose=0)[0]
            print(f"✅ Embedding extraído: {embedding.shape}, tipo: {embedding.dtype}")
            
            # Convertir a float64 para compatibilidad con K-means
            embedding = embedding.astype(np.float64)
            print(f"✅ Embedding convertido a float64: {embedding.dtype}")
            
            # 5. Predecir cluster
            print("\n🔄 Prediciendo cluster...")
            predicted_cluster = kmeans_model.predict([embedding])[0]
            print(f"✅ Cluster predicho: {predicted_cluster}")
            
            # 6. Calcular distancia
            cluster_center = kmeans_model.cluster_centers_[predicted_cluster]
            distance = np.linalg.norm(embedding - cluster_center)
            similarity_score = max(0, 100 - (distance / 10.0) * 100)
            print(f"✅ Similitud: {similarity_score:.1f}%")
            
            print("\n🎉 Pipeline completo funcionando!")
            
        except Exception as e:
            print(f"❌ Error en pipeline: {e}")

if __name__ == "__main__":
    test_full_pipeline()
