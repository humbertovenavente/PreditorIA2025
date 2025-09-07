#!/usr/bin/env python3
"""
Script de diagnóstico completo del sistema de análisis de moda
"""

import os
import sys
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesar imagen para el modelo"""
    try:
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
    except Exception as e:
        logger.error(f"Error preprocesando {image_path}: {e}")
        return None

def test_model_prediction():
    """Probar predicción del modelo"""
    logger.info("🔍 Probando predicción del modelo...")
    
    # Cargar modelo
    model_path = '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5'
    model = tf.keras.models.load_model(model_path)
    
    logger.info(f"✅ Modelo cargado: {model_path}")
    logger.info(f"📊 Output shape: {model.output_shape}")
    logger.info(f"📊 Número de clases: {model.output_shape[1]}")
    
    # Probar con imagen de prueba
    test_image = '/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg'
    if not os.path.exists(test_image):
        logger.error(f"❌ Imagen de prueba no encontrada: {test_image}")
        return None
    
    # Preprocesar imagen
    image = preprocess_image(test_image)
    if image is None:
        logger.error("❌ Error preprocesando imagen")
        return None
    
    image_batch = np.expand_dims(image, axis=0)
    
    # Hacer predicción
    prediction = model.predict(image_batch, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    
    logger.info(f"🎯 Predicción: {prediction}")
    logger.info(f"🎯 Clase predicha: {predicted_class}")
    logger.info(f"🎯 Confianza: {confidence:.4f}")
    logger.info(f"🎯 Probabilidades por clase: {[f'{i}: {p:.4f}' for i, p in enumerate(prediction)]}")
    
    return {
        'prediction': prediction,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'model_output_shape': model.output_shape
    }

def test_clustering():
    """Probar clustering"""
    logger.info("🔍 Probando clustering...")
    
    # Cargar datos de clustering
    clustering_path = '/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl'
    kmeans_path = '/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl'
    
    if not os.path.exists(clustering_path):
        logger.error(f"❌ Archivo de clustering no encontrado: {clustering_path}")
        return None
    
    if not os.path.exists(kmeans_path):
        logger.error(f"❌ Archivo K-means no encontrado: {kmeans_path}")
        return None
    
    with open(clustering_path, 'rb') as f:
        clustering_results = pickle.load(f)
    
    with open(kmeans_path, 'rb') as f:
        kmeans_model = pickle.load(f)
    
    logger.info(f"✅ Datos de clustering cargados")
    logger.info(f"📊 Número de clusters: {kmeans_model.n_clusters}")
    logger.info(f"📊 Dimensiones de centroides: {kmeans_model.cluster_centers_.shape}")
    logger.info(f"📊 Tipo de datos de centroides: {kmeans_model.cluster_centers_.dtype}")
    
    # Probar predicción de cluster
    test_image = '/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg'
    image = preprocess_image(test_image)
    if image is None:
        return None
    
    image_batch = np.expand_dims(image, axis=0)
    
    # Usar el modelo para obtener embedding
    model_path = '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5'
    model = tf.keras.models.load_model(model_path)
    embedding = model.predict(image_batch, verbose=0)[0]
    embedding = embedding.astype(np.float64)
    
    # Predecir cluster
    predicted_cluster = kmeans_model.predict([embedding])[0]
    cluster_center = kmeans_model.cluster_centers_[predicted_cluster]
    distance = np.linalg.norm(embedding - cluster_center)
    
    logger.info(f"🎯 Embedding shape: {embedding.shape}")
    logger.info(f"🎯 Cluster predicho: {predicted_cluster}")
    logger.info(f"🎯 Distancia al centro: {distance:.4f}")
    
    return {
        'predicted_cluster': predicted_cluster,
        'distance': distance,
        'embedding_shape': embedding.shape
    }

def test_complete_pipeline():
    """Probar pipeline completo"""
    logger.info("🔍 Probando pipeline completo...")
    
    # Cargar modelo
    model_path = '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5'
    model = tf.keras.models.load_model(model_path)
    
    # Cargar clustering
    clustering_path = '/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl'
    kmeans_path = '/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl'
    
    with open(clustering_path, 'rb') as f:
        clustering_results = pickle.load(f)
    
    with open(kmeans_path, 'rb') as f:
        kmeans_model = pickle.load(f)
    
    # Procesar imagen
    test_image = '/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg'
    image = preprocess_image(test_image)
    image_batch = np.expand_dims(image, axis=0)
    
    # Predicción de clase
    prediction = model.predict(image_batch, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    class_confidence = float(np.max(prediction))
    
    # Predicción de cluster
    embedding = prediction.astype(np.float64)
    predicted_cluster = kmeans_model.predict([embedding])[0]
    cluster_center = kmeans_model.cluster_centers_[predicted_cluster]
    cluster_distance = np.linalg.norm(embedding - cluster_center)
    cluster_confidence = max(0, 1 - cluster_distance / 10.0)
    
    logger.info("=" * 50)
    logger.info("📊 RESULTADOS DEL PIPELINE COMPLETO")
    logger.info("=" * 50)
    logger.info(f"🎯 Clase predicha: {predicted_class}")
    logger.info(f"🎯 Confianza de clase: {class_confidence:.4f}")
    logger.info(f"🎯 Cluster predicho: {predicted_cluster}")
    logger.info(f"🎯 Distancia al cluster: {cluster_distance:.4f}")
    logger.info(f"🎯 Confianza de cluster: {cluster_confidence:.4f}")
    
    # Verificar rangos
    if predicted_class < 0 or predicted_class > 10:
        logger.warning(f"⚠️  Clase predicha fuera del rango esperado: {predicted_class}")
    
    if predicted_cluster < 0 or predicted_cluster >= kmeans_model.n_clusters:
        logger.warning(f"⚠️  Cluster predicho fuera del rango esperado: {predicted_cluster}")
    
    return {
        'predicted_class': predicted_class,
        'class_confidence': class_confidence,
        'predicted_cluster': predicted_cluster,
        'cluster_confidence': cluster_confidence,
        'cluster_distance': cluster_distance
    }

def main():
    """Función principal"""
    logger.info("🚀 Iniciando diagnóstico completo del sistema...")
    
    # Test 1: Modelo
    logger.info("\n" + "="*50)
    logger.info("TEST 1: PREDICCIÓN DEL MODELO")
    logger.info("="*50)
    model_results = test_model_prediction()
    
    # Test 2: Clustering
    logger.info("\n" + "="*50)
    logger.info("TEST 2: CLUSTERING")
    logger.info("="*50)
    clustering_results = test_clustering()
    
    # Test 3: Pipeline completo
    logger.info("\n" + "="*50)
    logger.info("TEST 3: PIPELINE COMPLETO")
    logger.info("="*50)
    pipeline_results = test_complete_pipeline()
    
    # Resumen final
    logger.info("\n" + "="*50)
    logger.info("RESUMEN FINAL")
    logger.info("="*50)
    
    if model_results:
        logger.info(f"✅ Modelo: Clase {model_results['predicted_class']}, Confianza {model_results['confidence']:.4f}")
    else:
        logger.error("❌ Modelo: Error")
    
    if clustering_results:
        logger.info(f"✅ Clustering: Cluster {clustering_results['predicted_cluster']}, Distancia {clustering_results['distance']:.4f}")
    else:
        logger.error("❌ Clustering: Error")
    
    if pipeline_results:
        logger.info(f"✅ Pipeline: Clase {pipeline_results['predicted_class']}, Cluster {pipeline_results['predicted_cluster']}")
        logger.info(f"✅ Confianzas: Clase {pipeline_results['class_confidence']:.4f}, Cluster {pipeline_results['cluster_confidence']:.4f}")
    else:
        logger.error("❌ Pipeline: Error")

if __name__ == "__main__":
    main()


