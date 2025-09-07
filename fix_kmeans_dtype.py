#!/usr/bin/env python3
"""
Script para corregir el tipo de datos del modelo K-means
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans

def fix_kmeans_dtype():
    """Corregir el tipo de datos del modelo K-means"""
    print("🔧 Corrigiendo tipo de datos del modelo K-means...")
    
    # Cargar modelo original
    with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl', 'rb') as f:
        kmeans_original = pickle.load(f)
    
    print(f"Modelo original - centros dtype: {kmeans_original.cluster_centers_.dtype}")
    print(f"Modelo original - centros shape: {kmeans_original.cluster_centers_.shape}")
    
    # Crear nuevo modelo con tipo de datos correcto
    kmeans_fixed = KMeans(
        n_clusters=kmeans_original.n_clusters,
        random_state=kmeans_original.random_state,
        n_init=kmeans_original.n_init,
        max_iter=kmeans_original.max_iter,
        tol=kmeans_original.tol,
        copy_x=kmeans_original.copy_x,
        algorithm=kmeans_original.algorithm
    )
    
    # Convertir centros a float64
    centers_float64 = kmeans_original.cluster_centers_.astype(np.float64)
    kmeans_fixed.cluster_centers_ = centers_float64
    kmeans_fixed.labels_ = kmeans_original.labels_.astype(np.int32)
    kmeans_fixed.inertia_ = float(kmeans_original.inertia_)
    kmeans_fixed.n_iter_ = kmeans_original.n_iter_
    kmeans_fixed.n_features_in_ = kmeans_original.n_features_in_
    kmeans_fixed.feature_names_in_ = kmeans_original.feature_names_in_
    
    print(f"Modelo corregido - centros dtype: {kmeans_fixed.cluster_centers_.dtype}")
    print(f"Modelo corregido - centros shape: {kmeans_fixed.cluster_centers_.shape}")
    
    # Guardar modelo corregido
    with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model_fixed.pkl', 'wb') as f:
        pickle.dump(kmeans_fixed, f)
    
    print("✅ Modelo K-means corregido guardado")
    
    # Probar el modelo corregido
    test_embedding = np.random.random((1, 11)).astype(np.float64)
    try:
        cluster = kmeans_fixed.predict(test_embedding)[0]
        print(f"✅ Prueba exitosa - cluster predicho: {cluster}")
    except Exception as e:
        print(f"❌ Error en prueba: {e}")

if __name__ == "__main__":
    fix_kmeans_dtype()


