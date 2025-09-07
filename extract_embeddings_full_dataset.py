#!/usr/bin/env python3
"""
Script para extraer embeddings de todo el dataset de entrenamiento
usando el modelo correcto mobilenet_v2_final.h5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Configuración
MODEL_PATH = '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5'
TRAIN_PATH = '/home/jose/PreditorIA2025/data/processed/train'
OUTPUT_DIR = '/home/jose/PreditorIA2025/clustering_results'
N_CLUSTERS = 150
IMG_SIZE = (224, 224)

def load_model():
    """Cargar el modelo entrenado"""
    print("Cargando modelo MobileNetV2...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente: {model.input_shape}")
    return model

def get_image_paths():
    """Obtener todas las rutas de imágenes del dataset de entrenamiento"""
    print("Obteniendo rutas de imágenes...")
    image_paths = []
    categories = []
    
    for category in os.listdir(TRAIN_PATH):
        category_path = os.path.join(TRAIN_PATH, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_path, img_file)
                    image_paths.append(img_path)
                    categories.append(category)
    
    print(f"Total de imágenes encontradas: {len(image_paths)}")
    print(f"Categorías: {set(categories)}")
    return image_paths, categories

def extract_embeddings(model, image_paths, batch_size=32):
    """Extraer embeddings de todas las imágenes"""
    print("Extrayendo embeddings...")
    embeddings = []
    failed_images = []
    
    # Procesar en lotes para eficiencia
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                # Cargar y preprocesar imagen
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array[0])
            except Exception as e:
                print(f"Error cargando imagen {img_path}: {e}")
                failed_images.append(img_path)
                continue
        
        if batch_images:
            # Procesar lote
            batch_array = np.array(batch_images)
            batch_embeddings = model.predict(batch_array, verbose=0)
            embeddings.extend(batch_embeddings)
        
        if (i // batch_size + 1) % 50 == 0:
            print(f"Procesadas {i + len(batch_paths)}/{len(image_paths)} imágenes")
    
    print(f"Embeddings extraídos: {len(embeddings)}")
    print(f"Imágenes fallidas: {len(failed_images)}")
    
    return np.array(embeddings), failed_images

def perform_clustering(embeddings, n_clusters=150):
    """Realizar clustering K-means"""
    print(f"Realizando clustering con {n_clusters} clusters...")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calcular métricas de calidad
    silhouette = silhouette_score(embeddings, cluster_labels)
    calinski = calinski_harabasz_score(embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    
    return kmeans, cluster_labels, {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies_bouldin
    }

def analyze_clusters(cluster_labels, image_paths, categories):
    """Analizar la distribución y calidad de los clusters"""
    print("Analizando clusters...")
    
    # Distribución de clusters
    cluster_counts = Counter(cluster_labels)
    
    # Análisis por categoría
    category_cluster_analysis = {}
    for category in set(categories):
        category_indices = [i for i, cat in enumerate(categories) if cat == category]
        category_clusters = [cluster_labels[i] for i in category_indices]
        category_cluster_counts = Counter(category_clusters)
        category_cluster_analysis[category] = dict(category_cluster_counts)
    
    return cluster_counts, category_cluster_analysis

def create_visualizations(cluster_counts, category_cluster_analysis, metrics):
    """Crear visualizaciones del clustering"""
    print("Creando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribución de clusters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histograma de distribución de clusters
    axes[0, 0].hist(cluster_counts.values(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribución del Tamaño de Clusters')
    axes[0, 0].set_xlabel('Número de Imágenes por Cluster')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top 20 clusters más grandes
    top_clusters = dict(sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    axes[0, 1].bar(range(len(top_clusters)), list(top_clusters.values()))
    axes[0, 1].set_title('Top 20 Clusters por Tamaño')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Número de Imágenes')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Análisis por categoría
    category_totals = {cat: sum(clusters.values()) for cat, clusters in category_cluster_analysis.items()}
    axes[1, 0].pie(category_totals.values(), labels=category_totals.keys(), autopct='%1.1f%%')
    axes[1, 0].set_title('Distribución por Categoría')
    
    # Métricas de calidad
    metrics_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
    metrics_values = [metrics['silhouette'], metrics['calinski_harabasz'], metrics['davies_bouldin']]
    axes[1, 1].bar(metrics_names, metrics_values)
    axes[1, 1].set_title('Métricas de Calidad del Clustering')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Análisis detallado por categoría
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Crear heatmap de categorías vs clusters
    categories = list(category_cluster_analysis.keys())
    cluster_ids = sorted(set().union(*[clusters.keys() for clusters in category_cluster_analysis.values()]))
    
    # Crear matriz de datos
    heatmap_data = []
    for category in categories:
        row = []
        for cluster_id in cluster_ids:
            count = category_cluster_analysis[category].get(cluster_id, 0)
            row.append(count)
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # Crear heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=cluster_ids, 
                yticklabels=categories,
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                ax=ax)
    
    ax.set_title('Distribución de Categorías por Cluster')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Categoría')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(kmeans, cluster_labels, image_paths, categories, metrics, cluster_counts, category_cluster_analysis):
    """Guardar todos los resultados"""
    print("Guardando resultados...")
    
    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Guardar modelo de clustering
    with open(os.path.join(OUTPUT_DIR, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Guardar resultados completos
    results = {
        'cluster_labels': cluster_labels,
        'image_paths': image_paths,
        'categories': categories,
        'metrics': metrics,
        'cluster_counts': dict(cluster_counts),
        'category_cluster_analysis': category_cluster_analysis,
        'n_clusters': N_CLUSTERS,
        'total_images': len(image_paths)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'clustering_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Guardar reporte de texto
    with open(os.path.join(OUTPUT_DIR, 'clustering_report.txt'), 'w') as f:
        f.write("REPORTE DE CLUSTERING - DATASET COMPLETO\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total de imágenes: {len(image_paths)}\n")
        f.write(f"Número de clusters: {N_CLUSTERS}\n")
        f.write(f"Dimensiones de características: {kmeans.cluster_centers_.shape[1]}\n\n")
        
        f.write("MÉTRICAS DE CALIDAD:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Silhouette Score: {metrics['silhouette']:.4f}\n")
        f.write(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.4f}\n")
        f.write(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}\n\n")
        
        f.write("DISTRIBUCIÓN POR CLUSTER (Top 20):\n")
        f.write("-" * 40 + "\n")
        top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for cluster_id, count in top_clusters:
            percentage = (count / len(image_paths)) * 100
            f.write(f"Cluster {cluster_id}: {count} imágenes ({percentage:.1f}%)\n")
        
        f.write(f"\nDISTRIBUCIÓN POR CATEGORÍA:\n")
        f.write("-" * 30 + "\n")
        category_totals = {cat: sum(clusters.values()) for cat, clusters in category_cluster_analysis.items()}
        for category, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            percentage = (total / len(image_paths)) * 100
            f.write(f"{category}: {total} imágenes ({percentage:.1f}%)\n")

def main():
    """Función principal"""
    print("=" * 60)
    print("EXTRACCIÓN DE EMBEDDINGS Y CLUSTERING - DATASET COMPLETO")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Cargar modelo
    model = load_model()
    
    # 2. Obtener rutas de imágenes
    image_paths, categories = get_image_paths()
    
    # 3. Extraer embeddings
    embeddings, failed_images = extract_embeddings(model, image_paths)
    
    if len(embeddings) == 0:
        print("Error: No se pudieron extraer embeddings de ninguna imagen")
        return
    
    # 4. Realizar clustering
    kmeans, cluster_labels, metrics = perform_clustering(embeddings, N_CLUSTERS)
    
    # 5. Analizar clusters
    cluster_counts, category_cluster_analysis = analyze_clusters(cluster_labels, image_paths, categories)
    
    # 6. Crear visualizaciones
    create_visualizations(cluster_counts, category_cluster_analysis, metrics)
    
    # 7. Guardar resultados
    save_results(kmeans, cluster_labels, image_paths, categories, metrics, cluster_counts, category_cluster_analysis)
    
    end_time = time.time()
    print(f"\nProceso completado en {end_time - start_time:.2f} segundos")
    print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


