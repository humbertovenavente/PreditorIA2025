#!/usr/bin/env python3
"""
Script para realizar clustering usando el modelo entrenado y los datos procesados
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image
import glob

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_trained_model(model_path):
    """
    Carga el modelo entrenado
    """
    try:
        print(f"🔄 Cargando modelo entrenado: {model_path}")
        model = load_model(model_path)
        print(f"✅ Modelo cargado exitosamente")
        print(f"   - Capas: {len(model.layers)}")
        print(f"   - Entrada: {model.input_shape}")
        print(f"   - Salida: {model.output_shape}")
        
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo {model_path}: {e}")
        return None

def load_processed_images(data_dir, max_images=1000):
    """
    Carga las imágenes procesadas del dataset
    """
    try:
        print(f"📁 Cargando imágenes desde: {data_dir}")
        
        # Buscar todas las imágenes en los subdirectorios
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        # Limitar número de imágenes si es necesario
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
            print(f"   - Limitando a {max_images} imágenes")
        
        print(f"   - Encontradas {len(image_paths)} imágenes")
        
        return image_paths
        
    except Exception as e:
        print(f"❌ Error cargando imágenes: {e}")
        return []

def preprocess_image_for_model(image_path, target_size=(224, 224)):
    """
    Preprocesa una imagen para el modelo
    """
    try:
        # Cargar imagen
        img = Image.open(image_path)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize(target_size)
        
        # Convertir a array
        img_array = image.img_to_array(img)
        
        # Normalizar (0-1)
        img_array = img_array / 255.0
        
        # Expandir dimensiones para batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error preprocesando imagen {image_path}: {e}")
        return None

def extract_features_from_model(model, image_paths, batch_size=32):
    """
    Extrae características usando el modelo entrenado
    """
    try:
        print(f"🔍 Extrayendo características de {len(image_paths)} imágenes...")
        
        # Crear modelo para extraer características (penúltima capa)
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output  # Penúltima capa
        )
        
        features_list = []
        valid_paths = []
        
        # Procesar en lotes
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []
            
            for path in batch_paths:
                if os.path.exists(path):
                    img_array = preprocess_image_for_model(path)
                    if img_array is not None:
                        # Extraer características
                        features = feature_extractor.predict(img_array, verbose=0)
                        batch_features.append(features.flatten())
                        valid_paths.append(path)
            
            if batch_features:
                features_list.extend(batch_features)
            
            # Log progreso
            if (i + batch_size) % (batch_size * 5) == 0:
                print(f"   - Procesadas {min(i + batch_size, len(image_paths))}/{len(image_paths)} imágenes")
        
        if not features_list:
            print("❌ No se pudieron extraer características de ninguna imagen")
            return None, []
        
        features_matrix = np.array(features_list)
        print(f"✅ Características extraídas: {features_matrix.shape}")
        
        return features_matrix, valid_paths
        
    except Exception as e:
        print(f"❌ Error extrayendo características: {e}")
        return None, []

def find_optimal_clusters(features, max_clusters=20):
    """
    Encuentra el número óptimo de clusters
    """
    try:
        print("🎯 Buscando número óptimo de clusters...")
        
        k_range = range(2, min(max_clusters + 1, len(features) // 2))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            silhouette = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette)
            print(f"   - K={k}: Silhouette={silhouette:.3f}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        print(f"✅ K óptimo encontrado: {optimal_k} (Silhouette: {best_silhouette:.3f})")
        
        return optimal_k, silhouette_scores
        
    except Exception as e:
        print(f"❌ Error encontrando clusters óptimos: {e}")
        return 8, []

def perform_clustering(features, n_clusters=8):
    """
    Realiza clustering K-Means
    """
    try:
        print(f"🎯 Realizando clustering con {n_clusters} clusters...")
        
        # Estandarizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Aplicar PCA si hay muchas dimensiones
        if features_scaled.shape[1] > 50:
            print("📉 Aplicando PCA para reducción de dimensionalidad...")
            pca = PCA(n_components=50, random_state=42)
            features_scaled = pca.fit_transform(features_scaled)
            print(f"   - Componentes PCA: {features_scaled.shape[1]}")
            print(f"   - Varianza explicada: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            pca = None
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calcular métricas
        silhouette = silhouette_score(features_scaled, cluster_labels)
        calinski = calinski_harabasz_score(features_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
        
        print(f"✅ Clustering completado:")
        print(f"   - Silhouette Score: {silhouette:.3f}")
        print(f"   - Calinski-Harabasz Score: {calinski:.3f}")
        print(f"   - Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        return cluster_labels, kmeans, scaler, pca, {
            'silhouette': silhouette,
            'calinski': calinski,
            'davies_bouldin': davies_bouldin
        }
        
    except Exception as e:
        print(f"❌ Error en clustering: {e}")
        return None, None, None, None, {}

def analyze_clusters(cluster_labels, image_paths):
    """
    Analiza los clusters resultantes
    """
    try:
        print("📊 Analizando clusters...")
        
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        print(f"\n📈 Distribución de clusters:")
        cluster_stats = []
        
        for i in range(n_clusters):
            count = np.sum(cluster_labels == i)
            percentage = (count / len(cluster_labels)) * 100
            print(f"   Cluster {i}: {count} imágenes ({percentage:.1f}%)")
            
            # Obtener algunas imágenes de ejemplo
            cluster_images = [path for j, path in enumerate(image_paths) if cluster_labels[j] == i]
            sample_images = cluster_images[:5]  # Primeras 5 imágenes
            
            cluster_stats.append({
                'cluster_id': i,
                'count': count,
                'percentage': percentage,
                'sample_images': sample_images
            })
        
        return cluster_stats
        
    except Exception as e:
        print(f"❌ Error analizando clusters: {e}")
        return []

def create_visualizations(features, cluster_labels, image_paths, output_dir="clustering_results"):
    """
    Crea visualizaciones de los clusters
    """
    try:
        print("🎨 Creando visualizaciones...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Reducir a 2D para visualización
        if features.shape[1] > 2:
            pca_2d = PCA(n_components=2, random_state=42)
            features_2d = pca_2d.fit_transform(features)
        else:
            features_2d = features
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Clusters - Modelo Entrenado', fontsize=16)
        
        # 1. Scatter plot de clusters
        ax1 = axes[0, 0]
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7)
        ax1.set_title('Distribución de Clusters')
        ax1.set_xlabel('Componente Principal 1')
        ax1.set_ylabel('Componente Principal 2')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Distribución de clusters
        ax2 = axes[0, 1]
        unique, counts = np.unique(cluster_labels, return_counts=True)
        bars = ax2.bar(unique, counts, color=plt.cm.tab10(unique))
        ax2.set_title('Distribución de Imágenes por Cluster')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Número de Imágenes')
        
        # Añadir valores en las barras
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # 3. Histograma de distancias
        ax3 = axes[1, 0]
        distances = []
        for i, label in enumerate(cluster_labels):
            cluster_center = np.mean(features[cluster_labels == label], axis=0)
            dist = np.linalg.norm(features[i] - cluster_center)
            distances.append(dist)
        
        ax3.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribución de Distancias al Centroide')
        ax3.set_xlabel('Distancia al Centroide')
        ax3.set_ylabel('Frecuencia')
        
        # 4. Métricas de calidad
        ax4 = axes[1, 1]
        silhouette = silhouette_score(features, cluster_labels)
        
        metrics = ['Silhouette Score']
        values = [silhouette]
        colors = ['green' if silhouette > 0.5 else 'orange' if silhouette > 0.3 else 'red']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Métricas de Calidad del Clustering')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Guardar visualización
        output_path = os.path.join(output_dir, 'cluster_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualización guardada en: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creando visualizaciones: {e}")

def save_results(features, cluster_labels, image_paths, cluster_stats, metrics, output_dir="clustering_results"):
    """
    Guarda los resultados del clustering
    """
    try:
        print("💾 Guardando resultados...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Guardar datos en formato pickle
        import pickle
        results = {
            'features': features,
            'cluster_labels': cluster_labels,
            'image_paths': image_paths,
            'cluster_stats': cluster_stats,
            'metrics': metrics
        }
        
        results_path = os.path.join(output_dir, 'clustering_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Guardar reporte en texto
        report_path = os.path.join(output_dir, 'clustering_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE CLUSTERING - MODELO ENTRENADO\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total de imágenes: {len(image_paths)}\n")
            f.write(f"Número de clusters: {len(np.unique(cluster_labels))}\n")
            f.write(f"Dimensiones de características: {features.shape[1]}\n\n")
            
            f.write("MÉTRICAS DE CALIDAD:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Silhouette Score: {metrics.get('silhouette', 0):.4f}\n")
            f.write(f"Calinski-Harabasz Score: {metrics.get('calinski', 0):.4f}\n")
            f.write(f"Davies-Bouldin Score: {metrics.get('davies_bouldin', 0):.4f}\n\n")
            
            f.write("DISTRIBUCIÓN POR CLUSTER:\n")
            f.write("-" * 30 + "\n")
            for stat in cluster_stats:
                f.write(f"Cluster {stat['cluster_id']}:\n")
                f.write(f"  Imágenes: {stat['count']} ({stat['percentage']:.1f}%)\n")
                f.write(f"  Muestra de imágenes:\n")
                for img_path in stat['sample_images']:
                    f.write(f"    - {img_path}\n")
                f.write("\n")
        
        print(f"✅ Resultados guardados en: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error guardando resultados: {e}")

def main():
    """Función principal"""
    setup_logging()
    
    print("="*60)
    print("🎯 CLUSTERING CON MODELO ENTRENADO")
    print("="*60)
    
    # Rutas de archivos
    model_path = "data/logs/training/mobilenet_v2_final.h5"
    data_dir = "data/processed/train"  # Usar datos de entrenamiento
    max_images = 500  # Limitar para prueba rápida
    
    # 1. Cargar modelo entrenado
    model = load_trained_model(model_path)
    if model is None:
        return
    
    # 2. Cargar imágenes procesadas
    image_paths = load_processed_images(data_dir, max_images)
    if not image_paths:
        print("❌ No se encontraron imágenes procesadas")
        return
    
    # 3. Extraer características
    features, valid_paths = extract_features_from_model(model, image_paths)
    if features is None:
        return
    
    # 4. Buscar número óptimo de clusters
    optimal_k, silhouette_scores = find_optimal_clusters(features, max_clusters=15)
    
    # 5. Realizar clustering
    cluster_labels, kmeans, scaler, pca, metrics = perform_clustering(features, optimal_k)
    if cluster_labels is None:
        return
    
    # 6. Analizar clusters
    cluster_stats = analyze_clusters(cluster_labels, valid_paths)
    
    # 7. Crear visualizaciones
    create_visualizations(features, cluster_labels, valid_paths)
    
    # 8. Guardar resultados
    save_results(features, cluster_labels, valid_paths, cluster_stats, metrics)
    
    print(f"\n{'='*60}")
    print("✅ CLUSTERING COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("Archivos generados:")
    print("  - clustering_results/clustering_results.pkl")
    print("  - clustering_results/clustering_report.txt")
    print("  - clustering_results/cluster_analysis.png")
    print("\nPara ver los resultados:")
    print("  1. Revisa el archivo 'clustering_report.txt'")
    print("  2. Mira la visualización 'cluster_analysis.png'")
    print("  3. Usa el archivo .pkl para análisis adicionales")

if __name__ == "__main__":
    main()
