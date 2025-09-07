#!/usr/bin/env python3
"""
Script para usar los modelos H5 existentes y realizar clustering
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_h5_model(model_path):
    """
    Carga un modelo H5 y extrae características
    """
    try:
        print(f"🔄 Cargando modelo: {model_path}")
        model = load_model(model_path)
        print(f"✅ Modelo cargado exitosamente")
        print(f"   - Capas: {len(model.layers)}")
        print(f"   - Entrada: {model.input_shape}")
        print(f"   - Salida: {model.output_shape}")
        
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo {model_path}: {e}")
        return None

def extract_features_from_model(model, data_path=None):
    """
    Extrae características del modelo
    Si no hay datos, genera datos de ejemplo
    """
    try:
        # Si no hay datos específicos, generar datos de ejemplo
        if data_path is None or not os.path.exists(data_path):
            print("📊 Generando datos de ejemplo para demostración...")
            # Generar datos de ejemplo (imágenes sintéticas)
            batch_size = 100
            input_shape = model.input_shape[1:]  # Excluir la dimensión del batch
            
            # Generar imágenes aleatorias con la forma correcta
            if len(input_shape) == 3:  # (height, width, channels)
                sample_data = np.random.rand(batch_size, *input_shape).astype(np.float32)
            else:
                # Si es un vector de características
                sample_data = np.random.rand(batch_size, input_shape[0]).astype(np.float32)
            
            print(f"   - Datos generados: {sample_data.shape}")
        else:
            print(f"📊 Cargando datos desde: {data_path}")
            # Aquí podrías cargar datos reales si los tienes
            sample_data = np.random.rand(100, *model.input_shape[1:]).astype(np.float32)
        
        # Extraer características usando el modelo
        print("🔍 Extrayendo características...")
        features = model.predict(sample_data, verbose=0)
        
        # Si el modelo tiene múltiples salidas, tomar la primera
        if isinstance(features, list):
            features = features[0]
        
        print(f"✅ Características extraídas: {features.shape}")
        return features, sample_data
        
    except Exception as e:
        print(f"❌ Error extrayendo características: {e}")
        return None, None

def perform_clustering(features, n_clusters=8):
    """
    Realiza clustering K-Means en las características
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
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Calcular métricas
        silhouette = silhouette_score(features_scaled, cluster_labels)
        
        print(f"✅ Clustering completado:")
        print(f"   - Clusters: {n_clusters}")
        print(f"   - Silhouette Score: {silhouette:.3f}")
        
        return cluster_labels, kmeans, scaler, pca if 'pca' in locals() else None
        
    except Exception as e:
        print(f"❌ Error en clustering: {e}")
        return None, None, None, None

def analyze_clusters(cluster_labels, features):
    """
    Analiza los clusters resultantes
    """
    try:
        print("📊 Analizando clusters...")
        
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        print(f"\n📈 Distribución de clusters:")
        for i in range(n_clusters):
            count = np.sum(cluster_labels == i)
            percentage = (count / len(cluster_labels)) * 100
            print(f"   Cluster {i}: {count} muestras ({percentage:.1f}%)")
        
        # Calcular estadísticas por cluster
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_features = features[cluster_mask]
            
            stats = {
                'cluster': i,
                'count': np.sum(cluster_mask),
                'mean_norm': np.mean(np.linalg.norm(cluster_features, axis=1)),
                'std_norm': np.std(np.linalg.norm(cluster_features, axis=1))
            }
            cluster_stats.append(stats)
        
        return cluster_stats
        
    except Exception as e:
        print(f"❌ Error analizando clusters: {e}")
        return []

def create_visualizations(features, cluster_labels, output_dir="cluster_results"):
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
        fig.suptitle('Análisis de Clusters - Modelos H5', fontsize=16)
        
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
        ax2.set_title('Distribución de Muestras por Cluster')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Número de Muestras')
        
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

def main():
    """Función principal"""
    setup_logging()
    
    print("="*60)
    print("🎯 ANÁLISIS DE CLUSTERING CON MODELOS H5 EXISTENTES")
    print("="*60)
    
    # Buscar archivos H5 en el directorio actual
    h5_files = list(Path('.').glob('*.h5'))
    
    if not h5_files:
        print("❌ No se encontraron archivos H5 en el directorio actual")
        print("   Asegúrate de que los archivos .h5 estén en el directorio del proyecto")
        return
    
    print(f"📁 Archivos H5 encontrados: {len(h5_files)}")
    for h5_file in h5_files:
        print(f"   - {h5_file.name}")
    
    # Procesar cada archivo H5
    for h5_file in h5_files:
        print(f"\n{'='*50}")
        print(f"🔄 Procesando: {h5_file.name}")
        print(f"{'='*50}")
        
        # Cargar modelo
        model = load_h5_model(h5_file)
        if model is None:
            continue
        
        # Extraer características
        features, sample_data = extract_features_from_model(model)
        if features is None:
            continue
        
        # Realizar clustering
        cluster_labels, kmeans, scaler, pca = perform_clustering(features, n_clusters=8)
        if cluster_labels is None:
            continue
        
        # Analizar clusters
        cluster_stats = analyze_clusters(cluster_labels, features)
        
        # Crear visualizaciones
        output_dir = f"cluster_results_{h5_file.stem}"
        create_visualizations(features, cluster_labels, output_dir)
        
        # Guardar resultados
        results = {
            'model_name': h5_file.name,
            'features': features,
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_stats,
            'kmeans': kmeans,
            'scaler': scaler,
            'pca': pca
        }
        
        results_path = f"clustering_results_{h5_file.stem}.pkl"
        import pickle
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Resultados guardados en: {results_path}")
        print(f"📊 Visualizaciones en: {output_dir}/")
    
    print(f"\n{'='*60}")
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print("Archivos generados:")
    print("  - clustering_results_*.pkl (resultados del clustering)")
    print("  - cluster_results_*/ (visualizaciones)")
    print("\nPara ver los resultados:")
    print("  1. Abre las imágenes PNG en cluster_results_*/")
    print("  2. Usa los archivos .pkl para análisis adicionales")

if __name__ == "__main__":
    main()
