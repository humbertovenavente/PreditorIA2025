#!/usr/bin/env python3
"""
Script para analizar la calidad del clustering y verificar que agrupe estilos similares
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def load_clustering_results():
    """Cargar resultados del clustering"""
    try:
        with open('clustering_results/clustering_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("✅ Resultados de clustering cargados exitosamente")
        return results
    except Exception as e:
        print(f"❌ Error cargando clustering: {e}")
        return None

def analyze_cluster_quality(results):
    """Analizar la calidad del clustering"""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE CALIDAD DEL CLUSTERING")
    print("="*60)
    
    # Información básica
    labels = results['cluster_labels']
    features = results['features']
    image_paths = results['image_paths']
    n_clusters = len(set(labels))
    
    print(f"📈 Número de clusters: {n_clusters}")
    print(f"📸 Total de imágenes: {len(labels)}")
    print(f"🔢 Dimensiones de características: {features.shape}")
    
    # Distribución de clusters
    cluster_counts = Counter(labels)
    print(f"\n📊 Distribución de clusters:")
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        percentage = (count / len(labels)) * 100
        print(f"   Cluster {cluster_id}: {count} imágenes ({percentage:.1f}%)")
    
    # Métricas de calidad
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"\n📏 Métricas de calidad:")
        print(f"   Silhouette Score: {metrics.get('silhouette', 'N/A'):.3f}")
        print(f"   Calinski-Harabasz: {metrics.get('calinski', 'N/A'):.3f}")
        print(f"   Davies-Bouldin: {metrics.get('davies_bouldin', 'N/A'):.3f}")
    
    return cluster_counts, n_clusters

def analyze_cluster_coherence(results):
    """Analizar coherencia de clusters basándose en categorías"""
    print("\n" + "="*60)
    print("🎯 ANÁLISIS DE COHERENCIA DE CLUSTERS")
    print("="*60)
    
    labels = results['cluster_labels']
    image_paths = results['image_paths']
    
    # Extraer categorías de las rutas de archivos
    categories = []
    for path in image_paths:
        # Extraer categoría del path (ej: /path/to/category/image.jpg)
        parts = path.split('/')
        if len(parts) >= 2:
            category = parts[-2]  # Asumiendo que la categoría está en el penúltimo directorio
            categories.append(category)
        else:
            categories.append('unknown')
    
    # Analizar coherencia por cluster
    n_clusters = len(set(labels))
    cluster_coherence = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = np.array(labels) == cluster_id
        cluster_categories = [categories[i] for i in range(len(categories)) if cluster_mask[i]]
        
        if cluster_categories:
            category_counts = Counter(cluster_categories)
            most_common = category_counts.most_common(1)[0]
            coherence = most_common[1] / len(cluster_categories)
            
            cluster_coherence[cluster_id] = {
                'coherence': coherence,
                'dominant_category': most_common[0],
                'category_distribution': dict(category_counts)
            }
            
            print(f"\n🔍 Cluster {cluster_id}:")
            print(f"   Coherencia: {coherence:.2f}")
            print(f"   Categoría dominante: {most_common[0]} ({most_common[1]}/{len(cluster_categories)})")
            print(f"   Distribución: {dict(category_counts)}")
    
    return cluster_coherence

def identify_trendy_clusters(results, cluster_coherence):
    """Identificar qué clusters representan tendencias de moda"""
    print("\n" + "="*60)
    print("🌟 IDENTIFICACIÓN DE CLUSTERS TENDENCIOSOS")
    print("="*60)
    
    # Criterios para determinar si un cluster es "tendencia":
    # 1. Alta coherencia (más del 80% de la misma categoría)
    # 2. Tamaño moderado (no muy pequeño, no muy grande)
    # 3. Categorías que suelen ser tendencia o general (versátil)
    
    trendy_categories = ['dress', 'tops', 'shirts', 'jeans', 'pants', 'shoes', 'general']
    
    trendy_clusters = []
    for cluster_id, info in cluster_coherence.items():
        coherence = info['coherence']
        dominant_category = info['dominant_category']
        cluster_size = len([l for l in results['cluster_labels'] if l == cluster_id])
        
        # Criterios más flexibles para identificar tendencias
        is_trendy = (
            coherence > 0.8 and  # Alta coherencia (80%+)
            cluster_size >= 20 and  # Tamaño mínimo (20+ imágenes)
            cluster_size <= 300 and  # Tamaño máximo (300- imágenes)
            dominant_category in trendy_categories  # Categoría tendenciosa o general
        )
        
        if is_trendy:
            trendy_clusters.append(cluster_id)
            print(f"✅ Cluster {cluster_id}: TENDENCIA")
            print(f"   Categoría: {dominant_category}")
            print(f"   Coherencia: {coherence:.2f}")
            print(f"   Tamaño: {cluster_size} imágenes")
        else:
            print(f"❌ Cluster {cluster_id}: No tendencia")
            print(f"   Categoría: {dominant_category}")
            print(f"   Coherencia: {coherence:.2f}")
            print(f"   Tamaño: {cluster_size} imágenes")
    
    print(f"\n🎯 Clusters identificados como tendencia: {trendy_clusters}")
    return trendy_clusters

def create_clustering_visualization(results, cluster_coherence, trendy_clusters):
    """Crear visualizaciones del clustering"""
    print("\n" + "="*60)
    print("📊 CREANDO VISUALIZACIONES")
    print("="*60)
    
    # Crear directorio para visualizaciones
    os.makedirs('clustering_analysis', exist_ok=True)
    
    # 1. Distribución de clusters
    plt.figure(figsize=(12, 6))
    cluster_counts = Counter(results['cluster_labels'])
    clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]
    colors = ['green' if c in trendy_clusters else 'lightblue' for c in clusters]
    
    plt.bar(clusters, counts, color=colors)
    plt.xlabel('Cluster ID')
    plt.ylabel('Número de Imágenes')
    plt.title('Distribución de Clusters (Verde = Tendencia)')
    plt.grid(True, alpha=0.3)
    
    for i, (cluster, count) in enumerate(zip(clusters, counts)):
        plt.text(cluster, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis/cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Coherencia por cluster
    plt.figure(figsize=(12, 6))
    cluster_ids = list(cluster_coherence.keys())
    coherences = [cluster_coherence[cid]['coherence'] for cid in cluster_ids]
    colors = ['green' if cid in trendy_clusters else 'lightblue' for cid in cluster_ids]
    
    plt.bar(cluster_ids, coherences, color=colors)
    plt.xlabel('Cluster ID')
    plt.ylabel('Coherencia')
    plt.title('Coherencia por Cluster (Verde = Tendencia)')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Umbral de Tendencia (0.7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, (cluster, coh) in enumerate(zip(cluster_ids, coherences)):
        plt.text(cluster, coh + 0.01, f'{coh:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis/cluster_coherence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizaciones guardadas en 'clustering_analysis/'")

def main():
    """Función principal"""
    print("🔍 ANÁLISIS DE CALIDAD DEL CLUSTERING")
    print("="*60)
    
    # Cargar resultados
    results = load_clustering_results()
    if results is None:
        return
    
    # Analizar calidad
    cluster_counts, n_clusters = analyze_cluster_quality(results)
    
    # Analizar coherencia
    cluster_coherence = analyze_cluster_coherence(results)
    
    # Identificar clusters tendenciosos
    trendy_clusters = identify_trendy_clusters(results, cluster_coherence)
    
    # Crear visualizaciones
    create_clustering_visualization(results, cluster_coherence, trendy_clusters)
    
    # Guardar análisis
    analysis_results = {
        'trendy_clusters': trendy_clusters,
        'cluster_coherence': cluster_coherence,
        'cluster_counts': dict(cluster_counts)
    }
    
    with open('clustering_analysis/trendy_clusters.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print(f"\n✅ Análisis completado!")
    print(f"📁 Resultados guardados en 'clustering_analysis/trendy_clusters.pkl'")
    print(f"🎯 Clusters identificados como tendencia: {trendy_clusters}")

if __name__ == "__main__":
    main()
