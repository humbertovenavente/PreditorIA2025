#!/usr/bin/env python3
"""
Script para actualizar la aplicación con los resultados correctos del clustering
"""

import pickle
import numpy as np
import os
import shutil
from pathlib import Path

def load_clustering_results():
    """Cargar los resultados del clustering correcto"""
    results_path = '/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl'
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results

def update_fashion_trend_app():
    """Actualizar la aplicación con los nuevos resultados"""
    print("Actualizando Fashion Trend App...")
    
    # Cargar resultados
    results = load_clustering_results()
    
    # Crear directorio de modelos en la app si no existe
    models_dir = '/home/jose/PreditorIA2025/fashion_trend_app/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Copiar modelo de clustering
    kmeans_src = '/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl'
    kmeans_dst = '/home/jose/PreditorIA2025/fashion_trend_app/models/kmeans_model.pkl'
    shutil.copy2(kmeans_src, kmeans_dst)
    
    # Copiar resultados completos
    results_src = '/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl'
    results_dst = '/home/jose/PreditorIA2025/fashion_trend_app/models/clustering_results.pkl'
    shutil.copy2(results_src, results_dst)
    
    # Copiar visualizaciones
    viz_src = '/home/jose/PreditorIA2025/clustering_results/cluster_analysis.png'
    viz_dst = '/home/jose/PreditorIA2025/fashion_trend_app/static/images/cluster_analysis.png'
    os.makedirs(os.path.dirname(viz_dst), exist_ok=True)
    shutil.copy2(viz_src, viz_dst)
    
    viz_src2 = '/home/jose/PreditorIA2025/clustering_results/category_analysis.png'
    viz_dst2 = '/home/jose/PreditorIA2025/fashion_trend_app/static/images/category_analysis.png'
    shutil.copy2(viz_src2, viz_dst2)
    
    print("✅ Fashion Trend App actualizada")

def create_cluster_summary():
    """Crear un resumen de los clusters para análisis"""
    print("Creando resumen de clusters...")
    
    results = load_clustering_results()
    
    # Crear resumen por cluster
    cluster_summary = {}
    cluster_counts = results['cluster_counts']
    category_analysis = results['category_cluster_analysis']
    
    for cluster_id in range(150):
        if cluster_id in cluster_counts:
            count = cluster_counts[cluster_id]
            
            # Encontrar la categoría dominante en este cluster
            dominant_category = None
            max_category_count = 0
            
            for category, clusters in category_analysis.items():
                if cluster_id in clusters:
                    if clusters[cluster_id] > max_category_count:
                        max_category_count = clusters[cluster_id]
                        dominant_category = category
            
            cluster_summary[cluster_id] = {
                'size': count,
                'dominant_category': dominant_category,
                'category_percentage': (max_category_count / count * 100) if count > 0 else 0
            }
    
    # Guardar resumen
    summary_path = '/home/jose/PreditorIA2025/clustering_results/cluster_summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(cluster_summary, f)
    
    # Crear reporte de texto del resumen
    report_path = '/home/jose/PreditorIA2025/clustering_results/cluster_summary.txt'
    with open(report_path, 'w') as f:
        f.write("RESUMEN DE CLUSTERS - DATASET COMPLETO\n")
        f.write("=" * 50 + "\n\n")
        
        # Top 20 clusters más grandes
        f.write("TOP 20 CLUSTERS MÁS GRANDES:\n")
        f.write("-" * 40 + "\n")
        sorted_clusters = sorted(cluster_summary.items(), 
                               key=lambda x: x[1]['size'], reverse=True)[:20]
        
        for cluster_id, info in sorted_clusters:
            f.write(f"Cluster {cluster_id:3d}: {info['size']:4d} imágenes "
                   f"({info['size']/16959*100:5.1f}%) - "
                   f"Categoría dominante: {info['dominant_category']} "
                   f"({info['category_percentage']:5.1f}%)\n")
        
        f.write(f"\nESTADÍSTICAS GENERALES:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total de clusters: {len(cluster_summary)}\n")
        f.write(f"Clusters con imágenes: {len([c for c in cluster_summary.values() if c['size'] > 0])}\n")
        f.write(f"Tamaño promedio por cluster: {np.mean([c['size'] for c in cluster_summary.values()]):.1f}\n")
        f.write(f"Tamaño mediano por cluster: {np.median([c['size'] for c in cluster_summary.values()]):.1f}\n")
        
        # Análisis por categoría
        f.write(f"\nANÁLISIS POR CATEGORÍA:\n")
        f.write("-" * 30 + "\n")
        category_totals = {}
        for category, clusters in category_analysis.items():
            total = sum(clusters.values())
            category_totals[category] = total
        
        for category, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{category:12s}: {total:4d} imágenes ({total/16959*100:5.1f}%)\n")
    
    print("✅ Resumen de clusters creado")

def update_analysis_scripts():
    """Actualizar scripts de análisis para usar los nuevos resultados"""
    print("Actualizando scripts de análisis...")
    
    # Actualizar analyze_clusters.py
    analyze_script = '''#!/usr/bin/env python3
"""
Script para analizar los clusters del dataset completo
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def load_results():
    """Cargar resultados del clustering"""
    with open('/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_cluster_quality():
    """Analizar la calidad de los clusters"""
    results = load_results()
    
    print("ANÁLISIS DE CALIDAD DE CLUSTERS")
    print("=" * 40)
    print(f"Total de imágenes: {results['total_images']}")
    print(f"Número de clusters: {results['n_clusters']}")
    print(f"Silhouette Score: {results['metrics']['silhouette']:.4f}")
    print(f"Calinski-Harabasz Score: {results['metrics']['calinski_harabasz']:.4f}")
    print(f"Davies-Bouldin Score: {results['metrics']['davies_bouldin']:.4f}")
    
    # Análisis de distribución
    cluster_counts = list(results['cluster_counts'].values())
    print(f"\\nDistribución de clusters:")
    print(f"  Tamaño promedio: {np.mean(cluster_counts):.1f}")
    print(f"  Tamaño mediano: {np.median(cluster_counts):.1f}")
    print(f"  Desviación estándar: {np.std(cluster_counts):.1f}")
    print(f"  Clusters vacíos: {150 - len([c for c in cluster_counts if c > 0])}")

def analyze_category_distribution():
    """Analizar distribución por categorías"""
    results = load_results()
    
    print("\\nANÁLISIS POR CATEGORÍAS")
    print("=" * 30)
    
    category_totals = {}
    for category, clusters in results['category_cluster_analysis'].items():
        total = sum(clusters.values())
        category_totals[category] = total
    
    for category, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = total / results['total_images'] * 100
        print(f"{category:12s}: {total:4d} imágenes ({percentage:5.1f}%)")

def main():
    """Función principal"""
    analyze_cluster_quality()
    analyze_category_distribution()

if __name__ == "__main__":
    main()
'''
    
    with open('/home/jose/PreditorIA2025/analyze_clusters.py', 'w') as f:
        f.write(analyze_script)
    
    print("✅ Scripts de análisis actualizados")

def main():
    """Función principal"""
    print("=" * 60)
    print("ACTUALIZANDO SISTEMA DE CLUSTERING")
    print("=" * 60)
    
    # Actualizar aplicación
    update_fashion_trend_app()
    
    # Crear resumen
    create_cluster_summary()
    
    # Actualizar scripts
    update_analysis_scripts()
    
    print("\\n🎉 Actualización completada exitosamente!")
    print("\\nArchivos actualizados:")
    print("  - Fashion Trend App: modelos y visualizaciones")
    print("  - Resumen de clusters: cluster_summary.pkl y .txt")
    print("  - Scripts de análisis: analyze_clusters.py")

if __name__ == "__main__":
    main()


