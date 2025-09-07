#!/usr/bin/env python3
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
    print(f"\nDistribución de clusters:")
    print(f"  Tamaño promedio: {np.mean(cluster_counts):.1f}")
    print(f"  Tamaño mediano: {np.median(cluster_counts):.1f}")
    print(f"  Desviación estándar: {np.std(cluster_counts):.1f}")
    print(f"  Clusters vacíos: {150 - len([c for c in cluster_counts if c > 0])}")

def analyze_category_distribution():
    """Analizar distribución por categorías"""
    results = load_results()
    
    print("\nANÁLISIS POR CATEGORÍAS")
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
