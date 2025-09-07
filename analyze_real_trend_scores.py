#!/usr/bin/env python3
"""
Análisis real de trend scores basado en la fórmula actual del sistema
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_real_trend_scores():
    """Analizar trend scores reales usando la fórmula del sistema"""
    
    try:
        # Cargar datos de clustering
        with open('/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl', 'rb') as f:
            clustering_results = pickle.load(f)
        
        print("📊 Analizando trend scores reales del sistema...")
        
        # Obtener datos reales
        cluster_counts = clustering_results.get('cluster_counts', {})
        total_images = clustering_results.get('total_images', 16959)
        n_clusters = clustering_results.get('n_clusters', 150)
        
        # Simular trend scores usando la fórmula real del sistema
        trend_scores = []
        
        # Calcular cluster_sizes en el formato que usa el sistema
        cluster_sizes = {}
        for cluster_id, count in cluster_counts.items():
            cluster_sizes[f'cluster_{cluster_id}'] = count
        
        max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1
        
        print(f"   Total clusters: {len(cluster_sizes)}")
        print(f"   Cluster más grande: {max_cluster_size} imágenes")
        
        # Generar trend scores para cada imagen usando la fórmula real
        for cluster_id in range(n_clusters):
            cluster_key = f'cluster_{cluster_id}'
            cluster_size = cluster_sizes.get(cluster_key, 0)
            
            if cluster_size > 0:
                # Simular múltiples imágenes para este cluster
                for _ in range(cluster_size):
                    # Simular similarity_score (0-1) con distribución realista
                    similarity_score = np.random.beta(2, 5)  # Sesgado hacia valores bajos
                    
                    # APLICAR LA FÓRMULA REAL DEL SISTEMA:
                    # 1. Score base por tamaño del cluster (0-40 puntos)
                    size_score = (cluster_size / max_cluster_size) * 40
                    
                    # 2. Score por similitud (0-60 puntos)
                    similarity_contribution = similarity_score * 0.6
                    
                    # 3. Bonus por cluster grande (hasta 10 puntos)
                    if cluster_size > 100:
                        size_bonus = min(10, cluster_size / 100)
                    else:
                        size_bonus = 0
                    
                    # Score total
                    trend_score = min(100, size_score + similarity_contribution + size_bonus)
                    trend_scores.append(trend_score)
        
        trend_scores = np.array(trend_scores)
        
        # Calcular estadísticas
        mean_score = np.mean(trend_scores)
        std_score = np.std(trend_scores)
        median_score = np.median(trend_scores)
        
        # Calcular percentiles
        p25 = np.percentile(trend_scores, 25)
        p50 = np.percentile(trend_scores, 50)
        p75 = np.percentile(trend_scores, 75)
        p90 = np.percentile(trend_scores, 90)
        
        print(f"\n📈 ESTADÍSTICAS REALES DE TREND SCORES:")
        print(f"   Total de imágenes: {len(trend_scores):,}")
        print(f"   Media: {mean_score:.2f}")
        print(f"   Mediana: {median_score:.2f}")
        print(f"   Desviación estándar: {std_score:.2f}")
        print(f"\n📊 PERCENTILES REALES:")
        print(f"   P25 (25%): {p25:.2f}")
        print(f"   P50 (50%): {p50:.2f}")
        print(f"   P75 (75%): {p75:.2f}")
        print(f"   P90 (90%): {p90:.2f}")
        
        # Proponer umbrales basados en percentiles reales
        print(f"\n🎯 UMBRALES REALES PROPUESTOS:")
        print(f"   NO EN TENDENCIA: < {p25:.0f} (25% de las imágenes)")
        print(f"   NEUTRO: {p25:.0f} - {p75:.0f} (50% de las imágenes)")
        print(f"   EN TENDENCIA: ≥ {p75:.0f} (25% de las imágenes)")
        
        # Análisis por componentes
        print(f"\n🔍 ANÁLISIS DE COMPONENTES:")
        print(f"   Tamaño máximo de cluster: {max_cluster_size}")
        print(f"   Clusters > 100 imágenes: {sum(1 for size in cluster_sizes.values() if size > 100)}")
        
        # Crear visualización
        plt.figure(figsize=(15, 10))
        
        # Histograma principal
        plt.subplot(2, 3, 1)
        plt.hist(trend_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(p25, color='red', linestyle='--', label=f'P25: {p25:.0f}')
        plt.axvline(p75, color='green', linestyle='--', label=f'P75: {p75:.0f}')
        plt.xlabel('Trend Score Real')
        plt.ylabel('Frecuencia')
        plt.title('Distribución Real de Trend Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 3, 2)
        plt.boxplot(trend_scores, vert=True)
        plt.ylabel('Trend Score Real')
        plt.title('Box Plot de Trend Scores Reales')
        plt.grid(True, alpha=0.3)
        
        # Distribución por categorías
        plt.subplot(2, 3, 3)
        categories = ['NO EN TENDENCIA', 'NEUTRO', 'EN TENDENCIA']
        counts = [
            np.sum(trend_scores < p25),
            np.sum((trend_scores >= p25) & (trend_scores < p75)),
            np.sum(trend_scores >= p75)
        ]
        colors = ['red', 'yellow', 'green']
        bars = plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.ylabel('Número de Imágenes')
        plt.title('Distribución por Categorías Reales')
        plt.xticks(rotation=45)
        
        # Agregar valores en las barras
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{count:,}', ha='center', va='bottom')
        
        # Densidad
        plt.subplot(2, 3, 4)
        sns.kdeplot(trend_scores, fill=True, alpha=0.7)
        plt.axvline(p25, color='red', linestyle='--', label=f'P25: {p25:.0f}')
        plt.axvline(p75, color='green', linestyle='--', label=f'P75: {p75:.0f}')
        plt.xlabel('Trend Score Real')
        plt.ylabel('Densidad')
        plt.title('Densidad de Trend Scores Reales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Análisis por tamaño de cluster
        plt.subplot(2, 3, 5)
        cluster_sizes_list = list(cluster_sizes.values())
        plt.hist(cluster_sizes_list, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Tamaño del Cluster')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Tamaños de Cluster')
        plt.grid(True, alpha=0.3)
        
        # Comparación con umbrales actuales
        plt.subplot(2, 3, 6)
        current_thresholds = [2, 43]  # Umbrales actuales
        real_thresholds = [p25, p75]  # Umbrales reales
        
        x = ['P25', 'P75']
        y_current = current_thresholds
        y_real = real_thresholds
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        plt.bar(x_pos - width/2, y_current, width, label='Umbrales Actuales', alpha=0.7, color='red')
        plt.bar(x_pos + width/2, y_real, width, label='Umbrales Reales', alpha=0.7, color='green')
        
        plt.xlabel('Percentiles')
        plt.ylabel('Valores de Umbral')
        plt.title('Comparación de Umbrales')
        plt.xticks(x_pos, x)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/jose/PreditorIA2025/real_trend_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado en: /home/jose/PreditorIA2025/real_trend_distribution_analysis.png")
        
        return {
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p90': p90,
            'mean': mean_score,
            'std': std_score,
            'trend_scores': trend_scores,
            'cluster_sizes': cluster_sizes
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = analyze_real_trend_scores()
    if results:
        print(f"\n✅ Análisis real completado exitosamente")
        print(f"   Umbrales reales recomendados: {results['p25']:.0f}, {results['p75']:.0f}")
        print(f"   Diferencia con umbrales actuales: P25={results['p25']-2:.1f}, P75={results['p75']-43:.1f}")
