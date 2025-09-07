#!/usr/bin/env python3
"""
Debug del trend score para ver qué está pasando
"""

import numpy as np
import pickle

def debug_trend_score():
    """Debug del trend score con ejemplos reales"""
    
    try:
        # Cargar datos de clustering
        with open('/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl', 'rb') as f:
            clustering_results = pickle.load(f)
        
        print("🔍 Debug del TrendScore...")
        
        # Obtener datos reales
        cluster_counts = clustering_results.get('cluster_counts', {})
        
        # Calcular cluster_sizes en el formato que usa el sistema
        cluster_sizes = {}
        for cluster_id, count in cluster_counts.items():
            cluster_sizes[f'cluster_{cluster_id}'] = count
        
        max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1
        
        print(f"   Cluster más grande: {max_cluster_size} imágenes")
        
        # Probar con diferentes clusters
        test_clusters = [0, 1, 2, 10, 50, 100, 149]
        
        for cluster_id in test_clusters:
            cluster_key = f'cluster_{cluster_id}'
            cluster_size = cluster_sizes.get(cluster_key, 0)
            
            if cluster_size > 0:
                print(f"\n📊 Cluster {cluster_id}:")
                print(f"   Tamaño: {cluster_size} imágenes")
                
                # Probar con diferentes similitudes
                for sim in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    # Fórmula actual
                    size_score = (cluster_size / max_cluster_size) * 40
                    similarity_contribution = sim * 60  # CORREGIDO
                    
                    if cluster_size > 100:
                        size_bonus = min(10, cluster_size / 100)
                    else:
                        size_bonus = 0
                    
                    trend_score = min(100, size_score + similarity_contribution + size_bonus)
                    
                    print(f"   Similitud {sim}: {trend_score:.1f} puntos (tamaño: {size_score:.1f}, sim: {similarity_contribution:.1f}, bonus: {size_bonus:.1f})")
        
        # Verificar distribución real
        print(f"\n📈 Análisis de distribución:")
        all_scores = []
        
        for cluster_id in range(150):
            cluster_key = f'cluster_{cluster_id}'
            cluster_size = cluster_sizes.get(cluster_key, 0)
            
            if cluster_size > 0:
                # Simular 10 imágenes por cluster
                for _ in range(min(10, cluster_size)):
                    sim = np.random.beta(2, 5)  # Similitud realista
                    
                    size_score = (cluster_size / max_cluster_size) * 40
                    similarity_contribution = sim * 60
                    
                    if cluster_size > 100:
                        size_bonus = min(10, cluster_size / 100)
                    else:
                        size_bonus = 0
                    
                    trend_score = min(100, size_score + similarity_contribution + size_bonus)
                    all_scores.append(trend_score)
        
        all_scores = np.array(all_scores)
        
        print(f"   Total scores: {len(all_scores)}")
        print(f"   Media: {np.mean(all_scores):.2f}")
        print(f"   P25: {np.percentile(all_scores, 25):.2f}")
        print(f"   P50: {np.percentile(all_scores, 50):.2f}")
        print(f"   P75: {np.percentile(all_scores, 75):.2f}")
        print(f"   P90: {np.percentile(all_scores, 90):.2f}")
        
        # Verificar distribución con umbrales actuales
        no_tendencia = np.sum(all_scores < 16)
        neutro = np.sum((all_scores >= 16) & (all_scores < 42))
        en_tendencia = np.sum(all_scores >= 42)
        
        print(f"\n🎯 Distribución con umbrales actuales (16, 42):")
        print(f"   NO EN TENDENCIA (< 16): {no_tendencia} ({no_tendencia/len(all_scores)*100:.1f}%)")
        print(f"   NEUTRO (16-42): {neutro} ({neutro/len(all_scores)*100:.1f}%)")
        print(f"   EN TENDENCIA (≥42): {en_tendencia} ({en_tendencia/len(all_scores)*100:.1f}%)")
        
        # Sugerir umbrales más realistas
        p25 = np.percentile(all_scores, 25)
        p75 = np.percentile(all_scores, 75)
        
        print(f"\n💡 Umbrales sugeridos para distribución 25%-50%-25%:")
        print(f"   NO EN TENDENCIA: < {p25:.0f}")
        print(f"   NEUTRO: {p25:.0f}-{p75:.0f}")
        print(f"   EN TENDENCIA: ≥ {p75:.0f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_trend_score()
