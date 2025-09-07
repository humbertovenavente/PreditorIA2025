#!/usr/bin/env python3
"""
Verificar que la fórmula corregida del trend score genere la distribución correcta
"""

import numpy as np
import pickle

def verify_trend_score_formula():
    """Verificar que la fórmula corregida funcione correctamente"""
    
    try:
        # Cargar datos de clustering
        with open('/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl', 'rb') as f:
            clustering_results = pickle.load(f)
        
        print("🔍 Verificando fórmula corregida del TrendScore...")
        
        # Obtener datos reales
        cluster_counts = clustering_results.get('cluster_counts', {})
        total_images = clustering_results.get('total_images', 16959)
        n_clusters = clustering_results.get('n_clusters', 150)
        
        # Calcular cluster_sizes en el formato que usa el sistema
        cluster_sizes = {}
        for cluster_id, count in cluster_counts.items():
            cluster_sizes[f'cluster_{cluster_id}'] = count
        
        max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1
        
        print(f"   Total clusters: {len(cluster_sizes)}")
        print(f"   Cluster más grande: {max_cluster_size} imágenes")
        
        # Simular trend scores usando la fórmula CORREGIDA
        trend_scores = []
        
        for cluster_id in range(n_clusters):
            cluster_key = f'cluster_{cluster_id}'
            cluster_size = cluster_sizes.get(cluster_key, 0)
            
            if cluster_size > 0:
                # Simular múltiples imágenes para este cluster
                for _ in range(cluster_size):
                    # Simular similarity_score (0-1) con distribución realista
                    similarity_score = np.random.beta(2, 5)  # Sesgado hacia valores bajos
                    
                    # APLICAR LA FÓRMULA CORREGIDA:
                    # 1. Score base por tamaño del cluster (0-40 puntos)
                    size_score = (cluster_size / max_cluster_size) * 40
                    
                    # 2. Score por similitud (0-60 puntos) - CORREGIDO
                    similarity_contribution = similarity_score * 60  # Era 0.6, ahora 60
                    
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
        
        print(f"\n📈 ESTADÍSTICAS CON FÓRMULA CORREGIDA:")
        print(f"   Total de imágenes: {len(trend_scores):,}")
        print(f"   Media: {mean_score:.2f}")
        print(f"   Mediana: {median_score:.2f}")
        print(f"   Desviación estándar: {std_score:.2f}")
        print(f"\n📊 PERCENTILES CORREGIDOS:")
        print(f"   P25 (25%): {p25:.2f}")
        print(f"   P50 (50%): {p50:.2f}")
        print(f"   P75 (75%): {p75:.2f}")
        print(f"   P90 (90%): {p90:.2f}")
        
        # Verificar si los umbrales actuales (2, 17) son correctos
        print(f"\n🎯 VERIFICACIÓN DE UMBRALES:")
        print(f"   Umbrales actuales: < 2, 2-17, ≥17")
        print(f"   P25 real: {p25:.2f} (debería ser ~2)")
        print(f"   P75 real: {p75:.2f} (debería ser ~17)")
        
        # Calcular distribución
        no_tendencia = np.sum(trend_scores < 2)
        neutro = np.sum((trend_scores >= 2) & (trend_scores < 17))
        en_tendencia = np.sum(trend_scores >= 17)
        
        print(f"\n📊 DISTRIBUCIÓN CON UMBRALES ACTUALES:")
        print(f"   NO EN TENDENCIA (< 2): {no_tendencia:,} imágenes ({no_tendencia/len(trend_scores)*100:.1f}%)")
        print(f"   NEUTRO (2-17): {neutro:,} imágenes ({neutro/len(trend_scores)*100:.1f}%)")
        print(f"   EN TENDENCIA (≥17): {en_tendencia:,} imágenes ({en_tendencia/len(trend_scores)*100:.1f}%)")
        
        # Sugerir umbrales correctos si es necesario
        if abs(p25 - 2) > 5 or abs(p75 - 17) > 5:
            print(f"\n⚠️  UMBRALES NECESITAN AJUSTE:")
            print(f"   Umbrales sugeridos: < {p25:.0f}, {p25:.0f}-{p75:.0f}, ≥{p75:.0f}")
        else:
            print(f"\n✅ UMBRALES CORRECTOS: Los umbrales actuales coinciden con la distribución real")
        
        return {
            'p25': p25,
            'p75': p75,
            'mean': mean_score,
            'trend_scores': trend_scores
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = verify_trend_score_formula()
    if results:
        print(f"\n✅ Verificación completada")
        print(f"   Umbrales recomendados: {results['p25']:.0f}, {results['p75']:.0f}")
