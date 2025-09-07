"""
Módulo para métricas de clustering y selección de parámetros
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_clustering_metrics(embeddings: np.ndarray,
                               cluster_labels: np.ndarray,
                               algorithm: str = 'kmeans') -> Dict[str, float]:
    """
    Calcula métricas de clustering
    
    Args:
        embeddings: Embeddings de las imágenes
        cluster_labels: Etiquetas de cluster asignadas
        algorithm: Algoritmo usado ('kmeans' o 'dbscan')
    
    Returns:
        Diccionario con métricas
    """
    try:
        # Filtrar puntos de ruido para DBSCAN
        if algorithm == 'dbscan':
            mask = cluster_labels != -1
            if np.sum(mask) == 0:
                return {
                    'silhouette_score': -1.0,
                    'davies_bouldin_score': float('inf'),
                    'calinski_harabasz_score': 0.0,
                    'n_clusters': 0,
                    'n_noise': len(cluster_labels)
                }
            
            filtered_embeddings = embeddings[mask]
            filtered_labels = cluster_labels[mask]
        else:
            filtered_embeddings = embeddings
            filtered_labels = cluster_labels
        
        n_clusters = len(np.unique(filtered_labels))
        n_noise = np.sum(cluster_labels == -1) if algorithm == 'dbscan' else 0
        
        if n_clusters < 2:
            return {
                'silhouette_score': -1.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
        
        # Calcular métricas
        silhouette = silhouette_score(filtered_embeddings, filtered_labels)
        dbi = davies_bouldin_score(filtered_embeddings, filtered_labels)
        chi = calinski_harabasz_score(filtered_embeddings, filtered_labels)
        
        metrics = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(dbi),
            'calinski_harabasz_score': float(chi),
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise)
        }
        
        logger.info(f"Métricas calculadas - Silhouette: {silhouette:.3f}, DBI: {dbi:.3f}, CHI: {chi:.3f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculando métricas: {e}")
        return {
            'silhouette_score': -1.0,
            'davies_bouldin_score': float('inf'),
            'calinski_harabasz_score': 0.0,
            'n_clusters': 0,
            'n_noise': 0
        }

def find_optimal_k(embeddings: np.ndarray,
                  k_min: int = 4,
                  k_max: int = 30,
                  random_state: int = 42) -> Dict[str, Any]:
    """
    Encuentra el número óptimo de clusters para K-Means
    
    Args:
        embeddings: Embeddings de las imágenes
        k_min: Número mínimo de clusters
        k_max: Número máximo de clusters
        random_state: Semilla aleatoria
    
    Returns:
        Diccionario con resultados de la búsqueda
    """
    try:
        k_range = range(k_min, k_max + 1)
        inertias = []
        silhouette_scores = []
        dbi_scores = []
        
        logger.info(f"Buscando K óptimo en rango {k_min}-{k_max}")
        
        for k in k_range:
            # Entrenar K-Means
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calcular métricas
            inertia = kmeans.inertia_
            silhouette = silhouette_score(embeddings, cluster_labels)
            dbi = davies_bouldin_score(embeddings, cluster_labels)
            
            inertias.append(inertia)
            silhouette_scores.append(silhouette)
            dbi_scores.append(dbi)
            
            logger.info(f"K={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.3f}, DBI={dbi:.3f}")
        
        # Seleccionar K óptimo
        optimal_k = select_optimal_k(k_range, silhouette_scores, dbi_scores)
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'dbi_scores': dbi_scores,
            'optimal_k': optimal_k
        }
        
        logger.info(f"K óptimo seleccionado: {optimal_k}")
        return results
        
    except Exception as e:
        logger.error(f"Error buscando K óptimo: {e}")
        return {
            'k_range': [],
            'inertias': [],
            'silhouette_scores': [],
            'dbi_scores': [],
            'optimal_k': k_min
        }

def select_optimal_k(k_range: List[int],
                    silhouette_scores: List[float],
                    dbi_scores: List[float],
                    silhouette_threshold: float = 0.3,
                    dbi_threshold: float = 2.0) -> int:
    """
    Selecciona el K óptimo basado en las métricas
    
    Args:
        k_range: Rango de valores de K probados
        silhouette_scores: Puntuaciones de silhouette
        dbi_scores: Puntuaciones de DBI
        silhouette_threshold: Umbral mínimo de silhouette
        dbi_threshold: Umbral máximo de DBI
    
    Returns:
        K óptimo seleccionado
    """
    try:
        # Filtrar K válidos (silhouette > threshold, dbi < threshold)
        valid_k = []
        for i, k in enumerate(k_range):
            if silhouette_scores[i] > silhouette_threshold and dbi_scores[i] < dbi_threshold:
                valid_k.append((k, silhouette_scores[i], dbi_scores[i]))
        
        if not valid_k:
            # Si no hay K válidos, usar el de mayor silhouette
            best_idx = np.argmax(silhouette_scores)
            return k_range[best_idx]
        
        # Ordenar por silhouette descendente, luego por DBI ascendente
        valid_k.sort(key=lambda x: (-x[1], x[2]))
        
        # Seleccionar el mejor K
        optimal_k = valid_k[0][0]
        
        # Verificar mejora significativa vs K-1
        if optimal_k > k_range[0]:
            prev_idx = k_range.index(optimal_k - 1)
            if prev_idx < len(silhouette_scores):
                silhouette_improvement = silhouette_scores[k_range.index(optimal_k)] - silhouette_scores[prev_idx]
                dbi_improvement = dbi_scores[prev_idx] - dbi_scores[k_range.index(optimal_k)]
                
                # Si la mejora no es significativa, usar K-1
                if silhouette_improvement < 0.05 and dbi_improvement < 0.1:
                    optimal_k = optimal_k - 1
        
        return optimal_k
        
    except Exception as e:
        logger.error(f"Error seleccionando K óptimo: {e}")
        return k_range[0] if k_range else 4

def find_optimal_dbscan_params(embeddings: np.ndarray,
                              eps_grid: List[float] = [0.3, 0.5, 0.7, 0.9],
                              min_samples_grid: List[int] = [5, 10, 15, 20]) -> Dict[str, Any]:
    """
    Encuentra parámetros óptimos para DBSCAN
    
    Args:
        embeddings: Embeddings de las imágenes
        eps_grid: Valores de eps a probar
        min_samples_grid: Valores de min_samples a probar
        random_state: Semilla aleatoria
    
    Returns:
        Diccionario con resultados de la búsqueda
    """
    try:
        best_score = -1
        best_params = None
        best_labels = None
        results = []
        
        logger.info("Buscando parámetros óptimos para DBSCAN")
        
        for eps in eps_grid:
            for min_samples in min_samples_grid:
                # Entrenar DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(embeddings)
                
                # Calcular métricas
                n_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = np.sum(cluster_labels == -1)
                
                if n_clusters < 2:
                    score = -1
                else:
                    # Filtrar ruido para calcular silhouette
                    mask = cluster_labels != -1
                    if np.sum(mask) > 1:
                        score = silhouette_score(embeddings[mask], cluster_labels[mask])
                    else:
                        score = -1
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': score
                })
                
                logger.info(f"eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} ruido, silhouette={score:.3f}")
                
                # Actualizar mejor resultado
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
                    best_labels = cluster_labels
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_labels': best_labels,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error buscando parámetros DBSCAN: {e}")
        return {
            'best_params': None,
            'best_score': -1,
            'best_labels': None,
            'results': []
        }

def compare_algorithms(embeddings: np.ndarray,
                      kmeans_k: int = 10,
                      dbscan_eps: float = 0.5,
                      dbscan_min_samples: int = 10,
                      random_state: int = 42) -> Dict[str, Any]:
    """
    Compara K-Means y DBSCAN
    
    Args:
        embeddings: Embeddings de las imágenes
        kmeans_k: Número de clusters para K-Means
        dbscan_eps: Parámetro eps para DBSCAN
        dbscan_min_samples: Parámetro min_samples para DBSCAN
        random_state: Semilla aleatoria
    
    Returns:
        Diccionario con comparación de algoritmos
    """
    try:
        # K-Means
        kmeans = KMeans(n_clusters=kmeans_k, random_state=random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        kmeans_metrics = calculate_clustering_metrics(embeddings, kmeans_labels, 'kmeans')
        
        # DBSCAN
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_labels = dbscan.fit_predict(embeddings)
        dbscan_metrics = calculate_clustering_metrics(embeddings, dbscan_labels, 'dbscan')
        
        # Seleccionar mejor algoritmo
        if kmeans_metrics['silhouette_score'] > dbscan_metrics['silhouette_score']:
            best_algorithm = 'kmeans'
            best_labels = kmeans_labels
            best_metrics = kmeans_metrics
        else:
            best_algorithm = 'dbscan'
            best_labels = dbscan_labels
            best_metrics = dbscan_metrics
        
        return {
            'kmeans': {
                'labels': kmeans_labels,
                'metrics': kmeans_metrics
            },
            'dbscan': {
                'labels': dbscan_labels,
                'metrics': dbscan_metrics
            },
            'best_algorithm': best_algorithm,
            'best_labels': best_labels,
            'best_metrics': best_metrics
        }
        
    except Exception as e:
        logger.error(f"Error comparando algoritmos: {e}")
        return {
            'kmeans': {'labels': None, 'metrics': {}},
            'dbscan': {'labels': None, 'metrics': {}},
            'best_algorithm': 'kmeans',
            'best_labels': None,
            'best_metrics': {}
        }


