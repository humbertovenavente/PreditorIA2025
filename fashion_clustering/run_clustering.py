#!/usr/bin/env python3
"""
Script CLI para ejecutar clustering (K-Means y DBSCAN)
"""

import argparse
import logging
import yaml
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# Añadir el directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from fashion_clustering.utils.io import load_embeddings, save_clustering_results
from fashion_clustering.utils.metrics import (
    find_optimal_k, find_optimal_dbscan_params, compare_algorithms,
    calculate_clustering_metrics
)
from sklearn.cluster import KMeans, DBSCAN

def setup_logging(log_file: str = "reports/run.log", level: str = "INFO"):
    """Configura logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error cargando configuración: {e}")
        return {}

def run_kmeans_clustering(embeddings: np.ndarray, k: int, random_state: int = 42) -> tuple:
    """Ejecuta clustering K-Means"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Ejecutando K-Means con K={k}")
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calcular métricas
    metrics = calculate_clustering_metrics(embeddings, cluster_labels, 'kmeans')
    
    logger.info(f"✅ K-Means completado: {len(np.unique(cluster_labels))} clusters")
    logger.info(f"📊 Silhouette: {metrics['silhouette_score']:.3f}, DBI: {metrics['davies_bouldin_score']:.3f}")
    
    return cluster_labels, metrics

def run_dbscan_clustering(embeddings: np.ndarray, eps: float, min_samples: int) -> tuple:
    """Ejecuta clustering DBSCAN"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Ejecutando DBSCAN con eps={eps}, min_samples={min_samples}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    
    # Calcular métricas
    metrics = calculate_clustering_metrics(embeddings, cluster_labels, 'dbscan')
    
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    n_noise = np.sum(cluster_labels == -1)
    
    logger.info(f"✅ DBSCAN completado: {n_clusters} clusters, {n_noise} ruido")
    logger.info(f"📊 Silhouette: {metrics['silhouette_score']:.3f}, DBI: {metrics['davies_bouldin_score']:.3f}")
    
    return cluster_labels, metrics

def main():
    parser = argparse.ArgumentParser(description='Ejecutar clustering')
    parser.add_argument('--algo', type=str, choices=['kmeans', 'dbscan', 'auto'], 
                       default='auto', help='Algoritmo de clustering')
    parser.add_argument('--k_min', type=int, default=4,
                       help='K mínimo para K-Means')
    parser.add_argument('--k_max', type=int, default=30,
                       help='K máximo para K-Means')
    parser.add_argument('--k', type=int, default=None,
                       help='K fijo para K-Means (ignora búsqueda)')
    parser.add_argument('--dbscan_eps', type=float, default=None,
                       help='Eps fijo para DBSCAN')
    parser.add_argument('--dbscan_min_samples', type=int, default=None,
                       help='Min_samples fijo para DBSCAN')
    parser.add_argument('--dbscan_eps_grid', nargs='+', type=float, 
                       default=[0.3, 0.5, 0.7, 0.9],
                       help='Grid de valores eps para DBSCAN')
    parser.add_argument('--dbscan_min_samples_grid', nargs='+', type=int,
                       default=[5, 10, 15, 20],
                       help='Grid de valores min_samples para DBSCAN')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Semilla aleatoria')
    parser.add_argument('--config', type=str, default='fashion_clustering/config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--input_dir', type=str, default='reports',
                       help='Directorio de entrada')
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Configurar logging
    log_file = config.get('logging', {}).get('file', 'reports/run.log')
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_file, log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Iniciando clustering")
    
    try:
        # Cargar embeddings
        logger.info(f"📁 Cargando embeddings desde {args.input_dir}")
        embeddings_path = f"{args.input_dir}/embeddings.npy"
        metadata_path = f"{args.input_dir}/metadata.csv"
        
        embeddings, metadata = load_embeddings(embeddings_path, metadata_path)
        logger.info(f"✅ Embeddings cargados: {embeddings.shape}")
        
        # Configurar parámetros
        clustering_config = config.get('clustering', {})
        kmeans_config = clustering_config.get('kmeans', {})
        dbscan_config = clustering_config.get('dbscan', {})
        
        k_min = args.k_min or kmeans_config.get('k_min', 4)
        k_max = args.k_max or kmeans_config.get('k_max', 30)
        random_state = args.random_state or kmeans_config.get('random_state', 42)
        
        best_algorithm = None
        best_labels = None
        best_metrics = None
        best_parameters = None
        
        if args.algo == 'kmeans' or args.algo == 'auto':
            logger.info("🔍 Ejecutando K-Means")
            
            if args.k is not None:
                # K fijo
                cluster_labels, metrics = run_kmeans_clustering(embeddings, args.k, random_state)
                parameters = {'k': args.k, 'random_state': random_state}
            else:
                # Búsqueda de K óptimo
                logger.info(f"🔍 Buscando K óptimo en rango {k_min}-{k_max}")
                k_results = find_optimal_k(embeddings, k_min, k_max, random_state)
                optimal_k = k_results['optimal_k']
                
                cluster_labels, metrics = run_kmeans_clustering(embeddings, optimal_k, random_state)
                parameters = {
                    'k': optimal_k,
                    'k_range': k_results['k_range'],
                    'inertias': k_results['inertias'],
                    'silhouette_scores': k_results['silhouette_scores'],
                    'dbi_scores': k_results['dbi_scores'],
                    'random_state': random_state
                }
            
            if args.algo == 'kmeans' or best_algorithm is None:
                best_algorithm = 'kmeans'
                best_labels = cluster_labels
                best_metrics = metrics
                best_parameters = parameters
        
        if args.algo == 'dbscan' or args.algo == 'auto':
            logger.info("🔍 Ejecutando DBSCAN")
            
            if args.dbscan_eps is not None and args.dbscan_min_samples is not None:
                # Parámetros fijos
                cluster_labels, metrics = run_dbscan_clustering(
                    embeddings, args.dbscan_eps, args.dbscan_min_samples
                )
                parameters = {
                    'eps': args.dbscan_eps,
                    'min_samples': args.dbscan_min_samples
                }
            else:
                # Búsqueda de parámetros óptimos
                eps_grid = args.dbscan_eps_grid or dbscan_config.get('eps_grid', [0.3, 0.5, 0.7, 0.9])
                min_samples_grid = args.dbscan_min_samples_grid or dbscan_config.get('min_samples_grid', [5, 10, 15, 20])
                
                logger.info(f"🔍 Buscando parámetros óptimos para DBSCAN")
                dbscan_results = find_optimal_dbscan_params(embeddings, eps_grid, min_samples_grid)
                
                if dbscan_results['best_params'] is not None:
                    eps, min_samples = dbscan_results['best_params']
                    cluster_labels, metrics = run_dbscan_clustering(embeddings, eps, min_samples)
                    parameters = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'eps_grid': eps_grid,
                        'min_samples_grid': min_samples_grid,
                        'search_results': dbscan_results['results']
                    }
                else:
                    logger.warning("⚠️ No se encontraron parámetros válidos para DBSCAN")
                    # Saltar DBSCAN si no se encuentran parámetros válidos
                    if args.algo == 'dbscan':
                        logger.error("❌ No se puede ejecutar DBSCAN sin parámetros válidos")
                        return
            
            if args.algo == 'dbscan' or best_algorithm is None:
                best_algorithm = 'dbscan'
                best_labels = cluster_labels
                best_metrics = metrics
                best_parameters = parameters
        
        # Comparar algoritmos si es auto
        if args.algo == 'auto' and best_algorithm is not None:
            logger.info("🔄 Comparando algoritmos")
            comparison = compare_algorithms(embeddings, random_state=random_state)
            
            if comparison['best_algorithm'] != best_algorithm:
                logger.info(f"🔄 Mejor algoritmo: {comparison['best_algorithm']}")
                best_algorithm = comparison['best_algorithm']
                best_labels = comparison['best_labels']
                best_metrics = comparison['best_metrics']
        
        if best_labels is None:
            logger.error("❌ No se pudo ejecutar clustering")
            return
        
        # Guardar resultados
        logger.info("💾 Guardando resultados")
        saved_files = save_clustering_results(
            cluster_labels=best_labels,
            metrics=best_metrics,
            algorithm=best_algorithm,
            parameters=best_parameters,
            save_dir=args.output_dir
        )
        
        logger.info("🎉 Clustering completado exitosamente")
        logger.info(f"📊 Algoritmo: {best_algorithm}")
        logger.info(f"📊 Clusters: {len(np.unique(best_labels[best_labels != -1]))}")
        logger.info(f"📊 Silhouette: {best_metrics['silhouette_score']:.3f}")
        logger.info(f"📁 Archivos guardados:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  - {file_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Error en clustering: {e}")
        raise

if __name__ == "__main__":
    main()
