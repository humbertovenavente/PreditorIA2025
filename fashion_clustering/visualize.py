#!/usr/bin/env python3
"""
Script CLI para generar todas las visualizaciones
"""

import argparse
import logging
import yaml
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Añadir el directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from fashion_clustering.utils.io import (
    load_embeddings, load_clustering_results, load_dimensionality_reduction
)
from fashion_clustering.utils.viz import (
    plot_umap_clusters, plot_elbow_curve, plot_silhouette_analysis,
    plot_cluster_sizes, plot_cluster_prototypes, plot_temporal_trends,
    create_summary_plot
)
from fashion_clustering.utils.time import analyze_temporal_trends, compare_local_global

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

def create_umap_visualization(embeddings: np.ndarray, 
                            cluster_labels: np.ndarray,
                            output_dir: str) -> None:
    """Crea visualización UMAP"""
    logger = logging.getLogger(__name__)
    logger.info("🎨 Creando visualización UMAP")
    
    # Cargar embeddings UMAP si existen
    dim_reduction = load_dimensionality_reduction(output_dir)
    if 'umap' in dim_reduction:
        umap_embeddings = dim_reduction['umap']
        plot_umap_clusters(
            umap_embeddings, cluster_labels,
            save_path=f"{output_dir}/umap_clusters.png",
            title="Clusters en espacio UMAP"
        )
    else:
        logger.warning("⚠️ Embeddings UMAP no encontrados. Ejecute reduce_dim.py primero")

def create_elbow_visualization(stats: Dict[str, Any], output_dir: str) -> None:
    """Crea visualización de curva del codo"""
    logger = logging.getLogger(__name__)
    logger.info("📈 Creando curva del codo")
    
    parameters = stats.get('parameters', {})
    if 'k_range' in parameters and 'inertias' in parameters:
        plot_elbow_curve(
            parameters['k_range'],
            parameters['inertias'],
            save_path=f"{output_dir}/elbow.png",
            title="Método del Codo - K-Means"
        )
    else:
        logger.warning("⚠️ Datos de curva del codo no disponibles")

def create_silhouette_visualization(stats: Dict[str, Any], output_dir: str) -> None:
    """Crea visualización de análisis de silhouette"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Creando análisis de silhouette")
    
    parameters = stats.get('parameters', {})
    if all(key in parameters for key in ['k_range', 'silhouette_scores', 'dbi_scores']):
        plot_silhouette_analysis(
            parameters['k_range'],
            parameters['silhouette_scores'],
            parameters['dbi_scores'],
            save_path=f"{output_dir}/silhouette_analysis.png",
            title="Análisis de Silhouette y DBI"
        )
    else:
        logger.warning("⚠️ Datos de silhouette no disponibles")

def create_cluster_sizes_visualization(cluster_labels: np.ndarray, output_dir: str) -> None:
    """Crea visualización de tamaños de clusters"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Creando gráfico de tamaños de clusters")
    
    plot_cluster_sizes(
        cluster_labels,
        save_path=f"{output_dir}/cluster_sizes.png",
        title="Tamaños de Clusters"
    )

def create_prototypes_visualization(metadata: pd.DataFrame,
                                  cluster_labels: np.ndarray,
                                  embeddings: np.ndarray,
                                  output_dir: str,
                                  top_k: int = 10) -> None:
    """Crea mosaicos de prototipos"""
    logger = logging.getLogger(__name__)
    logger.info("🖼️ Creando mosaicos de prototipos")
    
    # Encontrar prototipos por cluster
    from fashion_clustering.utils.data import find_cluster_prototypes
    image_paths = metadata['image_path'].tolist()
    prototypes = find_cluster_prototypes(embeddings, cluster_labels, image_paths, top_k)
    
    if prototypes:
        plot_cluster_prototypes(
            prototypes,
            save_dir=output_dir,
            images_per_row=5
        )
    else:
        logger.warning("⚠️ No se pudieron encontrar prototipos")

def create_temporal_visualizations(metadata: pd.DataFrame,
                                 cluster_labels: np.ndarray,
                                 output_dir: str) -> None:
    """Crea visualizaciones temporales"""
    logger = logging.getLogger(__name__)
    logger.info("📅 Creando visualizaciones temporales")
    
    # Análisis temporal
    trends_data = analyze_temporal_trends(metadata, cluster_labels)
    if not trends_data.empty:
        plot_temporal_trends(
            trends_data,
            save_path=f"{output_dir}/cluster_trends.png",
            title="Tendencias Temporales de Clusters"
        )
        
        # Guardar datos temporales
        trends_data.to_csv(f"{output_dir}/cluster_trends.csv", index=False)
        logger.info("📊 Datos temporales guardados")
    else:
        logger.warning("⚠️ No hay datos temporales disponibles")
    
    # Análisis local vs global
    local_global_stats = compare_local_global(metadata, cluster_labels)
    if local_global_stats:
        import json
        with open(f"{output_dir}/local_global_stats.json", 'w', encoding='utf-8') as f:
            json.dump(local_global_stats, f, indent=2, ensure_ascii=False)
        logger.info("📊 Estadísticas local vs global guardadas")
    else:
        logger.warning("⚠️ No hay datos de fuente disponibles")

def create_summary_visualization(stats: Dict[str, Any], output_dir: str) -> None:
    """Crea gráfico resumen"""
    logger = logging.getLogger(__name__)
    logger.info("📋 Creando gráfico resumen")
    
    create_summary_plot(
        stats,
        save_path=f"{output_dir}/clustering_summary.png"
    )

def main():
    parser = argparse.ArgumentParser(description='Generar visualizaciones')
    parser.add_argument('--all', action='store_true',
                       help='Generar todas las visualizaciones')
    parser.add_argument('--umap', action='store_true',
                       help='Generar visualización UMAP')
    parser.add_argument('--elbow', action='store_true',
                       help='Generar curva del codo')
    parser.add_argument('--silhouette', action='store_true',
                       help='Generar análisis de silhouette')
    parser.add_argument('--sizes', action='store_true',
                       help='Generar gráfico de tamaños')
    parser.add_argument('--prototypes', action='store_true',
                       help='Generar mosaicos de prototipos')
    parser.add_argument('--temporal', action='store_true',
                       help='Generar visualizaciones temporales')
    parser.add_argument('--summary', action='store_true',
                       help='Generar gráfico resumen')
    parser.add_argument('--top_prototypes', type=int, default=10,
                       help='Número de prototipos por cluster')
    parser.add_argument('--config', type=str, default='fashion_clustering/config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--input_dir', type=str, default='reports',
                       help='Directorio de entrada')
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Si no se especifica nada, generar todas
    if not any([args.umap, args.elbow, args.silhouette, args.sizes, 
                args.prototypes, args.temporal, args.summary]):
        args.all = True
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Configurar logging
    log_file = config.get('logging', {}).get('file', 'reports/run.log')
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_file, log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Iniciando generación de visualizaciones")
    
    try:
        # Cargar datos
        logger.info(f"📁 Cargando datos desde {args.input_dir}")
        embeddings_path = f"{args.input_dir}/embeddings.npy"
        metadata_path = f"{args.input_dir}/metadata.csv"
        assignments_path = f"{args.input_dir}/cluster_assignments.csv"
        stats_path = f"{args.input_dir}/cluster_stats.json"
        
        embeddings, metadata = load_embeddings(embeddings_path, metadata_path)
        cluster_labels, stats = load_clustering_results(assignments_path, stats_path)
        
        logger.info(f"✅ Datos cargados: {embeddings.shape[0]} imágenes, {len(np.unique(cluster_labels[cluster_labels != -1]))} clusters")
        
        # Crear directorio de salida
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generar visualizaciones según argumentos
        if args.all or args.umap:
            create_umap_visualization(embeddings, cluster_labels, args.input_dir)
        
        if args.all or args.elbow:
            create_elbow_visualization(stats, args.output_dir)
        
        if args.all or args.silhouette:
            create_silhouette_visualization(stats, args.output_dir)
        
        if args.all or args.sizes:
            create_cluster_sizes_visualization(cluster_labels, args.output_dir)
        
        if args.all or args.prototypes:
            create_prototypes_visualization(
                metadata, cluster_labels, embeddings, 
                args.output_dir, args.top_prototypes
            )
        
        if args.all or args.temporal:
            create_temporal_visualizations(metadata, cluster_labels, args.output_dir)
        
        if args.all or args.summary:
            create_summary_visualization(stats, args.output_dir)
        
        logger.info("🎉 Visualizaciones generadas exitosamente")
        logger.info(f"📁 Archivos guardados en: {args.output_dir}")
        
        # Listar archivos generados
        output_path = Path(args.output_dir)
        generated_files = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))
        if generated_files:
            logger.info("📊 Archivos de visualización generados:")
            for file_path in generated_files:
                logger.info(f"  - {file_path.name}")
        
    except Exception as e:
        logger.error(f"❌ Error generando visualizaciones: {e}")
        raise

if __name__ == "__main__":
    main()
