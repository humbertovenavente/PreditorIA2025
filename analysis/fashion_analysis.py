#!/usr/bin/env python3
"""
Script principal para análisis de imágenes de moda
Integra extracción de características y clustering
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Añadir el directorio padre al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from analysis.feature_extractor import FashionFeatureExtractor, get_image_paths_from_database
from analysis.clustering import FashionClustering, evaluate_clustering_quality

def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("fashion_analysis.log"),
            logging.StreamHandler()
        ]
    )

def extract_and_cluster_images(model_name: str = 'resnet50', 
                              n_clusters: int = 8,
                              max_images: int = 1000,
                              use_pca: bool = True,
                              pca_components: int = 50,
                              find_optimal_k: bool = True,
                              max_optimal_k: int = 20):
    """
    Función principal que extrae características y realiza clustering
    
    Args:
        model_name: Modelo CNN a usar ('resnet50', 'vgg16', 'inception_v3')
        n_clusters: Número de clusters (si find_optimal_k=False)
        max_images: Máximo número de imágenes a procesar
        use_pca: Si usar PCA para reducción de dimensionalidad
        pca_components: Número de componentes PCA
        find_optimal_k: Si buscar automáticamente el K óptimo
        max_optimal_k: Máximo K para búsqueda óptima
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== INICIANDO ANÁLISIS DE IMÁGENES DE MODA ===")
        
        # 1. Obtener rutas de imágenes de la base de datos
        logger.info("Obteniendo rutas de imágenes de la base de datos...")
        image_paths = get_image_paths_from_database('fashion_images.db')
        
        if not image_paths:
            logger.error("No se encontraron imágenes en la base de datos")
            return
        
        # Limitar número de imágenes si es necesario
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
            logger.info(f"Limitando análisis a {max_images} imágenes")
        
        logger.info(f"Procesando {len(image_paths)} imágenes")
        
        # 2. Extraer características
        logger.info("=== EXTRAYENDO CARACTERÍSTICAS ===")
        extractor = FashionFeatureExtractor(model_name=model_name)
        features, valid_paths = extractor.extract_batch_features(image_paths, batch_size=32)
        
        if len(features) == 0:
            logger.error("No se pudieron extraer características de ninguna imagen")
            return
        
        logger.info(f"Características extraídas: {features.shape}")
        
        # Guardar características
        features_path = f"fashion_features_{model_name}.pkl"
        extractor.save_features(features, valid_paths, features_path)
        
        # 3. Realizar clustering
        logger.info("=== REALIZANDO CLUSTERING ===")
        clustering = FashionClustering(n_clusters=n_clusters)
        
        # Preparar características
        features_processed = clustering.prepare_features(
            features, 
            use_pca=use_pca, 
            n_components=pca_components
        )
        
        # Buscar K óptimo si se solicita
        if find_optimal_k:
            logger.info("Buscando número óptimo de clusters...")
            optimal_results = clustering.find_optimal_clusters(features_processed, max_optimal_k)
            optimal_k = optimal_results['optimal_k']
            logger.info(f"K óptimo encontrado: {optimal_k}")
        else:
            optimal_k = n_clusters
        
        # Entrenar clustering
        cluster_labels = clustering.fit_clustering(features_processed, optimal_k)
        
        # 4. Analizar resultados
        logger.info("=== ANALIZANDO RESULTADOS ===")
        analysis = clustering.analyze_clusters(valid_paths, cluster_labels)
        
        # Mostrar estadísticas
        print("\n" + "="*60)
        print("RESULTADOS DEL ANÁLISIS DE CLUSTERING")
        print("="*60)
        print(f"Modelo CNN: {model_name}")
        print(f"Número de clusters: {len(np.unique(cluster_labels))}")
        print(f"Total de imágenes: {len(valid_paths)}")
        print(f"Dimensiones de características: {features.shape[1]}")
        if use_pca:
            print(f"Componentes PCA: {features_processed.shape[1]}")
        
        print("\nDistribución por cluster:")
        for cluster_id in range(len(np.unique(cluster_labels))):
            count = np.sum(cluster_labels == cluster_id)
            percentage = (count / len(cluster_labels)) * 100
            print(f"  Cluster {cluster_id}: {count} imágenes ({percentage:.1f}%)")
        
        # Evaluar calidad del clustering
        quality_metrics = evaluate_clustering_quality(features_processed, cluster_labels)
        print(f"\nMétricas de calidad:")
        print(f"  Silhouette Score: {quality_metrics['silhouette_score']:.3f} ({quality_metrics['silhouette_interpretation']})")
        print(f"  Calinski-Harabasz Score: {quality_metrics['calinski_harabasz_score']:.3f}")
        print(f"  Davies-Bouldin Score: {quality_metrics['davies_bouldin_score']:.3f}")
        
        # 5. Crear visualizaciones
        logger.info("=== CREANDO VISUALIZACIONES ===")
        clustering.visualize_clusters(features_processed, cluster_labels)
        
        # 6. Guardar resultados
        logger.info("=== GUARDANDO RESULTADOS ===")
        results_path = f"clustering_results_{model_name}.pkl"
        clustering.save_clustering_results(features, valid_paths, cluster_labels, results_path)
        
        # Crear reporte detallado
        create_detailed_report(analysis, quality_metrics, model_name, features.shape)
        
        logger.info("=== ANÁLISIS COMPLETADO EXITOSAMENTE ===")
        print(f"\nResultados guardados en:")
        print(f"  - Características: {features_path}")
        print(f"  - Clustering: {results_path}")
        print(f"  - Visualizaciones: cluster_visualizations/")
        print(f"  - Reporte: fashion_analysis_report.txt")
        
    except Exception as e:
        logger.error(f"Error en el análisis: {e}")
        raise

def create_detailed_report(analysis: dict, quality_metrics: dict, model_name: str, feature_shape: tuple):
    """
    Crea un reporte detallado del análisis
    """
    try:
        with open("fashion_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write("REPORTE DE ANÁLISIS DE CLUSTERING DE IMÁGENES DE MODA\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Modelo CNN utilizado: {model_name}\n")
            f.write(f"Dimensiones de características: {feature_shape[1]}\n")
            f.write(f"Número total de imágenes: {analysis['total_images']}\n")
            f.write(f"Número de clusters: {analysis['n_clusters']}\n\n")
            
            f.write("MÉTRICAS DE CALIDAD:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Silhouette Score: {quality_metrics['silhouette_score']:.4f}\n")
            f.write(f"Interpretación: {quality_metrics['silhouette_interpretation']}\n")
            f.write(f"Calinski-Harabasz Score: {quality_metrics['calinski_harabasz_score']:.4f}\n")
            f.write(f"Davies-Bouldin Score: {quality_metrics['davies_bouldin_score']:.4f}\n\n")
            
            f.write("DISTRIBUCIÓN POR CLUSTER:\n")
            f.write("-" * 30 + "\n")
            for cluster_id, info in analysis['cluster_analysis'].items():
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Imágenes: {info['count']} ({info['percentage']:.1f}%)\n")
                f.write(f"  Plataformas: {info['platforms']}\n")
                f.write(f"  Muestra de imágenes:\n")
                for img_path in info['sample_images']:
                    f.write(f"    - {img_path}\n")
                f.write("\n")
            
            f.write("INTERPRETACIÓN DE RESULTADOS:\n")
            f.write("-" * 30 + "\n")
            f.write("Los clusters representan grupos de imágenes con características visuales similares.\n")
            f.write("Un Silhouette Score alto indica que las imágenes están bien agrupadas.\n")
            f.write("Los clusters pueden representar diferentes categorías de ropa o estilos de moda.\n")
        
        print("Reporte detallado creado: fashion_analysis_report.txt")
        
    except Exception as e:
        print(f"Error creando reporte: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Análisis de clustering de imágenes de moda')
    parser.add_argument('--model', choices=['resnet50', 'vgg16', 'inception_v3'], 
                       default='resnet50', help='Modelo CNN a usar')
    parser.add_argument('--clusters', type=int, default=8, 
                       help='Número de clusters (ignorado si --find-optimal)')
    parser.add_argument('--max-images', type=int, default=1000,
                       help='Máximo número de imágenes a procesar')
    parser.add_argument('--no-pca', action='store_true',
                       help='No usar PCA para reducción de dimensionalidad')
    parser.add_argument('--pca-components', type=int, default=50,
                       help='Número de componentes PCA')
    parser.add_argument('--find-optimal', action='store_true',
                       help='Buscar automáticamente el número óptimo de clusters')
    parser.add_argument('--max-optimal-k', type=int, default=20,
                       help='Máximo K para búsqueda óptima')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Crear directorio de análisis si no existe
    Path("analysis_results").mkdir(exist_ok=True)
    os.chdir("analysis_results")
    
    extract_and_cluster_images(
        model_name=args.model,
        n_clusters=args.clusters,
        max_images=args.max_images,
        use_pca=not args.no_pca,
        pca_components=args.pca_components,
        find_optimal_k=args.find_optimal,
        max_optimal_k=args.max_optimal_k
    )

if __name__ == "__main__":
    main()

