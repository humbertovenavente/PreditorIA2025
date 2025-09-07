#!/usr/bin/env python3
"""
Script CLI para resumir y etiquetar clusters
"""

import argparse
import logging
import yaml
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Añadir el directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from fashion_clustering.utils.io import (
    load_embeddings, load_clustering_results, save_cluster_summary
)
from fashion_clustering.utils.colors import analyze_cluster_colors
from fashion_clustering.utils.metrics import calculate_clustering_metrics

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

def find_cluster_prototypes(embeddings: np.ndarray, 
                          cluster_labels: np.ndarray, 
                          image_paths: List[str],
                          top_k: int = 10) -> Dict[int, List[str]]:
    """Encuentra prototipos (imágenes más cercanas al centroide) por cluster"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Buscando {top_k} prototipos por cluster")
    
    prototypes = {}
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        if label == -1:  # Saltar ruido
            continue
        
        # Obtener embeddings del cluster
        mask = cluster_labels == label
        cluster_embeddings = embeddings[mask]
        cluster_paths = [image_paths[i] for i in range(len(image_paths)) if mask[i]]
        
        if len(cluster_embeddings) == 0:
            continue
        
        # Calcular centroide
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calcular distancias al centroide
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Obtener índices de los más cercanos
        closest_indices = np.argsort(distances)[:min(top_k, len(distances))]
        
        # Obtener rutas de los prototipos
        cluster_prototypes = [cluster_paths[i] for i in closest_indices]
        prototypes[label] = cluster_prototypes
        
        logger.info(f"  Cluster {label}: {len(cluster_prototypes)} prototipos")
    
    return prototypes

def analyze_cluster_keywords(metadata: pd.DataFrame, 
                           cluster_labels: np.ndarray,
                           cluster_id: int) -> List[str]:
    """Analiza palabras clave de un cluster basado en metadatos"""
    try:
        # Obtener metadatos del cluster
        mask = cluster_labels == cluster_id
        cluster_metadata = metadata[mask]
        
        # Extraer palabras clave de diferentes columnas
        keywords = []
        
        # De la columna 'label' (categoría)
        if 'label' in cluster_metadata.columns:
            labels = cluster_metadata['label'].dropna()
            if not labels.empty:
                label_counts = Counter(labels)
                keywords.extend([f"{label}({count})" for label, count in label_counts.most_common(5)])
        
        # De la columna 'source'
        if 'source' in cluster_metadata.columns:
            sources = cluster_metadata['source'].dropna()
            if not sources.empty:
                source_counts = Counter(sources)
                keywords.extend([f"fuente:{source}({count})" for source, count in source_counts.most_common(3)])
        
        # De la columna 'split'
        if 'split' in cluster_metadata.columns:
            splits = cluster_metadata['split'].dropna()
            if not splits.empty:
                split_counts = Counter(splits)
                keywords.extend([f"split:{split}({count})" for split, count in split_counts.most_common(3)])
        
        return keywords[:10]  # Limitar a 10 palabras clave
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error analizando palabras clave del cluster {cluster_id}: {e}")
        return []

def generate_suggested_label(colors: List[str], 
                           keywords: List[str], 
                           cluster_size: int) -> str:
    """Genera etiqueta sugerida para el cluster"""
    try:
        # Palabras clave de estilo
        style_keywords = []
        
        # Analizar colores
        if colors:
            color_analysis = analyze_color_style(colors)
            style_keywords.extend(color_analysis)
        
        # Analizar palabras clave
        if keywords:
            keyword_analysis = analyze_keyword_style(keywords)
            style_keywords.extend(keyword_analysis)
        
        # Generar etiqueta basada en análisis
        if style_keywords:
            # Tomar las 2-3 palabras más relevantes
            top_keywords = Counter(style_keywords).most_common(3)
            label_parts = [kw[0] for kw in top_keywords]
            suggested_label = " ".join(label_parts)
        else:
            # Etiqueta genérica basada en tamaño
            if cluster_size > 1000:
                suggested_label = "Estilo popular dominante"
            elif cluster_size > 500:
                suggested_label = "Estilo común"
            else:
                suggested_label = "Estilo especializado"
        
        return suggested_label
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error generando etiqueta sugerida: {e}")
        return "Estilo no clasificado"

def analyze_color_style(colors: List[str]) -> List[str]:
    """Analiza estilo basado en colores"""
    style_keywords = []
    
    # Palabras clave por color
    color_styles = {
        'negro': ['elegante', 'formal', 'sofisticado'],
        'blanco': ['limpio', 'minimalista', 'fresco'],
        'rojo': ['llamativo', 'pasional', 'audaz'],
        'azul': ['clásico', 'confiable', 'tranquilo'],
        'verde': ['natural', 'relajante', 'ecológico'],
        'rosa': ['femenino', 'romántico', 'suave'],
        'gris': ['neutral', 'moderno', 'urbano'],
        'marrón': ['terroso', 'cálido', 'natural'],
        'beige': ['neutro', 'elegante', 'sutil'],
        'azul marino': ['clásico', 'profesional', 'serio']
    }
    
    for color in colors:
        color_lower = color.lower()
        for style_color, styles in color_styles.items():
            if style_color in color_lower:
                style_keywords.extend(styles)
                break
    
    return style_keywords

def analyze_keyword_style(keywords: List[str]) -> List[str]:
    """Analiza estilo basado en palabras clave"""
    style_keywords = []
    
    # Mapeo de palabras clave a estilos
    keyword_styles = {
        'dress': ['elegante', 'femenino'],
        'tops': ['casual', 'versátil'],
        'shoes': ['funcional', 'estilo'],
        'bags': ['accesorio', 'práctico'],
        'pants': ['casual', 'cómodo'],
        'hats': ['accesorio', 'estilo'],
        'jewelry': ['elegante', 'decorativo'],
        'scarves': ['accesorio', 'elegante'],
        'accessories': ['accesorio', 'estilo'],
        'train': ['entrenamiento', 'deportivo'],
        'val': ['validación', 'calidad'],
        'test': ['prueba', 'evaluación']
    }
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for kw, styles in keyword_styles.items():
            if kw in keyword_lower:
                style_keywords.extend(styles)
                break
    
    return style_keywords

def main():
    parser = argparse.ArgumentParser(description='Resumir y etiquetar clusters')
    parser.add_argument('--top_prototypes', type=int, default=10,
                       help='Número de prototipos por cluster')
    parser.add_argument('--n_colors', type=int, default=5,
                       help='Número de colores dominantes por cluster')
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
    logger.info("🚀 Iniciando resumen de clusters")
    
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
        
        # Obtener rutas de imágenes
        image_paths = metadata['image_path'].tolist()
        
        # Encontrar prototipos
        logger.info("🔍 Buscando prototipos de clusters")
        prototypes = find_cluster_prototypes(
            embeddings, cluster_labels, image_paths, args.top_prototypes
        )
        
        # Analizar cada cluster
        logger.info("🎨 Analizando clusters")
        clusters_analysis = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Saltar ruido
                continue
            
            logger.info(f"  Analizando cluster {label}")
            
            # Obtener imágenes del cluster
            mask = cluster_labels == label
            cluster_images = [image_paths[i] for i in range(len(image_paths)) if mask[i]]
            
            # Analizar colores
            color_analysis = analyze_cluster_colors(
                cluster_images, args.n_colors
            )
            
            # Analizar palabras clave
            keywords = analyze_cluster_keywords(metadata, cluster_labels, label)
            
            # Generar etiqueta sugerida
            suggested_label = generate_suggested_label(
                color_analysis['color_names'], keywords, len(cluster_images)
            )
            
            # Guardar análisis del cluster
            clusters_analysis[label] = {
                'size': len(cluster_images),
                'colors': color_analysis['color_names'],
                'palette_description': color_analysis['palette_description'],
                'keywords': keywords,
                'suggested_label': suggested_label,
                'prototypes': prototypes.get(label, [])
            }
        
        # Crear resumen completo
        logger.info("📝 Creando resumen completo")
        summary_data = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': stats.get('algorithm', 'N/A'),
            'total_images': len(embeddings),
            'n_clusters': len(unique_labels[unique_labels != -1]),
            'metrics': stats.get('metrics', {}),
            'clusters': clusters_analysis
        }
        
        # Guardar resumen
        logger.info("💾 Guardando resumen")
        summary_path = f"{args.output_dir}/cluster_summary.md"
        save_cluster_summary(summary_data, summary_path)
        
        logger.info("🎉 Resumen de clusters completado exitosamente")
        logger.info(f"📁 Resumen guardado en: {summary_path}")
        
        # Mostrar resumen por consola
        logger.info("\n📊 RESUMEN DE CLUSTERS:")
        for cluster_id, analysis in clusters_analysis.items():
            logger.info(f"  Cluster {cluster_id}: {analysis['size']} imágenes")
            logger.info(f"    Colores: {', '.join(analysis['colors'][:3])}")
            logger.info(f"    Etiqueta: {analysis['suggested_label']}")
            logger.info(f"    Palabras clave: {', '.join(analysis['keywords'][:3])}")
        
    except Exception as e:
        logger.error(f"❌ Error en resumen de clusters: {e}")
        raise

if __name__ == "__main__":
    main()


