#!/usr/bin/env python3
"""
Script CLI para reducción de dimensionalidad (PCA y UMAP)
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

from fashion_clustering.utils.io import load_embeddings, save_dimensionality_reduction
from sklearn.decomposition import PCA
import umap

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

def run_pca(embeddings: np.ndarray, n_components: int, random_state: int = 42) -> np.ndarray:
    """Ejecuta PCA"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Ejecutando PCA con {n_components} componentes")
    
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_embeddings = pca.fit_transform(embeddings)
    
    logger.info(f"✅ PCA completado: {pca_embeddings.shape}")
    logger.info(f"📊 Varianza explicada: {pca.explained_variance_ratio_.sum():.3f}")
    
    return pca_embeddings

def run_umap(embeddings: np.ndarray, n_components: int, n_neighbors: int = 15, 
             min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Ejecuta UMAP"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Ejecutando UMAP con {n_components} componentes")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    
    logger.info(f"✅ UMAP completado: {umap_embeddings.shape}")
    
    return umap_embeddings

def main():
    parser = argparse.ArgumentParser(description='Reducción de dimensionalidad')
    parser.add_argument('--method', type=str, choices=['pca', 'umap', 'both'], 
                       default='both', help='Método de reducción')
    parser.add_argument('--n_components', type=int, default=50,
                       help='Número de componentes (PCA) o dimensiones (UMAP)')
    parser.add_argument('--umap_components', type=int, default=2,
                       help='Número de componentes para UMAP')
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='Número de vecinos para UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='Distancia mínima para UMAP')
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
    logger.info("🚀 Iniciando reducción de dimensionalidad")
    
    try:
        # Cargar embeddings
        logger.info(f"📁 Cargando embeddings desde {args.input_dir}")
        embeddings_path = f"{args.input_dir}/embeddings.npy"
        metadata_path = f"{args.input_dir}/metadata.csv"
        
        embeddings, metadata = load_embeddings(embeddings_path, metadata_path)
        logger.info(f"✅ Embeddings cargados: {embeddings.shape}")
        
        # Configurar parámetros
        dim_config = config.get('dimensionality_reduction', {})
        pca_config = dim_config.get('pca', {})
        umap_config = dim_config.get('umap', {})
        
        pca_embeddings = None
        umap_embeddings = None
        
        # Ejecutar PCA
        if args.method in ['pca', 'both']:
            n_components = args.n_components or pca_config.get('n_components', 50)
            random_state = args.random_state or pca_config.get('random_state', 42)
            
            pca_embeddings = run_pca(embeddings, n_components, random_state)
        
        # Ejecutar UMAP
        if args.method in ['umap', 'both']:
            n_components = args.umap_components or umap_config.get('n_components', 2)
            n_neighbors = args.n_neighbors or umap_config.get('n_neighbors', 15)
            min_dist = args.min_dist or umap_config.get('min_dist', 0.1)
            random_state = args.random_state or umap_config.get('random_state', 42)
            
            umap_embeddings = run_umap(embeddings, n_components, n_neighbors, min_dist, random_state)
        
        # Guardar resultados
        logger.info("💾 Guardando resultados")
        saved_files = save_dimensionality_reduction(
            pca_embeddings=pca_embeddings,
            umap_embeddings=umap_embeddings,
            save_dir=args.output_dir
        )
        
        logger.info("🎉 Reducción de dimensionalidad completada exitosamente")
        logger.info(f"📁 Archivos guardados:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  - {file_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Error en reducción de dimensionalidad: {e}")
        raise

if __name__ == "__main__":
    main()


