"""
Módulo para entrada/salida de datos
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

def save_embeddings(embeddings: np.ndarray,
                   metadata: pd.DataFrame,
                   save_dir: str = "reports") -> Dict[str, str]:
    """
    Guarda embeddings y metadatos
    
    Args:
        embeddings: Array de embeddings
        metadata: DataFrame con metadatos
        save_dir: Directorio para guardar
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar embeddings
        embeddings_path = save_path / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # Guardar metadatos
        metadata_path = save_path / "metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        
        logger.info(f"Embeddings guardados en {embeddings_path}")
        logger.info(f"Metadatos guardados en {metadata_path}")
        
        return {
            'embeddings': str(embeddings_path),
            'metadata': str(metadata_path)
        }
        
    except Exception as e:
        logger.error(f"Error guardando embeddings: {e}")
        raise

def load_embeddings(embeddings_path: str,
                   metadata_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Carga embeddings y metadatos
    
    Args:
        embeddings_path: Ruta a embeddings.npy
        metadata_path: Ruta a metadata.csv
    
    Returns:
        Tuple con (embeddings, metadata)
    """
    try:
        # Cargar embeddings
        embeddings = np.load(embeddings_path)
        
        # Cargar metadatos
        metadata = pd.read_csv(metadata_path)
        
        logger.info(f"Embeddings cargados: {embeddings.shape}")
        logger.info(f"Metadatos cargados: {metadata.shape}")
        
        return embeddings, metadata
        
    except Exception as e:
        logger.error(f"Error cargando embeddings: {e}")
        raise

def save_clustering_results(cluster_labels: np.ndarray,
                          metrics: Dict[str, Any],
                          algorithm: str,
                          parameters: Dict[str, Any],
                          save_dir: str = "reports") -> Dict[str, str]:
    """
    Guarda resultados de clustering
    
    Args:
        cluster_labels: Etiquetas de cluster
        metrics: Métricas de clustering
        algorithm: Algoritmo utilizado
        parameters: Parámetros utilizados
        save_dir: Directorio para guardar
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Crear DataFrame de asignaciones
        assignments_df = pd.DataFrame({
            'cluster_id': cluster_labels
        })
        
        # Añadir scores individuales si están disponibles
        if 'individual_silhouette' in metrics:
            assignments_df['silhouette_score'] = metrics['individual_silhouette']
        
        # Guardar asignaciones
        assignments_path = save_path / "cluster_assignments.csv"
        assignments_df.to_csv(assignments_path, index=False)
        
        # Crear estadísticas de clusters
        cluster_stats = {
            'algorithm': algorithm,
            'parameters': parameters,
            'metrics': metrics,
            'n_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
            'n_noise': np.sum(cluster_labels == -1) if -1 in cluster_labels else 0,
            'total_images': len(cluster_labels)
        }
        
        # Añadir tamaños de clusters
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = {}
        for label, count in zip(unique_labels, counts):
            if label == -1:
                cluster_sizes['noise'] = int(count)
            else:
                cluster_sizes[f'cluster_{label}'] = int(count)
        
        cluster_stats['cluster_sizes'] = cluster_sizes
        
        # Convertir tipos numpy a Python nativo para JSON
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        cluster_stats = convert_numpy_types(cluster_stats)
        
        # Guardar estadísticas
        stats_path = save_path / "cluster_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Asignaciones guardadas en {assignments_path}")
        logger.info(f"Estadísticas guardadas en {stats_path}")
        
        return {
            'assignments': str(assignments_path),
            'stats': str(stats_path)
        }
        
    except Exception as e:
        logger.error(f"Error guardando resultados de clustering: {e}")
        raise

def load_clustering_results(assignments_path: str,
                          stats_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Carga resultados de clustering
    
    Args:
        assignments_path: Ruta a cluster_assignments.csv
        stats_path: Ruta a cluster_stats.json
    
    Returns:
        Tuple con (cluster_labels, stats)
    """
    try:
        # Cargar asignaciones
        assignments_df = pd.read_csv(assignments_path)
        cluster_labels = assignments_df['cluster_id'].values
        
        # Cargar estadísticas
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        logger.info(f"Resultados de clustering cargados: {len(cluster_labels)} imágenes")
        logger.info(f"Algoritmo: {stats.get('algorithm', 'N/A')}")
        
        return cluster_labels, stats
        
    except Exception as e:
        logger.error(f"Error cargando resultados de clustering: {e}")
        raise

def save_dimensionality_reduction(pca_embeddings: Optional[np.ndarray] = None,
                                 umap_embeddings: Optional[np.ndarray] = None,
                                 save_dir: str = "reports") -> Dict[str, str]:
    """
    Guarda resultados de reducción de dimensionalidad
    
    Args:
        pca_embeddings: Embeddings PCA
        umap_embeddings: Embeddings UMAP
        save_dir: Directorio para guardar
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        if pca_embeddings is not None:
            pca_path = save_path / "pca.npy"
            np.save(pca_path, pca_embeddings)
            saved_files['pca'] = str(pca_path)
            logger.info(f"Embeddings PCA guardados en {pca_path}")
        
        if umap_embeddings is not None:
            umap_path = save_path / "umap.npy"
            np.save(umap_path, umap_embeddings)
            saved_files['umap'] = str(umap_path)
            logger.info(f"Embeddings UMAP guardados en {umap_path}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"Error guardando reducción de dimensionalidad: {e}")
        raise

def load_dimensionality_reduction(save_dir: str = "reports") -> Dict[str, np.ndarray]:
    """
    Carga resultados de reducción de dimensionalidad
    
    Args:
        save_dir: Directorio donde están guardados
    
    Returns:
        Diccionario con embeddings cargados
    """
    try:
        save_path = Path(save_dir)
        embeddings = {}
        
        # Cargar PCA si existe
        pca_path = save_path / "pca.npy"
        if pca_path.exists():
            embeddings['pca'] = np.load(pca_path)
            logger.info(f"Embeddings PCA cargados: {embeddings['pca'].shape}")
        
        # Cargar UMAP si existe
        umap_path = save_path / "umap.npy"
        if umap_path.exists():
            embeddings['umap'] = np.load(umap_path)
            logger.info(f"Embeddings UMAP cargados: {embeddings['umap'].shape}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error cargando reducción de dimensionalidad: {e}")
        return {}

def save_temporal_analysis(trends_data: pd.DataFrame,
                          local_global_stats: Dict[str, Any],
                          save_dir: str = "reports") -> Dict[str, str]:
    """
    Guarda análisis temporal y local vs global
    
    Args:
        trends_data: DataFrame con tendencias temporales
        local_global_stats: Estadísticas local vs global
        save_dir: Directorio para guardar
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Guardar tendencias temporales
        if not trends_data.empty:
            trends_path = save_path / "cluster_trends.csv"
            trends_data.to_csv(trends_path, index=False)
            saved_files['trends'] = str(trends_path)
            logger.info(f"Tendencias temporales guardadas en {trends_path}")
        
        # Guardar estadísticas local vs global
        if local_global_stats:
            stats_path = save_path / "local_global_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(local_global_stats, f, indent=2, ensure_ascii=False)
            saved_files['local_global'] = str(stats_path)
            logger.info(f"Estadísticas local vs global guardadas en {stats_path}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"Error guardando análisis temporal: {e}")
        raise

def save_cluster_summary(summary_data: Dict[str, Any],
                        save_path: str = "reports/cluster_summary.md") -> str:
    """
    Guarda resumen de clusters en formato Markdown
    
    Args:
        summary_data: Datos del resumen
        save_path: Ruta para guardar el archivo
    
    Returns:
        Ruta del archivo guardado
    """
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# Resumen de Clusters de Estilos\n\n")
            f.write(f"**Fecha de análisis:** {summary_data.get('timestamp', 'N/A')}\n")
            f.write(f"**Algoritmo:** {summary_data.get('algorithm', 'N/A')}\n")
            f.write(f"**Total de imágenes:** {summary_data.get('total_images', 'N/A')}\n")
            f.write(f"**Número de clusters:** {summary_data.get('n_clusters', 'N/A')}\n\n")
            
            # Métricas
            metrics = summary_data.get('metrics', {})
            f.write("## Métricas de Calidad\n\n")
            f.write(f"- **Silhouette Score:** {metrics.get('silhouette_score', 'N/A'):.3f}\n")
            f.write(f"- **Davies-Bouldin Index:** {metrics.get('davies_bouldin_score', 'N/A'):.3f}\n")
            f.write(f"- **Calinski-Harabasz Index:** {metrics.get('calinski_harabasz_score', 'N/A'):.3f}\n\n")
            
            # Clusters individuales
            clusters = summary_data.get('clusters', {})
            f.write("## Análisis por Cluster\n\n")
            
            for cluster_id, cluster_info in clusters.items():
                f.write(f"### Cluster {cluster_id}\n\n")
                f.write(f"**Tamaño:** {cluster_info.get('size', 'N/A')} imágenes\n")
                f.write(f"**Colores dominantes:** {cluster_info.get('colors', 'N/A')}\n")
                f.write(f"**Palabras clave:** {cluster_info.get('keywords', 'N/A')}\n")
                f.write(f"**Etiqueta sugerida:** {cluster_info.get('suggested_label', 'N/A')}\n\n")
                
                # Prototipos
                prototypes = cluster_info.get('prototypes', [])
                if prototypes:
                    f.write("**Prototipos:**\n")
                    for i, prototype in enumerate(prototypes[:10], 1):
                        f.write(f"{i}. {prototype}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        logger.info(f"Resumen de clusters guardado en {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error guardando resumen de clusters: {e}")
        raise

def check_existing_files(save_dir: str = "reports") -> Dict[str, bool]:
    """
    Verifica qué archivos ya existen
    
    Args:
        save_dir: Directorio a verificar
    
    Returns:
        Diccionario con estado de archivos
    """
    try:
        save_path = Path(save_dir)
        
        files_to_check = [
            'embeddings.npy',
            'metadata.csv',
            'pca.npy',
            'umap.npy',
            'cluster_assignments.csv',
            'cluster_stats.json',
            'cluster_trends.csv',
            'local_global_stats.json',
            'cluster_summary.md'
        ]
        
        existing_files = {}
        for filename in files_to_check:
            file_path = save_path / filename
            existing_files[filename] = file_path.exists()
        
        return existing_files
        
    except Exception as e:
        logger.error(f"Error verificando archivos existentes: {e}")
        return {}

def cleanup_old_files(save_dir: str = "reports", 
                     keep_embeddings: bool = True) -> None:
    """
    Limpia archivos antiguos (opcional)
    
    Args:
        save_dir: Directorio a limpiar
        keep_embeddings: Si mantener embeddings
    """
    try:
        save_path = Path(save_dir)
        
        files_to_remove = [
            'pca.npy',
            'umap.npy',
            'cluster_assignments.csv',
            'cluster_stats.json',
            'cluster_trends.csv',
            'local_global_stats.json',
            'cluster_summary.md'
        ]
        
        if not keep_embeddings:
            files_to_remove.extend(['embeddings.npy', 'metadata.csv'])
        
        for filename in files_to_remove:
            file_path = save_path / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Archivo eliminado: {filename}")
        
    except Exception as e:
        logger.error(f"Error limpiando archivos: {e}")
