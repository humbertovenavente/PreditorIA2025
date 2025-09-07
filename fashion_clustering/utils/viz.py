"""
Módulo para visualizaciones de clustering
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_umap_clusters(umap_embeddings: np.ndarray,
                      cluster_labels: np.ndarray,
                      save_path: str = "reports/umap_clusters.png",
                      title: str = "Clusters en espacio UMAP",
                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza clusters en espacio UMAP
    
    Args:
        umap_embeddings: Embeddings UMAP (N, 2)
        cluster_labels: Etiquetas de cluster
        save_path: Ruta para guardar la imagen
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    try:
        plt.figure(figsize=figsize)
        
        # Filtrar ruido si es DBSCAN
        if -1 in cluster_labels:
            mask = cluster_labels != -1
            filtered_embeddings = umap_embeddings[mask]
            filtered_labels = cluster_labels[mask]
            
            # Plotear puntos de ruido en gris
            noise_mask = cluster_labels == -1
            if np.any(noise_mask):
                plt.scatter(umap_embeddings[noise_mask, 0], 
                           umap_embeddings[noise_mask, 1], 
                           c='gray', alpha=0.3, s=20, label='Ruido')
        else:
            filtered_embeddings = umap_embeddings
            filtered_labels = cluster_labels
        
        # Plotear clusters
        unique_labels = np.unique(filtered_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = filtered_labels == label
            plt.scatter(filtered_embeddings[mask, 0], 
                       filtered_embeddings[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', 
                       alpha=0.7, s=30)
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Guardar
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico UMAP guardado en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando gráfico UMAP: {e}")

def plot_elbow_curve(k_range: List[int],
                    inertias: List[float],
                    save_path: str = "reports/elbow.png",
                    title: str = "Método del Codo - K-Means",
                    figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualiza la curva del codo para K-Means
    
    Args:
        k_range: Rango de valores de K
        inertias: Valores de inercia
        save_path: Ruta para guardar la imagen
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    try:
        plt.figure(figsize=figsize)
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inercia')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Destacar el punto óptimo si está disponible
        if len(k_range) > 1:
            # Calcular segunda derivada para encontrar el codo
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 1
                if elbow_idx < len(k_range):
                    plt.axvline(x=k_range[elbow_idx], color='red', linestyle='--', 
                               alpha=0.7, label=f'Punto del codo (K={k_range[elbow_idx]})')
                    plt.legend()
        
        # Guardar
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Curva del codo guardada en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando curva del codo: {e}")

def plot_silhouette_analysis(k_range: List[int],
                            silhouette_scores: List[float],
                            dbi_scores: List[float],
                            save_path: str = "reports/silhouette_analysis.png",
                            title: str = "Análisis de Silhouette y DBI",
                            figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Visualiza análisis de silhouette y DBI
    
    Args:
        k_range: Rango de valores de K
        silhouette_scores: Puntuaciones de silhouette
        dbi_scores: Puntuaciones de DBI
        save_path: Ruta para guardar la imagen
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Gráfico de Silhouette
        ax1.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters (K)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score vs K')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Umbral (0.3)')
        ax1.legend()
        
        # Gráfico de DBI
        ax2.plot(k_range, dbi_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters (K)')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('DBI vs K')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Umbral (2.0)')
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Guardar
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Análisis de silhouette guardado en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando análisis de silhouette: {e}")

def plot_cluster_sizes(cluster_labels: np.ndarray,
                      save_path: str = "reports/cluster_sizes.png",
                      title: str = "Tamaños de Clusters",
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualiza tamaños de clusters
    
    Args:
        cluster_labels: Etiquetas de cluster
        save_path: Ruta para guardar la imagen
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    try:
        # Contar tamaños de clusters
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        # Filtrar ruido si existe
        if -1 in unique_labels:
            noise_idx = np.where(unique_labels == -1)[0][0]
            noise_count = counts[noise_idx]
            unique_labels = np.delete(unique_labels, noise_idx)
            counts = np.delete(counts, noise_idx)
        else:
            noise_count = 0
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(unique_labels)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))))
        
        # Añadir etiquetas en las barras
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.xlabel('Cluster ID')
        plt.ylabel('Número de Imágenes')
        plt.title(title)
        plt.xticks(range(len(unique_labels)), [f'C{label}' for label in unique_labels])
        plt.grid(True, alpha=0.3, axis='y')
        
        # Añadir información de ruido si existe
        if noise_count > 0:
            plt.text(0.02, 0.98, f'Ruido: {noise_count} imágenes', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Guardar
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de tamaños de clusters guardado en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando gráfico de tamaños: {e}")

def plot_cluster_prototypes(prototypes: Dict[int, List[str]],
                           save_dir: str = "reports",
                           images_per_row: int = 5,
                           figsize: Tuple[int, int] = (15, 3)) -> None:
    """
    Crea mosaicos de prototipos por cluster
    
    Args:
        prototypes: Diccionario {cluster_id: [image_paths]}
        save_dir: Directorio para guardar las imágenes
        images_per_row: Imágenes por fila
        figsize: Tamaño de la figura
    """
    try:
        from PIL import Image
        
        for cluster_id, image_paths in prototypes.items():
            if not image_paths:
                continue
            
            n_images = len(image_paths)
            n_rows = (n_images + images_per_row - 1) // images_per_row
            
            fig, axes = plt.subplots(n_rows, images_per_row, figsize=figsize)
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, img_path in enumerate(image_paths):
                row = i // images_per_row
                col = i % images_per_row
                
                try:
                    img = Image.open(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f'Prototipo {i+1}')
                    axes[row, col].axis('off')
                except Exception as e:
                    logger.warning(f"Error cargando {img_path}: {e}")
                    axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                    axes[row, col].axis('off')
            
            # Ocultar ejes vacíos
            for i in range(n_images, n_rows * images_per_row):
                row = i // images_per_row
                col = i % images_per_row
                axes[row, col].axis('off')
            
            plt.suptitle(f'Prototipos del Cluster {cluster_id}', fontsize=16)
            plt.tight_layout()
            
            # Guardar
            save_path = f"{save_dir}/cluster_{cluster_id}_prototypes.jpg"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Prototipos del cluster {cluster_id} guardados en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando mosaicos de prototipos: {e}")

def plot_temporal_trends(trends_data: pd.DataFrame,
                        save_path: str = "reports/cluster_trends.png",
                        title: str = "Tendencias Temporales de Clusters",
                        figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Visualiza tendencias temporales de clusters
    
    Args:
        trends_data: DataFrame con datos temporales
        save_path: Ruta para guardar la imagen
        title: Título del gráfico
        figsize: Tamaño de la figura
    """
    try:
        plt.figure(figsize=figsize)
        
        # Plotear líneas para cada cluster
        for cluster_id in trends_data.columns:
            if cluster_id != 'month':
                plt.plot(trends_data['month'], trends_data[cluster_id], 
                        marker='o', linewidth=2, label=f'Cluster {cluster_id}')
        
        plt.xlabel('Mes')
        plt.ylabel('Proporción de Imágenes')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Guardar
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Tendencias temporales guardadas en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando gráfico de tendencias: {e}")

def create_summary_plot(metrics: Dict[str, Any],
                       save_path: str = "reports/clustering_summary.png",
                       figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Crea gráfico resumen del clustering
    
    Args:
        metrics: Diccionario con métricas
        save_path: Ruta para guardar la imagen
        figsize: Tamaño de la figura
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Métricas principales
        ax1 = axes[0, 0]
        metric_names = ['Silhouette', 'DBI', 'CHI']
        metric_values = [
            metrics.get('silhouette_score', 0),
            metrics.get('davies_bouldin_score', 0),
            metrics.get('calinski_harabasz_score', 0)
        ]
        bars = ax1.bar(metric_names, metric_values, color=['green', 'red', 'blue'])
        ax1.set_title('Métricas de Calidad')
        ax1.set_ylabel('Valor')
        
        # Añadir valores en las barras
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Distribución de clusters
        ax2 = axes[0, 1]
        cluster_sizes = metrics.get('cluster_sizes', {})
        if cluster_sizes:
            clusters = list(cluster_sizes.keys())
            sizes = list(cluster_sizes.values())
            ax2.bar(clusters, sizes, color=plt.cm.Set3(np.linspace(0, 1, len(clusters))))
            ax2.set_title('Distribución de Clusters')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Número de Imágenes')
        
        # Información del algoritmo
        ax3 = axes[1, 0]
        ax3.text(0.1, 0.8, f"Algoritmo: {metrics.get('algorithm', 'N/A')}", 
                transform=ax3.transAxes, fontsize=12, weight='bold')
        ax3.text(0.1, 0.7, f"Número de clusters: {metrics.get('n_clusters', 'N/A')}", 
                transform=ax3.transAxes, fontsize=12)
        ax3.text(0.1, 0.6, f"Total de imágenes: {metrics.get('total_images', 'N/A')}", 
                transform=ax3.transAxes, fontsize=12)
        ax3.text(0.1, 0.5, f"Imágenes de ruido: {metrics.get('n_noise', 'N/A')}", 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Información del Clustering')
        ax3.axis('off')
        
        # Parámetros utilizados
        ax4 = axes[1, 1]
        params = metrics.get('parameters', {})
        param_text = "Parámetros:\n"
        for key, value in params.items():
            param_text += f"{key}: {value}\n"
        ax4.text(0.1, 0.9, param_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Parámetros Utilizados')
        ax4.axis('off')
        
        plt.suptitle('Resumen del Clustering de Estilos', fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Guardar
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico resumen guardado en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando gráfico resumen: {e}")


