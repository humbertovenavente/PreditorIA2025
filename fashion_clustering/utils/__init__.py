"""
Módulos utilitarios para Fashion Clustering
"""

from .data import load_dataset, preprocess_image, create_metadata
from .vision import load_mobilenetv2, extract_embeddings
from .metrics import calculate_clustering_metrics, find_optimal_k
from .viz import plot_umap_clusters, plot_elbow_curve, plot_silhouette_analysis
from .colors import extract_dominant_colors, get_color_names
from .io import save_embeddings, load_embeddings
from .time import analyze_temporal_trends, compare_local_global

__all__ = [
    'load_dataset',
    'preprocess_image',
    'create_metadata',
    'load_mobilenetv2',
    'extract_embeddings',
    'calculate_clustering_metrics',
    'find_optimal_k',
    'plot_umap_clusters',
    'plot_elbow_curve',
    'plot_silhouette_analysis',
    'extract_dominant_colors',
    'get_color_names',
    'save_embeddings',
    'load_embeddings',
    'save_results',
    'analyze_temporal_trends',
    'compare_local_global'
]
