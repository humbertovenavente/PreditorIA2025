"""
Fashion Clustering Package
Módulo de clustering de estilos para análisis de moda guatemalteca
"""

__version__ = "1.0.0"
__author__ = "PreditorIA2025"

from .utils.data import load_dataset, preprocess_image
from .utils.vision import load_mobilenetv2, extract_embeddings
from .utils.metrics import calculate_clustering_metrics
from .utils.viz import plot_umap_clusters, plot_elbow_curve
from .utils.colors import extract_dominant_colors
from .utils.io import save_embeddings, load_embeddings
from .utils.time import analyze_temporal_trends

__all__ = [
    'load_dataset',
    'preprocess_image', 
    'load_mobilenetv2',
    'extract_embeddings',
    'calculate_clustering_metrics',
    'plot_umap_clusters',
    'plot_elbow_curve',
    'extract_dominant_colors',
    'save_embeddings',
    'load_embeddings',
    'analyze_temporal_trends'
]


