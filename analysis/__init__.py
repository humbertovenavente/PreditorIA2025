"""
Módulo de análisis de imágenes de moda
Incluye extracción de características y clustering con K-Means
"""

from .feature_extractor import FashionFeatureExtractor
from .clustering import FashionClustering, evaluate_clustering_quality

__all__ = ['FashionFeatureExtractor', 'FashionClustering', 'evaluate_clustering_quality']

