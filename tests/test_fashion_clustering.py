"""
Tests básicos para Fashion Clustering
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Añadir el directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from fashion_clustering.utils.data import create_metadata, preprocess_image
from fashion_clustering.utils.metrics import calculate_clustering_metrics, find_optimal_k
from fashion_clustering.utils.colors import rgb_to_color_name, get_color_names
from fashion_clustering.utils.io import save_embeddings, load_embeddings

class TestDataUtils:
    """Tests para utilidades de datos"""
    
    def test_create_metadata(self):
        """Test creación de metadatos"""
        image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        split_labels = ['train', 'val', 'test']
        category_labels = ['dress', 'top', 'pants']
        
        metadata = create_metadata(image_paths, split_labels, category_labels)
        
        assert len(metadata) == 3
        assert 'image_path' in metadata.columns
        assert 'split' in metadata.columns
        assert 'label' in metadata.columns
        assert 'timestamp' in metadata.columns
        assert 'source' in metadata.columns
    
    def test_create_metadata_with_timestamps(self):
        """Test creación de metadatos con timestamps personalizados"""
        image_paths = ['img1.jpg', 'img2.jpg']
        split_labels = ['train', 'val']
        category_labels = ['dress', 'top']
        timestamps = ['2024-01-01', '2024-01-02']
        sources = ['local', 'global']
        
        metadata = create_metadata(image_paths, split_labels, category_labels, timestamps, sources)
        
        assert metadata['timestamp'].tolist() == timestamps
        assert metadata['source'].tolist() == sources

class TestMetricsUtils:
    """Tests para utilidades de métricas"""
    
    def test_calculate_clustering_metrics(self):
        """Test cálculo de métricas de clustering"""
        # Crear datos de prueba
        embeddings = np.random.rand(100, 10)
        cluster_labels = np.random.randint(0, 5, 100)
        
        metrics = calculate_clustering_metrics(embeddings, cluster_labels, 'kmeans')
        
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'n_clusters' in metrics
        assert 'n_noise' in metrics
        
        # Verificar que las métricas son valores numéricos válidos
        assert isinstance(metrics['silhouette_score'], float)
        assert isinstance(metrics['davies_bouldin_score'], float)
        assert isinstance(metrics['calinski_harabasz_score'], float)
        assert isinstance(metrics['n_clusters'], int)
        assert isinstance(metrics['n_noise'], int)
    
    def test_calculate_clustering_metrics_dbscan(self):
        """Test cálculo de métricas para DBSCAN con ruido"""
        # Crear datos de prueba con ruido
        embeddings = np.random.rand(100, 10)
        cluster_labels = np.random.randint(0, 3, 100)
        cluster_labels[0:10] = -1  # Añadir ruido
        
        metrics = calculate_clustering_metrics(embeddings, cluster_labels, 'dbscan')
        
        assert metrics['n_noise'] == 10
        assert metrics['n_clusters'] == 3
    
    def test_find_optimal_k(self):
        """Test búsqueda de K óptimo"""
        # Crear datos de prueba con clusters claros
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Crear 3 clusters bien separados
        cluster1 = np.random.normal(0, 0.5, (n_samples//3, n_features))
        cluster2 = np.random.normal(5, 0.5, (n_samples//3, n_features))
        cluster3 = np.random.normal(-5, 0.5, (n_samples//3, n_features))
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        
        results = find_optimal_k(embeddings, k_min=2, k_max=10)
        
        assert 'k_range' in results
        assert 'inertias' in results
        assert 'silhouette_scores' in results
        assert 'dbi_scores' in results
        assert 'optimal_k' in results
        
        assert len(results['k_range']) == 9  # 2 a 10 inclusive
        assert len(results['inertias']) == 9
        assert len(results['silhouette_scores']) == 9
        assert len(results['dbi_scores']) == 9
        assert 2 <= results['optimal_k'] <= 10

class TestColorUtils:
    """Tests para utilidades de colores"""
    
    def test_rgb_to_color_name(self):
        """Test conversión RGB a nombre de color"""
        # Test colores básicos
        assert rgb_to_color_name((255, 0, 0)) == 'rojo'
        assert rgb_to_color_name((0, 255, 0)) == 'verde'
        assert rgb_to_color_name((0, 0, 255)) == 'azul'
        assert rgb_to_color_name((255, 255, 255)) == 'blanco'
        assert rgb_to_color_name((0, 0, 0)) == 'negro'
    
    def test_get_color_names(self):
        """Test obtención de nombres de colores"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        names = get_color_names(colors)
        
        assert len(names) == 3
        assert all(isinstance(name, str) for name in names)
        assert 'rojo' in names
        assert 'verde' in names
        assert 'azul' in names

class TestIOUtils:
    """Tests para utilidades de I/O"""
    
    def test_save_load_embeddings(self):
        """Test guardado y carga de embeddings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear datos de prueba
            embeddings = np.random.rand(50, 128)
            metadata = pd.DataFrame({
                'image_path': [f'img_{i}.jpg' for i in range(50)],
                'split': ['train'] * 30 + ['val'] * 20,
                'label': ['dress'] * 25 + ['top'] * 25,
                'timestamp': ['2024-01-01'] * 50,
                'source': ['local'] * 50
            })
            
            # Guardar
            saved_files = save_embeddings(embeddings, metadata, temp_dir)
            
            assert 'embeddings' in saved_files
            assert 'metadata' in saved_files
            
            # Verificar que los archivos existen
            assert os.path.exists(saved_files['embeddings'])
            assert os.path.exists(saved_files['metadata'])
            
            # Cargar
            loaded_embeddings, loaded_metadata = load_embeddings(
                saved_files['embeddings'], saved_files['metadata']
            )
            
            # Verificar que los datos son iguales
            np.testing.assert_array_equal(embeddings, loaded_embeddings)
            pd.testing.assert_frame_equal(metadata, loaded_metadata)

class TestIntegration:
    """Tests de integración"""
    
    def test_end_to_end_workflow(self):
        """Test flujo completo de trabajo"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear datos de prueba
            n_samples = 100
            n_features = 128
            
            embeddings = np.random.rand(n_samples, n_features)
            cluster_labels = np.random.randint(0, 5, n_samples)
            
            metadata = pd.DataFrame({
                'image_path': [f'img_{i}.jpg' for i in range(n_samples)],
                'split': ['train'] * 70 + ['val'] * 30,
                'label': ['dress'] * 50 + ['top'] * 50,
                'timestamp': ['2024-01-01'] * n_samples,
                'source': ['local'] * n_samples
            })
            
            # Guardar embeddings
            save_embeddings(embeddings, metadata, temp_dir)
            
            # Cargar embeddings
            loaded_embeddings, loaded_metadata = load_embeddings(
                f"{temp_dir}/embeddings.npy",
                f"{temp_dir}/metadata.csv"
            )
            
            # Calcular métricas
            metrics = calculate_clustering_metrics(loaded_embeddings, cluster_labels, 'kmeans')
            
            # Verificar que todo funciona
            assert loaded_embeddings.shape == (n_samples, n_features)
            assert len(loaded_metadata) == n_samples
            assert 'silhouette_score' in metrics

if __name__ == "__main__":
    pytest.main([__file__])
