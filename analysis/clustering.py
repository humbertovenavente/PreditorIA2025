#!/usr/bin/env python3
"""
Módulo para clustering de imágenes de moda usando K-Means
Agrupa imágenes en clusters de tendencias similares
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import pickle
import os

logger = logging.getLogger(__name__)

class FashionClustering:
    """
    Clustering de imágenes de moda usando K-Means
    """
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        """
        Inicializa el clustering
        
        Args:
            n_clusters: Número de clusters (K)
            random_state: Semilla para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.features_scaled = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def prepare_features(self, features: np.ndarray, use_pca: bool = True, n_components: int = 50) -> np.ndarray:
        """
        Prepara las características para clustering
        
        Args:
            features: Matriz de características
            use_pca: Si usar PCA para reducción de dimensionalidad
            n_components: Número de componentes para PCA
            
        Returns:
            Características preparadas
        """
        try:
            logger.info(f"Preparando características: {features.shape}")
            
            # Estandarizar características
            self.features_scaled = self.scaler.fit_transform(features)
            logger.info("Características estandarizadas")
            
            # Aplicar PCA si se solicita
            if use_pca and features.shape[1] > n_components:
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                features_processed = self.pca.fit_transform(self.features_scaled)
                logger.info(f"PCA aplicado: {features_processed.shape}")
                logger.info(f"Varianza explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
            else:
                features_processed = self.features_scaled
                logger.info("PCA omitido")
            
            return features_processed
            
        except Exception as e:
            logger.error(f"Error preparando características: {e}")
            raise
    
    def find_optimal_clusters(self, features: np.ndarray, max_clusters: int = 20) -> Dict:
        """
        Encuentra el número óptimo de clusters usando múltiples métricas
        
        Args:
            features: Características preparadas
            max_clusters: Número máximo de clusters a evaluar
            
        Returns:
            Diccionario con métricas y número óptimo de clusters
        """
        try:
            logger.info("Buscando número óptimo de clusters...")
            
            k_range = range(2, min(max_clusters + 1, len(features) // 2))
            inertias = []
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            for k in k_range:
                # Entrenar K-Means
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # Calcular métricas
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(features, cluster_labels))
                calinski_scores.append(calinski_harabasz_score(features, cluster_labels))
                davies_bouldin_scores.append(davies_bouldin_score(features, cluster_labels))
                
                logger.info(f"K={k}: Silhouette={silhouette_scores[-1]:.3f}, "
                          f"Calinski-Harabasz={calinski_scores[-1]:.3f}")
            
            # Encontrar K óptimo basado en silhouette score
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            results = {
                'k_range': list(k_range),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'optimal_k': optimal_k,
                'best_silhouette': max(silhouette_scores)
            }
            
            logger.info(f"Número óptimo de clusters: {optimal_k}")
            logger.info(f"Mejor silhouette score: {max(silhouette_scores):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error encontrando clusters óptimos: {e}")
            raise
    
    def fit_clustering(self, features: np.ndarray, optimal_k: Optional[int] = None) -> np.ndarray:
        """
        Entrena el modelo de clustering
        
        Args:
            features: Características preparadas
            optimal_k: Número de clusters (si None, usa self.n_clusters)
            
        Returns:
            Etiquetas de clusters
        """
        try:
            k = optimal_k if optimal_k is not None else self.n_clusters
            
            logger.info(f"Entrenando K-Means con {k} clusters...")
            
            # Entrenar K-Means
            self.kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            self.cluster_labels = self.kmeans.fit_predict(features)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            # Calcular métricas finales
            silhouette = silhouette_score(features, self.cluster_labels)
            calinski = calinski_harabasz_score(features, self.cluster_labels)
            davies_bouldin = davies_bouldin_score(features, self.cluster_labels)
            
            logger.info(f"Clustering completado:")
            logger.info(f"  Silhouette Score: {silhouette:.3f}")
            logger.info(f"  Calinski-Harabasz Score: {calinski:.3f}")
            logger.info(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
            
            return self.cluster_labels
            
        except Exception as e:
            logger.error(f"Error entrenando clustering: {e}")
            raise
    
    def analyze_clusters(self, image_paths: List[str], cluster_labels: np.ndarray) -> Dict:
        """
        Analiza los clusters resultantes
        
        Args:
            image_paths: Lista de rutas de imágenes
            cluster_labels: Etiquetas de clusters
            
        Returns:
            Diccionario con análisis de clusters
        """
        try:
            logger.info("Analizando clusters...")
            
            # Crear DataFrame para análisis
            df = pd.DataFrame({
                'image_path': image_paths,
                'cluster': cluster_labels
            })
            
            # Estadísticas por cluster
            cluster_stats = df.groupby('cluster').agg({
                'image_path': 'count'
            }).rename(columns={'image_path': 'count'})
            
            cluster_stats['percentage'] = (cluster_stats['count'] / len(df)) * 100
            
            # Análisis de rutas para identificar patrones
            cluster_analysis = {}
            
            for cluster_id in range(self.n_clusters):
                cluster_images = df[df['cluster'] == cluster_id]['image_path'].tolist()
                
                # Extraer información de las rutas
                platforms = []
                for path in cluster_images:
                    if 'instagram' in path.lower():
                        platforms.append('instagram')
                    elif 'web' in path.lower():
                        platforms.append('web')
                    else:
                        platforms.append('unknown')
                
                cluster_analysis[cluster_id] = {
                    'count': len(cluster_images),
                    'percentage': (len(cluster_images) / len(df)) * 100,
                    'platforms': pd.Series(platforms).value_counts().to_dict(),
                    'sample_images': cluster_images[:5]  # Primeras 5 imágenes como muestra
                }
            
            results = {
                'cluster_stats': cluster_stats,
                'cluster_analysis': cluster_analysis,
                'total_images': len(df),
                'n_clusters': len(cluster_stats)
            }
            
            logger.info("Análisis de clusters completado")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analizando clusters: {e}")
            raise
    
    def visualize_clusters(self, features: np.ndarray, cluster_labels: np.ndarray, 
                          output_dir: str = "cluster_visualizations"):
        """
        Crea visualizaciones de los clusters
        
        Args:
            features: Características (2D para visualización)
            cluster_labels: Etiquetas de clusters
            output_dir: Directorio para guardar visualizaciones
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Reducir a 2D si es necesario
            if features.shape[1] > 2:
                pca_2d = PCA(n_components=2, random_state=self.random_state)
                features_2d = pca_2d.fit_transform(features)
            else:
                features_2d = features
            
            # Crear figura con subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Análisis de Clusters de Imágenes de Moda', fontsize=16)
            
            # 1. Scatter plot de clusters
            ax1 = axes[0, 0]
            scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.7)
            ax1.set_title('Distribución de Clusters')
            ax1.set_xlabel('Componente Principal 1')
            ax1.set_ylabel('Componente Principal 2')
            plt.colorbar(scatter, ax=ax1)
            
            # 2. Distribución de clusters
            ax2 = axes[0, 1]
            unique, counts = np.unique(cluster_labels, return_counts=True)
            ax2.bar(unique, counts, color=plt.cm.tab10(unique))
            ax2.set_title('Distribución de Imágenes por Cluster')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Número de Imágenes')
            
            # 3. Distancias al centroide
            ax3 = axes[1, 0]
            distances = []
            for i, label in enumerate(cluster_labels):
                center = self.cluster_centers[label]
                dist = np.linalg.norm(features[i] - center)
                distances.append(dist)
            
            ax3.hist(distances, bins=30, alpha=0.7, color='skyblue')
            ax3.set_title('Distribución de Distancias al Centroide')
            ax3.set_xlabel('Distancia al Centroide')
            ax3.set_ylabel('Frecuencia')
            
            # 4. Métricas de calidad
            ax4 = axes[1, 1]
            silhouette = silhouette_score(features, cluster_labels)
            calinski = calinski_harabasz_score(features, cluster_labels)
            davies_bouldin = davies_bouldin_score(features, cluster_labels)
            
            metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
            values = [silhouette, calinski, davies_bouldin]
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
            ax4.set_title('Métricas de Calidad del Clustering')
            ax4.set_ylabel('Score')
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Guardar visualización
            output_path = os.path.join(output_dir, 'cluster_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualización guardada en: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creando visualización: {e}")
            raise
    
    def save_clustering_results(self, features: np.ndarray, image_paths: List[str], 
                              cluster_labels: np.ndarray, output_path: str):
        """
        Guarda los resultados del clustering
        
        Args:
            features: Características originales
            image_paths: Rutas de imágenes
            cluster_labels: Etiquetas de clusters
            output_path: Ruta donde guardar los resultados
        """
        try:
            results = {
                'features': features,
                'image_paths': image_paths,
                'cluster_labels': cluster_labels,
                'cluster_centers': self.cluster_centers,
                'n_clusters': self.n_clusters,
                'scaler': self.scaler,
                'pca': self.pca,
                'kmeans': self.kmeans
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Resultados del clustering guardados en: {output_path}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
            raise

def evaluate_clustering_quality(features: np.ndarray, cluster_labels: np.ndarray) -> Dict:
    """
    Evalúa la calidad del clustering usando múltiples métricas
    
    Args:
        features: Características
        cluster_labels: Etiquetas de clusters
        
    Returns:
        Diccionario con métricas de calidad
    """
    try:
        metrics = {
            'silhouette_score': silhouette_score(features, cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(features, cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(features, cluster_labels),
            'n_clusters': len(np.unique(cluster_labels)),
            'n_samples': len(features)
        }
        
        # Interpretación de métricas
        if metrics['silhouette_score'] > 0.5:
            metrics['silhouette_interpretation'] = 'Clustering fuerte'
        elif metrics['silhouette_score'] > 0.3:
            metrics['silhouette_interpretation'] = 'Clustering razonable'
        else:
            metrics['silhouette_interpretation'] = 'Clustering débil'
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluando calidad del clustering: {e}")
        return {}

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Ejemplo de uso
    from feature_extractor import FashionFeatureExtractor, load_features
    
    # Cargar características previamente extraídas
    try:
        features, image_paths = load_features("fashion_features.pkl")
        
        # Crear clustering
        clustering = FashionClustering(n_clusters=8)
        
        # Preparar características
        features_processed = clustering.prepare_features(features, use_pca=True, n_components=50)
        
        # Encontrar número óptimo de clusters
        optimal_results = clustering.find_optimal_clusters(features_processed, max_clusters=15)
        
        # Entrenar con número óptimo
        cluster_labels = clustering.fit_clustering(features_processed, optimal_results['optimal_k'])
        
        # Analizar clusters
        analysis = clustering.analyze_clusters(image_paths, cluster_labels)
        
        # Crear visualizaciones
        clustering.visualize_clusters(features_processed, cluster_labels)
        
        # Guardar resultados
        clustering.save_clustering_results(features, image_paths, cluster_labels, "clustering_results.pkl")
        
        print("Clustering completado exitosamente!")
        print(f"Número de clusters: {len(np.unique(cluster_labels))}")
        print(f"Silhouette Score: {optimal_results['best_silhouette']:.3f}")
        
    except FileNotFoundError:
        print("Archivo de características no encontrado. Ejecuta primero feature_extractor.py")
    except Exception as e:
        print(f"Error en clustering: {e}")

