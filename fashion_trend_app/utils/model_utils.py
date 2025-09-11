"""
Model utilities for Fashion Trend Analysis App
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, model_path: str, clustering_data_dir: str):
        self.model_path = model_path
        self.clustering_data_dir = clustering_data_dir
        self.model = None
        self.feature_extractor = None
        self.clustering_data = None
        self.cluster_centers = None
        self.cluster_stats = None
        
    def load_models(self) -> bool:
        """Load all required models and data"""
        try:
            # Load vision model
            self._load_vision_model()
            
            # Load clustering data
            self._load_clustering_data()
            
            # Calculate cluster centers
            self._calculate_cluster_centers()
            
            # Load cluster statistics
            self._load_cluster_stats()
            
            logger.info("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def _load_vision_model(self):
        """Load the vision model"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Loaded trained model from {self.model_path}")
        else:
            # Fallback to pre-trained MobileNetV2
            self.model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet'
            )
            logger.info("✅ Using pre-trained MobileNetV2")
        
        # Create feature extractor (penultimate layer)
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output
        )
    
    def _load_clustering_data(self):
        """Load clustering data from reports directory"""
        try:
            embeddings_path = os.path.join(self.clustering_data_dir, 'embeddings.npy')
            metadata_path = os.path.join(self.clustering_data_dir, 'metadata.csv')
            assignments_path = os.path.join(self.clustering_data_dir, 'cluster_assignments.csv')
            umap_path = os.path.join(self.clustering_data_dir, 'umap.npy')
            
            self.clustering_data = {
                'embeddings': np.load(embeddings_path),
                'metadata': pd.read_csv(metadata_path),
                'assignments': pd.read_csv(assignments_path),
                'umap': np.load(umap_path) if os.path.exists(umap_path) else None
            }
            
            logger.info("✅ Clustering data loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading clustering data: {e}")
            raise
    
    def _calculate_cluster_centers(self):
        """Calculate cluster centers from embeddings"""
        if self.clustering_data is None:
            return
        
        try:
            embeddings = self.clustering_data['embeddings']
            assignments = self.clustering_data['assignments']
            
            self.cluster_centers = {}
            for cluster_id in assignments['cluster_id'].unique():
                if cluster_id == -1:  # Skip noise
                    continue
                
                cluster_mask = assignments['cluster_id'] == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                
                if len(cluster_embeddings) > 0:
                    self.cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
            
            logger.info(f"✅ Calculated {len(self.cluster_centers)} cluster centers")
            
        except Exception as e:
            logger.error(f"❌ Error calculating cluster centers: {e}")
            raise
    
    def _load_cluster_stats(self):
        """Load cluster statistics"""
        try:
            stats_path = os.path.join(self.clustering_data_dir, 'cluster_stats.json')
            with open(stats_path, 'r') as f:
                self.cluster_stats = json.load(f)
            logger.info("✅ Cluster statistics loaded")
        except Exception as e:
            logger.error(f"❌ Error loading cluster stats: {e}")
            self.cluster_stats = {}
    
    def predict_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
        """Predict on a single image"""
        try:
            # Import here to avoid circular imports
            from fashion_clustering.utils.data import preprocess_image
            
            # Preprocess image
            image = preprocess_image(image_path, target_size=(224, 224))
            image_batch = np.expand_dims(image, axis=0)
            
            # Extract embedding
            embedding = self.feature_extractor.predict(image_batch, verbose=0)[0]
            
            # Get class prediction if model has classification head
            predicted_class = None
            confidence = None
            
            if len(self.model.output_shape) > 1 and self.model.output_shape[1] > 1:
                prediction = self.model.predict(image_batch, verbose=0)[0]
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
            
            return embedding, predicted_class, confidence
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            return None, None, None
    
    def find_closest_cluster(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Find the closest cluster for an embedding"""
        if self.cluster_centers is None:
            return None, 0.0
        
        try:
            min_distance = float('inf')
            closest_cluster = None
            
            for cluster_id, center in self.cluster_centers.items():
                distance = np.linalg.norm(embedding - center)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            # Normalize distance to similarity score (0-100)
            max_distance = 10.0  # Adjust based on dataset
            similarity_score = max(0, 100 - (min_distance / max_distance) * 100)
            
            return closest_cluster, similarity_score
            
        except Exception as e:
            logger.error(f"❌ Error finding closest cluster: {e}")
            return None, 0.0
    
    def calculate_trend_score(self, cluster_id: int, similarity_score: float = None) -> int:
        """Calculate trend score based ONLY on cluster size (new simplified approach)"""
        if self.cluster_stats is None or cluster_id is None:
            return 50  # Neutral score
        
        try:
            # Get cluster statistics
            cluster_sizes = self.cluster_stats.get('cluster_sizes', {})
            cluster_key = f'cluster_{cluster_id}'
            
            if cluster_key in cluster_sizes:
                cluster_size = cluster_sizes[cluster_key]
            else:
                cluster_size = 0
            
            # Calculate trend score based ONLY on cluster size (0-100)
            max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1
            trend_score = round(100 * cluster_size / max_cluster_size)
            
            return int(trend_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating trend score: {e}")
            return 50
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cluster"""
        if self.cluster_stats is None or cluster_id is None:
            return None
        
        try:
            cluster_sizes = self.cluster_stats.get('cluster_sizes', {})
            cluster_key = f'cluster_{cluster_id}'
            
            size = cluster_sizes.get(cluster_key, 0)
            
            return {
                'size': size,
                'description': f'Cluster {cluster_id} with {size} similar images',
                'algorithm': self.cluster_stats.get('algorithm', 'unknown'),
                'silhouette_score': self.cluster_stats.get('silhouette_score', 0),
                'dbi_score': self.cluster_stats.get('dbi_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cluster info: {e}")
            return None
    
    def is_trending(self, cluster_id: int, trend_threshold: int = 60) -> bool:
        """Determine if a cluster is trending based on size"""
        trend_score = self.calculate_trend_score(cluster_id)
        return trend_score >= trend_threshold
    
    def get_trend_info(self, cluster_id: int, trend_threshold: int = 60) -> Dict[str, Any]:
        """Get complete trend information for a cluster"""
        trend_score = self.calculate_trend_score(cluster_id)
        is_trending = trend_score >= trend_threshold
        
        return {
            'cluster_id': cluster_id,
            'trend_score': trend_score,
            'is_trending': is_trending,
            'trend_label': f"En tendencia ({trend_score}/100)" if is_trending else f"No en tendencia ({trend_score}/100)",
            'threshold': trend_threshold
        }

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        if self.cluster_stats is None:
            return {}
        
        try:
            cluster_sizes = self.cluster_stats.get('cluster_sizes', {})
            
            return {
                'total_clusters': len(cluster_sizes),
                'total_images': sum(cluster_sizes.values()),
                'algorithm': self.cluster_stats.get('algorithm', 'unknown'),
                'silhouette_score': self.cluster_stats.get('silhouette_score', 0),
                'dbi_score': self.cluster_stats.get('dbi_score', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting overall stats: {e}")
            return {}


