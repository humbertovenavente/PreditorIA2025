
import os
import pickle
import numpy as np
import tensorflow as tf
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FashionTrendAnalyzer:
    """
    Analizador de tendencias de moda que combina modelo H5 y K-means
    usando fórmula heurística: (h5_score * 0.6) + (kmeans_score * 0.4)
    """
    
    def __init__(self, model_path: str, kmeans_path: str, clustering_results_path: str):
        self.model_path = model_path
        self.kmeans_path = kmeans_path
        self.clustering_results_path = clustering_results_path
        
        # Modelos cargados
        self.h5_model = None
        self.kmeans_model = None
        self.pca_model = None
        self.clustering_data = None
        self.cluster_stats = None
        
        # Estadísticas del dataset
        self.total_images = 0
        self.cluster_sizes = {}
        
    def load_models(self) -> bool:
        """Cargar todos los modelos necesarios"""
        try:
            logger.info("Cargando modelos para análisis de tendencias...")
            
            # Cargar modelo H5
            self._load_h5_model()
            
            # Cargar modelo K-means
            self._load_kmeans_model()
            
            # Cargar modelo PCA
            self._load_pca_model()
            
            # Cargar datos de clustering
            self._load_clustering_data()
            
            # Calcular estadísticas de clusters
            self._calculate_cluster_stats()
            
            logger.info(" Todos los modelos cargados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f" Error cargando modelos: {e}")
            return False
    
    def _load_h5_model(self):
        """Cargar modelo H5 entrenado"""
        try:
            if os.path.exists(self.model_path):
                # Cargar el modelo completo
                self.h5_model = tf.keras.models.load_model(self.model_path)
                logger.info(f" Modelo H5 cargado desde {self.model_path}")
                logger.info(f" Salida del modelo: {self.h5_model.output_shape}")
                
            else:
                # Fallback a MobileNetV2 preentrenado con 11 clases
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
                # Agregar capas de clasificación para 11 clases
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                predictions = tf.keras.layers.Dense(11, activation='softmax')(x)
                
                self.h5_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                logger.info("Usando MobileNetV2 modificado como fallback")
        except Exception as e:
            logger.error(f"Error cargando modelo H5: {e}")
            raise
    
    def _load_kmeans_model(self):
        """Cargar modelo K-means"""
        try:
            with open(self.kmeans_path, 'rb') as f:
                self.kmeans_model = pickle.load(f)
            logger.info(f"Modelo K-means cargado desde {self.kmeans_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo K-means: {e}")
            raise
    
    def _load_pca_model(self):
        """Cargar modelo PCA"""
        try:
            pca_path = 'models/pca_model.pkl'
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            logger.info(f"Modelo PCA cargado desde {pca_path}")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo PCA: {e}")
            self.pca_model = None
    
    def _load_clustering_data(self):
        """Cargar datos de clustering"""
        try:
            with open(self.clustering_results_path, 'rb') as f:
                self.clustering_data = pickle.load(f)
            logger.info(f"Datos de clustering cargados desde {self.clustering_results_path}")
        except Exception as e:
            logger.error(f"Error cargando datos de clustering: {e}")
            raise
    
    def _calculate_cluster_stats(self):
        """Calcular estadísticas de clusters"""
        try:
            if self.clustering_data is None:
                return
            
            # Obtener información de clusters
            cluster_labels = self.clustering_data.get('cluster_labels', [])
            self.total_images = len(cluster_labels)
            
            # Contar imágenes por cluster
            unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
            self.cluster_sizes = dict(zip(unique_clusters, counts))
            
            logger.info(f"Estadísticas calculadas: {len(self.cluster_sizes)} clusters, {self.total_images} imágenes")
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocesar imagen para análisis"""
        try:
            from PIL import Image
            
            # Cargar imagen
            img = Image.open(image_path)
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar
            img = img.resize((224, 224))
            
            # Convertir a array
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Normalización (0-1)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {e}")
            return None
    
    def calculate_h5_score(self, image_path: str) -> float:
        """
        Calcular score del modelo H5 (0-1)
        Usa la confianza de la predicción como score
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_path)
            if image is None:
                return 0.0
            
            # Preparar batch
            image_batch = np.expand_dims(image, axis=0)
            
            # Obtener predicción del modelo H5
            prediction = self.h5_model.predict(image_batch, verbose=0)[0]
            
            # Obtener confianza (probabilidad máxima)
            h5_score = float(np.max(prediction))
            
            logger.info(f" Score H5: {h5_score:.4f}")
            return h5_score
            
        except Exception as e:
            logger.error(f"Error calculando score H5: {e}")
            return 0.0
    
    def calculate_kmeans_score(self, image_path: str) -> float:
        """
        Calcular score del K-means (0-1)
        Usa la proporción: imágenes_en_cluster / total_imágenes
        """
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_path)
            if image is None:
                return 0.0
            
            # Preparar batch
            image_batch = np.expand_dims(image, axis=0)
            
            # Extraer embedding de la penúltima capa (1,280 dimensiones) según metodología real
            try:
                # Crear modelo intermedio para extraer embeddings de la penúltima capa
                from tensorflow.keras.models import Model
                intermediate_model = Model(inputs=self.h5_model.input, outputs=self.h5_model.layers[-2].output)
                embedding_1280 = intermediate_model.predict(image_batch, verbose=0)[0]
                
                # Aplicar PCA para reducir de 1,280 a 50 dimensiones (según metodología)
                if hasattr(self, 'pca_model') and self.pca_model is not None:
                    embedding = self.pca_model.transform([embedding_1280])[0].astype(np.float64)
                    logger.info(f"Embedding PCA: {embedding.shape} (1,280 -> 50 dimensiones)")
                else:
                    # Fallback: usar solo las primeras 50 dimensiones si no hay PCA
                    embedding = embedding_1280[:50].astype(np.float64)
                    logger.warning("Usando fallback sin PCA - solo primeras 50 dimensiones")
                    
            except Exception as e:
                logger.warning(f"Error extrayendo embedding de penúltima capa: {e}")
                # Fallback: usar la salida final
                prediction = self.h5_model.predict(image_batch, verbose=0)[0]
                expected_dims = self.kmeans_model.cluster_centers_.shape[1]
                if len(prediction) >= expected_dims:
                    embedding = prediction[:expected_dims].astype(np.float64)
                else:
                    embedding = np.zeros(expected_dims, dtype=np.float64)
                    embedding[:len(prediction)] = prediction.astype(np.float64)
            
            embedding = embedding.reshape(1, -1)  # Asegurar forma (1, n_features)
            
            try:
                # Predecir cluster
                predicted_cluster = self.kmeans_model.predict(embedding)[0]
                logger.info(f"Cluster predicho correctamente: {predicted_cluster}")
                
            except Exception as cluster_error:
                logger.error(f" Error en predicción de cluster: {cluster_error}")
                logger.error(f"   Embedding shape después de reshape: {embedding.shape}")
                logger.error(f"   K-means cluster centers shape: {self.kmeans_model.cluster_centers_.shape}")
                
                # Fallback más inteligente usando distancia a centroides
                if hasattr(self.kmeans_model, 'cluster_centers_'):
                    try:
                        distances = np.linalg.norm(self.kmeans_model.cluster_centers_ - embedding.flatten(), axis=1)
                        predicted_cluster = int(np.argmin(distances))
                        logger.warning(f" Usando fallback por distancia: cluster {predicted_cluster}")
                    except Exception as distance_error:
                        logger.error(f"Error en fallback por distancia: {distance_error}")
                        # Último recurso
                        predicted_cluster = int(np.sum(embedding.flatten()) % len(self.cluster_sizes))
                        logger.warning(f"Usando fallback aleatorio: cluster {predicted_cluster}")
                else:
                    # Último recurso
                    predicted_cluster = int(np.sum(embedding.flatten()) % len(self.cluster_sizes))
                    logger.warning(f" Usando fallback aleatorio: cluster {predicted_cluster}")
            
            # Calcular score K-means mejorado basado en el cluster más grande
            cluster_size = self.cluster_sizes.get(predicted_cluster, 0)
            
            if self.total_images > 0 and len(self.cluster_sizes) > 0:
                # Usar el cluster más grande como referencia (más justo)
                max_cluster_size = max(self.cluster_sizes.values())
                
                # NUEVO SCORE K-MEANS MÁS GENEROSO Y INTELIGENTE
                relative_size = cluster_size / max_cluster_size if max_cluster_size > 0 else 0
                
                # 1. Score base más suave (menos estricto)
                base_score = relative_size ** 0.4  # Exponente más bajo = más generoso
                
                # 2. Bonus por cluster decente (no solo los gigantes)
                if cluster_size >= 30:  # Clusters grandes
                    size_bonus = 0.3
                elif cluster_size >= 15:  # Clusters medianos
                    size_bonus = 0.2
                elif cluster_size >= 5:  # Clusters pequeños pero válidos
                    size_bonus = 0.15
                else:
                    size_bonus = 0.1  # Incluso clusters muy pequeños tienen valor
                
                # 3. Bonus por diversidad (no solo el cluster #1 es trendy)
                total_clusters = len(self.cluster_sizes)
                sorted_sizes = sorted(self.cluster_sizes.values(), reverse=True)
                cluster_rank = sorted_sizes.index(cluster_size) + 1 if cluster_size in sorted_sizes else total_clusters
                
                if cluster_rank <= 10:  # Top 10 clusters
                    diversity_bonus = 0.2
                elif cluster_rank <= 30:  # Top 30 clusters
                    diversity_bonus = 0.15
                elif cluster_rank <= 60:  # Top 60 clusters
                    diversity_bonus = 0.1
                else:
                    diversity_bonus = 0.05
                
                # 4. Score final más balanceado
                kmeans_score = base_score + size_bonus + diversity_bonus
                
                # 5. Asegurar rango válido con mínimo más generoso
                kmeans_score = max(0.15, min(1.0, kmeans_score))  # Mínimo 0.15 (mucho más generoso)
            else:
                kmeans_score = 0.0
            
            logger.info(f"Score K-means mejorado: {kmeans_score:.4f} (cluster {predicted_cluster}, {cluster_size} imágenes, tamaño relativo: {relative_size:.2f})")
            return kmeans_score
            
        except Exception as e:
            logger.error(f" Error calculando score K-means: {e}")
            return 0.0
    
    def calculate_combined_trend_score(self, image_path: str) -> Dict[str, Any]:
        """
        Calcular score de tendencia combinado usando fórmula INTELIGENTE y BALANCEADA
        que aprovecha tanto el modelo entrenado como los clusters
        """
        try:
            logger.info(f"Analizando tendencia para: {os.path.basename(image_path)}")
            
            # Calcular scores individuales
            h5_score = self.calculate_h5_score(image_path)
            kmeans_score = self.calculate_kmeans_score(image_path)
            
            # NUEVA FÓRMULA INTELIGENTE Y BALANCEADA
            # 1. Score base del modelo H5 (más peso porque está entrenado específicamente)
            h5_weight = 0.7  # Más peso al modelo entrenado
            kmeans_weight = 0.3  # Menos peso al clustering
            
            # 2. Aplicar pesos balanceados
            weighted_score = (h5_score * h5_weight) + (kmeans_score * kmeans_weight)
            
            # 3. Boost inteligente: si ambos scores son buenos, dar bonus
            if h5_score > 0.15 and kmeans_score > 0.05:
                synergy_bonus = min(0.15, (h5_score * kmeans_score) * 2)  # Bonus por sinergia
                weighted_score += synergy_bonus
            
            # 4. Normalización inteligente para distribución más natural
            # Usar función sigmoide para suavizar la curva
            import math
            normalized_score = 1 / (1 + math.exp(-4 * (weighted_score - 0.3)))
            
            # 5. Score final combinado (híbrido entre weighted y normalizado)
            combined_score = (weighted_score * 0.6) + (normalized_score * 0.4)
            
            # 6. UMBRAL FIJO A 60% - Como solicitado por el usuario
            threshold = 0.6  # Solo "EN TENDENCIA" si es mayor a 60%
            
            # 7. Determinar tendencia con umbral dinámico
            is_trending = combined_score >= threshold
            
            # 8. Etiqueta con umbral fijo de 60%
            if combined_score >= 0.6:
                trend_label = "EN TENDENCIA"
            else:
                trend_label = "NO EN TENDENCIA"
            
            # Obtener información adicional del cluster
            cluster_info = self._get_cluster_info(image_path)
            logger.info(f"DEBUG: cluster_info obtenido: {cluster_info}")
            
            result = {
                'h5_score': h5_score,
                'kmeans_score': kmeans_score,
                'combined_score': combined_score,
                'weighted_score': weighted_score,
                'normalized_score': normalized_score,
                'is_trending': is_trending,
                'trend_label': trend_label,
                'threshold': threshold,
                'cluster_info': cluster_info,
                'formula_used': f'Inteligente: H5({h5_weight}) + K-means({kmeans_weight}) + Sinergia + Sigmoide',
                'image_path': image_path
            }
            
            # Calcular valores para mostrar en la fórmula
            synergy_bonus = 0
            if h5_score > 0.15 and kmeans_score > 0.05:
                synergy_bonus = min(0.15, (h5_score * kmeans_score) * 2)
            
            weighted_score = (h5_score * h5_weight) + (kmeans_score * kmeans_weight) + synergy_bonus
            normalized_score = 1 / (1 + math.exp(-4 * (weighted_score - 0.3)))
            
            logger.info(f"FÓRMULA DETALLADA: H5({h5_score:.3f}×{h5_weight}) + K-means({kmeans_score:.3f}×{kmeans_weight}) + Sinergia({synergy_bonus:.3f}) = {weighted_score:.3f}")
            logger.info(f"Sigmoide({weighted_score:.3f}) = {normalized_score:.3f} → Final: {combined_score:.3f} | {trend_label} (Umbral: {threshold:.3f})")
            return result
            
        except Exception as e:
            logger.error(f" Error en análisis combinado: {e}")
            return {
                'h5_score': 0.0,
                'kmeans_score': 0.0,
                'combined_score': 0.0,
                'weighted_score': 0.0,
                'normalized_score': 0.0,
                'is_trending': False,
                'trend_label': "ERROR EN ANÁLISIS",
                'threshold': 0.3,
                'cluster_info': None,
                'formula_used': 'Inteligente: H5(0.7) + K-means(0.3) + Sinergia + Sigmoide',
                'image_path': image_path,
                'error': str(e)
            }
    
    def _get_cluster_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Obtener información detallada del cluster asignado"""
        try:
            # Preprocesar imagen
            image = self.preprocess_image(image_path)
            if image is None:
                return None
            
            # Preparar batch
            image_batch = np.expand_dims(image, axis=0)
            
            # Extraer embedding de la penúltima capa (1,280 dimensiones) según metodología real
            try:
                # Crear modelo intermedio para extraer embeddings de la penúltima capa
                from tensorflow.keras.models import Model
                intermediate_model = Model(inputs=self.h5_model.input, outputs=self.h5_model.layers[-2].output)
                embedding_1280 = intermediate_model.predict(image_batch, verbose=0)[0]
                
                # Aplicar PCA para reducir de 1,280 a 50 dimensiones (según metodología)
                if hasattr(self, 'pca_model') and self.pca_model is not None:
                    embedding = self.pca_model.transform([embedding_1280])[0].astype(np.float64)
                    logger.info(f"Embedding PCA: {embedding.shape} (1,280 -> 50 dimensiones)")
                else:
                    # Fallback: usar solo las primeras 50 dimensiones si no hay PCA
                    embedding = embedding_1280[:50].astype(np.float64)
                    logger.warning("Usando fallback sin PCA - solo primeras 50 dimensiones")
                    
            except Exception as e:
                logger.warning(f"Error extrayendo embedding de penúltima capa: {e}")
                # Fallback: usar la salida final
                prediction = self.h5_model.predict(image_batch, verbose=0)[0]
                expected_dims = self.kmeans_model.cluster_centers_.shape[1]
                if len(prediction) >= expected_dims:
                    embedding = prediction[:expected_dims].astype(np.float64)
                else:
                    embedding = np.zeros(expected_dims, dtype=np.float64)
                    embedding[:len(prediction)] = prediction.astype(np.float64)
            
            try:
                # Predecir cluster
                predicted_cluster = self.kmeans_model.predict([embedding])[0]
                
            except Exception as cluster_error:
                logger.error(f" Error en predicción de cluster (_get_cluster_info): {cluster_error}")
                # Fallback más inteligente usando distancia a centroides
                if hasattr(self.kmeans_model, 'cluster_centers_'):
                    distances = np.linalg.norm(self.kmeans_model.cluster_centers_ - embedding, axis=1)
                    predicted_cluster = int(np.argmin(distances))
                    logger.warning(f" Usando fallback por distancia en _get_cluster_info: cluster {predicted_cluster}")
                else:
                    # Último recurso
                    predicted_cluster = int(np.sum(embedding) % len(self.cluster_sizes))
                    logger.warning(f" Usando fallback aleatorio en _get_cluster_info: cluster {predicted_cluster}")
            
            # Obtener estadísticas del cluster
            cluster_size = self.cluster_sizes.get(predicted_cluster, 0)
            cluster_percentage = (cluster_size / self.total_images) * 100 if self.total_images > 0 else 0
            
            return {
                'cluster_id': int(predicted_cluster),
                'cluster_size': cluster_size,
                'cluster_percentage': round(cluster_percentage, 2),
                'total_images': self.total_images,
                'total_clusters': len(self.cluster_sizes)
            }
            
        except Exception as e:
            logger.error(f" Error obteniendo info del cluster: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información de los modelos cargados"""
        return {
            'h5_model_path': self.model_path,
            'kmeans_model_path': self.kmeans_path,
            'clustering_results_path': self.clustering_results_path,
            'total_images': self.total_images,
            'total_clusters': len(self.cluster_sizes),
            'formula': '(H5 * 0.6) + (K-means * 0.4)',
            'threshold': 0.6,
            'models_loaded': all([
                self.h5_model is not None,
                self.kmeans_model is not None,
                self.clustering_data is not None
            ])
        }


def create_trend_analyzer() -> FashionTrendAnalyzer:
    """Crear instancia del analizador de tendencias"""
    model_path = 'models/mobilenet_v2_final.h5'
    kmeans_path = 'models/kmeans_model.pkl'
    clustering_results_path = 'models/clustering_results.pkl'
    
    analyzer = FashionTrendAnalyzer(model_path, kmeans_path, clustering_results_path)
    return analyzer


# Función de conveniencia para análisis rápido
def analyze_fashion_trend(image_path: str) -> Dict[str, Any]:
    """
    Función de conveniencia para analizar una imagen
    """
    analyzer = create_trend_analyzer()
    
    if not analyzer.load_models():
        return {
            'error': 'No se pudieron cargar los modelos',
            'is_trending': False,
            'trend_label': 'ERROR'
        }
    
    return analyzer.calculate_combined_trend_score(image_path)


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = analyze_fashion_trend(image_path)
        print(f"Análisis de tendencia para: {image_path}")
        print(f"Resultado: {result['trend_label']}")
        print(f"Score combinado: {result['combined_score']:.4f}")
        print(f"Score H5: {result['h5_score']:.4f}")
        print(f"Score K-means: {result['kmeans_score']:.4f}")
    else:
        print("Uso: python trend_analyzer.py <ruta_imagen>")
