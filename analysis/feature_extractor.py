#!/usr/bin/env python3
"""
Módulo para extracción de características de imágenes de moda
Extrae vectores de características de la penúltima capa de una CNN pre-entrenada
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import pickle

logger = logging.getLogger(__name__)

class FashionFeatureExtractor:
    """
    Extractor de características para imágenes de moda usando CNNs pre-entrenadas
    """
    
    def __init__(self, model_name: str = 'resnet50', input_size: Tuple[int, int] = (224, 224)):
        """
        Inicializa el extractor de características
        
        Args:
            model_name: Nombre del modelo pre-entrenado ('resnet50', 'vgg16', 'inception_v3')
            input_size: Tamaño de entrada para las imágenes
        """
        self.model_name = model_name
        self.input_size = input_size
        self.model = None
        self.feature_extractor = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo pre-entrenado y crea el extractor de características"""
        try:
            if self.model_name == 'resnet50':
                self.model = ResNet50(weights='imagenet', include_top=True, input_shape=(*self.input_size, 3))
            elif self.model_name == 'vgg16':
                self.model = VGG16(weights='imagenet', include_top=True, input_shape=(*self.input_size, 3))
            elif self.model_name == 'inception_v3':
                self.model = InceptionV3(weights='imagenet', include_top=True, input_shape=(*self.input_size, 3))
            else:
                raise ValueError(f"Modelo no soportado: {self.model_name}")
            
            # Crear extractor de características (penúltima capa)
            # Excluimos la última capa (clasificación) para obtener características
            self.feature_extractor = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-2].output  # Penúltima capa
            )
            
            logger.info(f"Modelo {self.model_name} cargado exitosamente")
            logger.info(f"Tamaño de características: {self.feature_extractor.output_shape[1]}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo {self.model_name}: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocesa una imagen para el modelo
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Array preprocesado de la imagen
        """
        try:
            # Cargar imagen
            img = Image.open(image_path)
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar
            img = img.resize(self.input_size)
            
            # Convertir a array
            img_array = image.img_to_array(img)
            
            # Expandir dimensiones para batch
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocesar según el modelo
            img_array = preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen {image_path}: {e}")
            return None
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extrae características de una imagen
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Vector de características o None si hay error
        """
        try:
            # Preprocesar imagen
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return None
            
            # Extraer características
            features = self.feature_extractor.predict(img_array, verbose=0)
            
            # Aplanar el vector de características
            features = features.flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de {image_path}: {e}")
            return None
    
    def extract_batch_features(self, image_paths: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Extrae características de un lote de imágenes
        
        Args:
            image_paths: Lista de rutas de imágenes
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            Tupla con (matriz de características, lista de rutas válidas)
        """
        features_list = []
        valid_paths = []
        
        logger.info(f"Extrayendo características de {len(image_paths)} imágenes...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []
            
            for path in batch_paths:
                if os.path.exists(path):
                    features = self.extract_features(path)
                    if features is not None:
                        batch_features.append(features)
                        valid_paths.append(path)
                    else:
                        logger.warning(f"No se pudieron extraer características de: {path}")
                else:
                    logger.warning(f"Archivo no encontrado: {path}")
            
            if batch_features:
                features_list.extend(batch_features)
            
            # Log progreso
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Procesadas {min(i + batch_size, len(image_paths))}/{len(image_paths)} imágenes")
        
        if not features_list:
            logger.error("No se pudieron extraer características de ninguna imagen")
            return np.array([]), []
        
        features_matrix = np.array(features_list)
        logger.info(f"Características extraídas: {features_matrix.shape}")
        
        return features_matrix, valid_paths
    
    def save_features(self, features: np.ndarray, image_paths: List[str], output_path: str):
        """
        Guarda las características extraídas en un archivo
        
        Args:
            features: Matriz de características
            image_paths: Lista de rutas de imágenes
            output_path: Ruta donde guardar los datos
        """
        try:
            data = {
                'features': features,
                'image_paths': image_paths,
                'model_name': self.model_name,
                'input_size': self.input_size,
                'feature_dimension': features.shape[1] if len(features.shape) > 1 else len(features)
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Características guardadas en: {output_path}")
            
        except Exception as e:
            logger.error(f"Error guardando características: {e}")
            raise
    
    def load_features(self, input_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Carga características previamente extraídas
        
        Args:
            input_path: Ruta del archivo de características
            
        Returns:
            Tupla con (matriz de características, lista de rutas)
        """
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Características cargadas desde: {input_path}")
            logger.info(f"Dimensiones: {data['features'].shape}")
            
            return data['features'], data['image_paths']
            
        except Exception as e:
            logger.error(f"Error cargando características: {e}")
            raise

def get_image_paths_from_database(db_path: str) -> List[str]:
    """
    Obtiene las rutas de todas las imágenes de la base de datos
    
    Args:
        db_path: Ruta a la base de datos
        
    Returns:
        Lista de rutas de imágenes
    """
    try:
        from storage.database import ImageDatabase
        
        db = ImageDatabase()
        images = db.get_all_images()
        
        # Filtrar solo imágenes con calidad aceptable
        valid_images = [img for img in images if img.get('quality_score', 0) > 0.5]
        
        image_paths = [img['file_path'] for img in valid_images if os.path.exists(img['file_path'])]
        
        logger.info(f"Encontradas {len(image_paths)} imágenes válidas en la base de datos")
        
        return image_paths
        
    except Exception as e:
        logger.error(f"Error obteniendo rutas de imágenes: {e}")
        return []

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Ejemplo de uso
    extractor = FashionFeatureExtractor(model_name='resnet50')
    
    # Obtener rutas de imágenes de la base de datos
    image_paths = get_image_paths_from_database('fashion_images.db')
    
    if image_paths:
        # Extraer características
        features, valid_paths = extractor.extract_batch_features(image_paths[:100])  # Procesar primeras 100
        
        if len(features) > 0:
            # Guardar características
            output_path = "fashion_features.pkl"
            extractor.save_features(features, valid_paths, output_path)
            print(f"Características extraídas y guardadas: {features.shape}")
        else:
            print("No se pudieron extraer características")
    else:
        print("No se encontraron imágenes en la base de datos")

