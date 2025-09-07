"""
Módulo para manejo de datos y preprocesamiento de imágenes
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import tensorflow as tf
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_dataset(data_root: str, 
                splits: Optional[List[str]] = None,
                image_extensions: List[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Carga el dataset desde la estructura de directorios
    
    Args:
        data_root: Ruta raíz del dataset
        splits: Lista de splits a cargar (train, val, test)
        image_extensions: Extensiones de imagen válidas
    
    Returns:
        Tuple con (image_paths, split_labels, category_labels)
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if splits is None:
        splits = ['train', 'val', 'test']
    
    data_root = Path(data_root)
    image_paths = []
    split_labels = []
    category_labels = []
    
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            logger.warning(f"Split {split} no encontrado en {split_dir}")
            continue
        
        # Buscar subdirectorios de categorías
        for category_dir in split_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                
                # Buscar imágenes en la categoría
                for img_file in category_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        image_paths.append(str(img_file))
                        split_labels.append(split)
                        category_labels.append(category)
        
        # Si no hay subdirectorios, buscar imágenes directamente
        if not any(d.is_dir() for d in split_dir.iterdir()):
            for img_file in split_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    image_paths.append(str(img_file))
                    split_labels.append(split)
                    category_labels.append('unknown')
    
    logger.info(f"Cargadas {len(image_paths)} imágenes de {len(set(split_labels))} splits")
    return image_paths, split_labels, category_labels

def preprocess_image(image_path: str, 
                    target_size: Tuple[int, int] = (224, 224),
                    normalize: bool = True) -> Optional[np.ndarray]:
    """
    Preprocesa una imagen para el modelo
    
    Args:
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo (alto, ancho)
        normalize: Si normalizar con estadísticas ImageNet
    
    Returns:
        Array numpy preprocesado o None si hay error
    """
    try:
        # Cargar imagen
        img = Image.open(image_path)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize(target_size)
        
        # Convertir a array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalizar
        if normalize:
            # Normalización ImageNet
            img_array = img_array / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
        else:
            img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocesando {image_path}: {e}")
        return None

def create_metadata(image_paths: List[str], 
                   split_labels: List[str],
                   category_labels: List[str],
                   timestamps: Optional[List[str]] = None,
                   sources: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Crea DataFrame de metadatos
    
    Args:
        image_paths: Lista de rutas de imágenes
        split_labels: Lista de etiquetas de split
        category_labels: Lista de etiquetas de categoría
        timestamps: Lista de timestamps (opcional)
        sources: Lista de fuentes (opcional)
    
    Returns:
        DataFrame con metadatos
    """
    metadata = {
        'image_path': image_paths,
        'split': split_labels,
        'label': category_labels
    }
    
    if timestamps is not None:
        metadata['timestamp'] = timestamps
    else:
        # Generar timestamps basados en fecha de modificación
        timestamps = []
        for img_path in image_paths:
            try:
                mtime = os.path.getmtime(img_path)
                timestamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
                timestamps.append(timestamp)
            except:
                timestamps.append('unknown')
        metadata['timestamp'] = timestamps
    
    if sources is not None:
        metadata['source'] = sources
    else:
        # Inferir fuente basada en la ruta
        sources = []
        for img_path in image_paths:
            if 'global' in img_path.lower() or 'web' in img_path.lower():
                sources.append('global')
            else:
                sources.append('local')
        metadata['source'] = sources
    
    return pd.DataFrame(metadata)

def create_data_generator(image_paths: List[str],
                         batch_size: int = 32,
                         target_size: Tuple[int, int] = (224, 224),
                         normalize: bool = True,
                         shuffle: bool = True) -> tf.data.Dataset:
    """
    Crea generador de datos TensorFlow
    
    Args:
        image_paths: Lista de rutas de imágenes
        batch_size: Tamaño del lote
        target_size: Tamaño objetivo de las imágenes
        normalize: Si normalizar las imágenes
        shuffle: Si mezclar los datos
    
    Returns:
        Dataset de TensorFlow
    """
    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32)
        
        if normalize:
            img = img / 255.0
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std
        else:
            img = img / 255.0
        
        return img
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def find_cluster_prototypes(embeddings: np.ndarray, cluster_labels: np.ndarray, 
                           image_paths: List[str], top_k: int = 10) -> Dict[int, List[str]]:
    """Encuentra los prototipos (imágenes más cercanas al centroide) de cada cluster"""
    prototypes = {}
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Saltar ruido en DBSCAN
            continue
            
        # Obtener embeddings del cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_paths = [image_paths[i] for i in range(len(image_paths)) if cluster_mask[i]]
        
        if len(cluster_embeddings) == 0:
            continue
            
        # Calcular centroide
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calcular distancias al centroide
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Obtener los top_k más cercanos
        top_indices = np.argsort(distances)[:min(top_k, len(distances))]
        prototypes[cluster_id] = [cluster_paths[i] for i in top_indices]
    
    return prototypes
