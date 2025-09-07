"""
Módulo para visión computacional y extracción de características
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_mobilenetv2(checkpoint_path: Optional[str] = None,
                    freeze_layers: bool = True,
                    embedding_layer: int = -2) -> tf.keras.Model:
    """
    Carga MobileNetV2 preentrenada o con checkpoint afinado
    
    Args:
        checkpoint_path: Ruta al checkpoint afinado (opcional)
        freeze_layers: Si congelar las capas base
        embedding_layer: Índice de la capa de embedding
    
    Returns:
        Modelo TensorFlow configurado
    """
    try:
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Cargar modelo afinado
            model = tf.keras.models.load_model(checkpoint_path)
            logger.info(f"Modelo afinado cargado desde {checkpoint_path}")
        else:
            # Cargar MobileNetV2 preentrenada
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Crear modelo con capa de embedding
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1280, activation='relu', name='embedding')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            predictions = tf.keras.layers.Dense(11, activation='softmax', name='predictions')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            logger.info("MobileNetV2 preentrenada cargada")
        
        # Congelar capas si se especifica
        if freeze_layers:
            for layer in model.layers[:-2]:  # Congelar todas excepto las últimas 2
                layer.trainable = False
            logger.info("Capas base congeladas")
        
        # Crear extractor de características
        embedding_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[embedding_layer].output
        )
        
        logger.info(f"Extractor de características creado (capa {embedding_layer})")
        return embedding_model
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

def extract_embeddings(model: tf.keras.Model,
                      image_paths: List[str],
                      batch_size: int = 32,
                      target_size: Tuple[int, int] = (224, 224),
                      normalize: bool = True) -> np.ndarray:
    """
    Extrae embeddings de una lista de imágenes
    
    Args:
        model: Modelo de extracción de características
        image_paths: Lista de rutas de imágenes
        batch_size: Tamaño del lote
        target_size: Tamaño objetivo de las imágenes
        normalize: Si normalizar las imágenes
    
    Returns:
        Array numpy con embeddings (N, embedding_dim)
    """
    try:
        # Crear dataset
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
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Extraer embeddings
        embeddings = []
        for batch in dataset:
            batch_embeddings = model.predict(batch, verbose=0)
            embeddings.append(batch_embeddings)
        
        # Concatenar todos los embeddings
        embeddings = np.vstack(embeddings)
        
        logger.info(f"Extraídos {len(embeddings)} embeddings de dimensión {embeddings.shape[1]}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error extrayendo embeddings: {e}")
        raise

def extract_embeddings_batch(model: tf.keras.Model,
                            image_paths: List[str],
                            batch_size: int = 32,
                            target_size: Tuple[int, int] = (224, 224),
                            normalize: bool = True,
                            progress_callback: Optional[callable] = None) -> np.ndarray:
    """
    Extrae embeddings por lotes con callback de progreso
    
    Args:
        model: Modelo de extracción de características
        image_paths: Lista de rutas de imágenes
        batch_size: Tamaño del lote
        target_size: Tamaño objetivo de las imágenes
        normalize: Si normalizar las imágenes
        progress_callback: Función de callback para progreso
    
    Returns:
        Array numpy con embeddings
    """
    try:
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Cargar y preprocesar lote
            batch_images = []
            for path in batch_paths:
                try:
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
                    
                    batch_images.append(img)
                except Exception as e:
                    logger.warning(f"Error procesando {path}: {e}")
                    # Crear imagen vacía como fallback
                    empty_img = tf.zeros((*target_size, 3), dtype=tf.float32)
                    batch_images.append(empty_img)
            
            if batch_images:
                batch_tensor = tf.stack(batch_images)
                batch_embeddings = model.predict(batch_tensor, verbose=0)
                embeddings.append(batch_embeddings)
            
            # Callback de progreso
            if progress_callback:
                progress = (i // batch_size + 1) / total_batches
                progress_callback(progress)
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            logger.info(f"Extraídos {len(embeddings)} embeddings de dimensión {embeddings.shape[1]}")
        else:
            logger.error("No se pudieron extraer embeddings")
            embeddings = np.array([])
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error extrayendo embeddings por lotes: {e}")
        raise

def get_model_info(model: tf.keras.Model) -> dict:
    """
    Obtiene información del modelo
    
    Args:
        model: Modelo TensorFlow
    
    Returns:
        Diccionario con información del modelo
    """
    info = {
        'total_layers': len(model.layers),
        'trainable_layers': sum(1 for layer in model.layers if layer.trainable),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params()
    }
    
    # Información de capas
    layer_info = []
    for i, layer in enumerate(model.layers):
        layer_info.append({
            'index': i,
            'name': layer.name,
            'type': type(layer).__name__,
            'trainable': layer.trainable,
            'output_shape': layer.output_shape
        })
    
    info['layers'] = layer_info
    return info


