#!/usr/bin/env python3
"""
Sistema de Análisis de Tendencias de Moda con IA - VERSIÓN SIMPLIFICADA
Usa solo los archivos ya entrenados: MobileNetV2 + PCA + K-means
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import threading
import time
import colorsys
from sklearn.cluster import KMeans
import warnings

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Cache para optimizar rendimiento
_color_cache = {}
_model_cache = {}
_MAX_CACHE_SIZE = 100

def clear_cache():
    """Limpiar cache para liberar memoria"""
    global _color_cache, _model_cache
    _color_cache.clear()
    _model_cache.clear()
    logger.info("Cache limpiado")

def manage_cache_size():
    """Gestionar tamaño del cache para evitar uso excesivo de memoria"""
    global _color_cache, _model_cache
    
    if len(_color_cache) > _MAX_CACHE_SIZE:
        keys_to_remove = list(_color_cache.keys())[:len(_color_cache) - _MAX_CACHE_SIZE]
        for key in keys_to_remove:
            del _color_cache[key]
    
    if len(_model_cache) > _MAX_CACHE_SIZE:
        keys_to_remove = list(_model_cache.keys())[:len(_model_cache) - _MAX_CACHE_SIZE]
        for key in keys_to_remove:
            del _model_cache[key]

# Importar analizador de tendencias
from utils.trend_analyzer import FashionTrendAnalyzer, create_trend_analyzer

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesar imagen para el modelo - versión básica"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error preprocesando {image_path}: {e}")
        return None

def preprocess_image_enhanced(image_path, target_size=(224, 224)):
    """Preprocesar imagen mejorado para el modelo H5"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Mejoras de calidad
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.05)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.02)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.03)
        
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalización para MobileNetV2
        img_array = (img_array / 127.5) - 1.0
        img_array = np.clip(img_array, -1.0, 1.0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error en preprocesamiento mejorado {image_path}: {e}")
        return preprocess_image(image_path, target_size)

# Configuración
app = Flask(__name__, static_folder='/home/jose/static', static_url_path='/static')
app.config['SECRET_KEY'] = 'fashion_trend_2025_guatemala'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/home/jose/static/images/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para modelos
model = None
clustering_data = None
cluster_centers = None
cluster_stats = None
trend_analyzer = None
models_loaded = False
loading_progress = 0
loading_status = "Iniciando..."

# Variables globales para progreso de análisis
analysis_progress = 0
analysis_status = "Iniciando..."
analysis_active = False
analysis_result = None
analysis_filename = None

def load_models_with_progress():
    """Cargar modelos con indicador de progreso"""
    global model, clustering_data, cluster_centers, cluster_stats, trend_analyzer, models_loaded, loading_progress, loading_status
    
    try:
        loading_progress = 0
        loading_status = "Iniciando carga de modelos..."
        logger.info("Iniciando carga de modelos...")
        
        # Paso 1: Cargar modelo MobileNetV2 entrenado (20%)
        loading_progress = 10
        loading_status = "Cargando modelo MobileNetV2..."
        logger.info("Cargando modelo MobileNetV2...")
        
        model_path = 'models/mobilenet_v2_final.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Modelo cargado desde {model_path}")
        else:
            loading_status = "Cargando MobileNetV2 preentrenado..."
            model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet'
            )
            logger.info("Usando MobileNetV2 preentrenado")
        
        loading_progress = 30
        loading_status = "Cargando datos de clustering..."
        
        # Paso 2: Cargar datos de clustering (30%)
        logger.info("Cargando datos de clustering...")
        clustering_data = load_clustering_data()
        
        loading_progress = 70
        loading_status = "Calculando centros de clusters..."
        
        # Paso 3: Calcular centros de clusters (20%)
        logger.info("Calculando centros de clusters...")
        cluster_centers = calculate_cluster_centers()
        
        loading_progress = 90
        loading_status = "Cargando estadísticas..."
        
        # Paso 4: Cargar estadísticas (10%)
        logger.info("Cargando estadísticas...")
        cluster_stats = load_cluster_stats()
        
        loading_progress = 95
        loading_status = "Inicializando analizador de tendencias..."
        
        # Paso 5: Inicializar analizador de tendencias (5%)
        logger.info("Inicializando analizador de tendencias...")
        trend_analyzer = create_trend_analyzer()
        if not trend_analyzer.load_models():
            raise Exception("No se pudo cargar el analizador de tendencias")
        logger.info("Analizador de tendencias inicializado")
        
        loading_progress = 100
        loading_status = "¡Modelos cargados exitosamente!"
        models_loaded = True
        
        logger.info("Todos los modelos cargados exitosamente")
        
    except Exception as e:
        loading_status = f"Error: {str(e)}"
        logger.error(f"Error cargando modelos: {e}")
        models_loaded = False

def load_clustering_data():
    """Cargar datos de clustering desde fashion_trend_app/models/"""
    try:
        # Cargar resultados completos del clustering
        with open('models/clustering_results.pkl', 'rb') as f:
            clustering_results = pickle.load(f)
        
        # Cargar modelo K-means
        with open('models/kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        
        # Cargar modelo PCA
        pca_model = None
        try:
            with open('models/pca_model.pkl', 'rb') as f:
                pca_model = pickle.load(f)
            logger.info("Modelo PCA cargado exitosamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo PCA: {e}")
        
        # Corregir tipo de datos si es necesario
        if kmeans_model.cluster_centers_.dtype != np.float64:
            kmeans_model.cluster_centers_ = kmeans_model.cluster_centers_.astype(np.float64)
        
        return {
            'cluster_labels': clustering_results.get('cluster_labels', None),
            'image_paths': clustering_results.get('image_paths', None),
            'categories': clustering_results.get('categories', None),
            'metrics': clustering_results.get('metrics', {}),
            'cluster_counts': clustering_results.get('cluster_counts', {}),
            'category_cluster_analysis': clustering_results.get('category_cluster_analysis', {}),
            'kmeans_model': kmeans_model,
            'pca_model': pca_model,
            'n_clusters': clustering_results.get('n_clusters', 150),
            'total_images': clustering_results.get('total_images', 16959)
        }
    except Exception as e:
        logger.error(f"Error cargando datos de clustering: {e}")
        return None

def calculate_cluster_centers():
    """Calcular centros de clusters usando el modelo K-means"""
    if clustering_data is None:
        return None
    
    try:
        kmeans_model = clustering_data['kmeans_model']
        cluster_centers = {}
        
        for i, center in enumerate(kmeans_model.cluster_centers_):
            cluster_centers[i] = center
        
        return cluster_centers
    except Exception as e:
        logger.error(f"Error calculando centros de clusters: {e}")
        return None

def load_cluster_stats():
    """Cargar estadísticas de clusters desde los datos reales de clustering"""
    try:
        clustering_results_path = 'models/clustering_results.pkl'
        if os.path.exists(clustering_results_path):
            with open(clustering_results_path, 'rb') as f:
                clustering_data = pickle.load(f)
            
            cluster_sizes = clustering_data.get('cluster_sizes', {})
            total_images = clustering_data.get('total_images', 16959)
            total_clusters = clustering_data.get('total_clusters', 150)
            
            stats = {
                'cluster_sizes': cluster_sizes,
                'algorithm': 'kmeans',
                'total_clusters': total_clusters,
                'total_images': total_images,
                'silhouette_score': clustering_data.get('silhouette_score', 0.0),
                'calinski_harabasz_score': clustering_data.get('calinski_harabasz_score', 0.0),
                'davies_bouldin_score': clustering_data.get('davies_bouldin_score', 0.0)
            }
            
            logger.info(f"Estadísticas reales cargadas: {total_clusters} clusters, {total_images} imágenes")
            return stats
        else:
            stats = {
                'cluster_sizes': {},
                'algorithm': 'kmeans',
                'total_clusters': 150,
                'total_images': 16959
            }
            logger.warning("Usando estadísticas básicas (archivo no encontrado)")
            return stats
        
    except Exception as e:
        logger.error(f"Error cargando estadísticas: {e}")
        return None

def predict_image(image_path):
    """Realizar predicción en una imagen con modelo H5"""
    if not models_loaded:
        return None, None, None, "Modelos aún cargando..."
    
    # Verificar cache primero
    cache_key = f"pred_{os.path.getmtime(image_path)}_{os.path.getsize(image_path)}"
    if cache_key in _model_cache:
        logger.info("Usando cache para predicción del modelo")
        return _model_cache[cache_key]
    
    try:
        # Preprocesar imagen con mejoras
        image = preprocess_image_enhanced(image_path, target_size=(224, 224))
        if image is None:
            return None, None, None, "Error en preprocesamiento de imagen"
            
        image_batch = np.expand_dims(image, axis=0)
        
        # Usar el modelo completo para predicción de clase
        prediction = model.predict(image_batch, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Extraer embedding de la penúltima capa (256 dimensiones) según metodología real
        try:
            intermediate_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            embedding_256 = intermediate_model.predict(image_batch, verbose=0)[0]
            
            # Aplicar PCA para reducir de 256 a 50 dimensiones (según metodología)
            if 'pca_model' in clustering_data and clustering_data['pca_model'] is not None:
                pca_model = clustering_data['pca_model']
                embedding = pca_model.transform([embedding_256])[0].astype(np.float64)
                logger.info(f"Embedding PCA: {embedding.shape} (256 -> 50 dimensiones)")
            else:
                embedding = embedding_256[:50].astype(np.float64)
                logger.warning("Usando fallback sin PCA - solo primeras 50 dimensiones")
                
        except Exception as e:
            logger.warning(f"Error extrayendo embedding de penúltima capa: {e}")
            if len(prediction) >= 50:
                embedding = prediction[:50].astype(np.float64)
            else:
                embedding = np.zeros(50, dtype=np.float64)
                embedding[:len(prediction)] = prediction.astype(np.float64)
        
        # Usar el modelo K-means para predecir el cluster
        predicted_cluster = None
        cluster_confidence = 0.0
        
        if clustering_data and 'kmeans_model' in clustering_data:
            kmeans_model = clustering_data['kmeans_model']
            try:
                predicted_cluster = kmeans_model.predict([embedding])[0]
                
                # Calcular confianza del cluster
                cluster_center = kmeans_model.cluster_centers_[predicted_cluster]
                distance = np.linalg.norm(embedding - cluster_center)
                cluster_confidence = max(0, 1 - distance / 10)
            except Exception as e:
                logger.warning(f"Error en predicción de cluster: {e}")
                predicted_cluster = 0
                cluster_confidence = 0.1
        
        logger.info(f"Predicción: clase={predicted_class}, confianza={confidence:.3f}, cluster={predicted_cluster}, cluster_conf={cluster_confidence:.3f}")
        
        result = (embedding, predicted_class, confidence, None)
        
        # Guardar en cache y gestionar tamaño
        _model_cache[cache_key] = result
        manage_cache_size()
        
        return result
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return None, None, None, str(e)

def analyze_image_colors(image_path):
    """Analizar colores dominantes de la imagen con algoritmo MEJORADO y más preciso"""
    start_time = time.time()
    
    # Verificar cache primero
    cache_key = f"{os.path.getmtime(image_path)}_{os.path.getsize(image_path)}"
    if cache_key in _color_cache:
        logger.info("Usando cache para análisis de colores")
        return _color_cache[cache_key]
    
    try:
        # Cargar imagen con mejor resolución para mayor precisión
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        # Usar resolución más alta para mejor detección de colores
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Convertir a array numpy
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # PASO 1: Filtrar píxeles extremos y ruido de manera más eficiente
        # Convertir a float para evitar overflow
        pixels_float = pixels.astype(np.float32)
        
        # Calcular brightness de manera vectorizada
        brightness = np.mean(pixels_float, axis=1)
        
        # Filtrar píxeles válidos (no extremadamente oscuros o claros)
        valid_mask = (brightness > 15) & (brightness < 240)
        filtered_pixels = pixels_float[valid_mask]
        
        # Si hay muy pocos píxeles válidos, usar todos
        if len(filtered_pixels) < 500:  # Aumentado para mejor calidad
            filtered_pixels = pixels_float
        
        # Muestreo inteligente para mejorar velocidad sin perder calidad
        if len(filtered_pixels) > 10000:  # Si hay demasiados píxeles
            # Muestreo estratificado: tomar píxeles representativos
            step = len(filtered_pixels) // 8000
            filtered_pixels = filtered_pixels[::step]
        
        # PASO 2: Usar clustering optimizado con más clusters para mejor precisión
        n_clusters = min(8, max(3, len(filtered_pixels) // 1000))  # Dinámico según la imagen
        
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,  # Más inicializaciones para mejor resultado
            max_iter=100  # Más iteraciones para convergencia
        )
        labels = kmeans.fit_predict(filtered_pixels)
        
        # PASO 3: Analizar clusters y obtener colores más precisos
        color_centers = kmeans.cluster_centers_
        cluster_sizes = np.bincount(labels)
        
        # Ordenar por tamaño pero también considerar la importancia del color
        cluster_info = []
        for idx, (center, size) in enumerate(zip(color_centers, cluster_sizes)):
            r, g, b = center.astype(int)
            
            # Calcular importancia del color (no solo tamaño)
            brightness = (r + g + b) / 3
            saturation = calculate_saturation(r, g, b)
            
            # Score combinado: tamaño + importancia visual
            importance_score = size * (1 + saturation * 0.5)
            
            cluster_info.append({
                'center': (r, g, b),
                'size': size,
                'importance': importance_score,
                'brightness': brightness,
                'saturation': saturation
            })
        
        # Ordenar por importancia (no solo tamaño)
        cluster_info.sort(key=lambda x: x['importance'], reverse=True)
        
        # PASO 4: Convertir a nombres de colores con análisis mejorado
        dominant_colors = []
        for cluster in cluster_info:
            r, g, b = cluster['center']
            
            # Usar análisis híbrido RGB + HSV para mayor precisión
            color_name = analyze_color_hybrid(r, g, b)
            
            # Evitar duplicados y colores muy similares
            if color_name and color_name not in dominant_colors:
                # Verificar que no sea muy similar a colores ya detectados
                if not is_similar_color_name(color_name, dominant_colors):
                    dominant_colors.append(color_name)
            
            # Limitar a 3-4 colores principales
            if len(dominant_colors) >= 4:
                break
        
        # PASO 5: Validación y fallback
        if not dominant_colors:
            logger.info("Usando análisis directo como fallback")
            # Análisis directo del color promedio
            avg_color = np.mean(filtered_pixels, axis=0).astype(int)
            r, g, b = avg_color
            main_color = analyze_color_hybrid(r, g, b)
            dominant_colors = [main_color] if main_color else ['multicolor']
        
        # Limitar a máximo 3 colores para consistencia
        dominant_colors = dominant_colors[:3]
        
        # Guardar en cache
        manage_cache_size()
        _color_cache[cache_key] = dominant_colors
        
        logger.info(f"Colores detectados MEJORADOS: {dominant_colors}")
        logger.info(f"Tiempo de análisis: {time.time() - start_time:.2f}s")
        
        return dominant_colors
        
    except Exception as e:
        logger.error(f"Error en análisis de colores mejorado: {e}")
        # Fallback robusto
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            data = np.array(image)
            avg_color = np.mean(data.reshape(-1, 3), axis=0).astype(int)
            r, g, b = avg_color
            basic_color = analyze_color_hybrid(r, g, b)
            return [basic_color] if basic_color else ['multicolor']
        except:
            return ['multicolor']

def rgb_to_color_name(r, g, b):
    """Convertir valores RGB a nombres de colores usando sistema de puntuación flexible"""
    # Paleta de colores de referencia con rangos amplios
    color_palette = {
        # Blancos y neutros
        'blanco': {'rgb': (255, 255, 255), 'range': (200, 255), 'min_score': 0.6},
        'gris claro': {'rgb': (200, 200, 200), 'range': (180, 220), 'min_score': 0.7},
        'gris': {'rgb': (128, 128, 128), 'range': (100, 160), 'min_score': 0.7},
        'gris oscuro': {'rgb': (64, 64, 64), 'range': (40, 100), 'min_score': 0.7},
        'negro': {'rgb': (0, 0, 0), 'range': (0, 40), 'min_score': 0.6},
        
        # Rojos - MEJORADOS para detectar rojos vibrantes (PRIORIDAD ALTA)
        'rojo brillante': {'rgb': (255, 0, 0), 'range': (80, 255), 'min_score': 0.2},
        'rojo': {'rgb': (200, 50, 50), 'range': (80, 220), 'min_score': 0.2},
        'rojo vibrante': {'rgb': (220, 20, 20), 'range': (80, 240), 'min_score': 0.2},
        'rojo oscuro': {'rgb': (120, 20, 20), 'range': (60, 140), 'min_score': 0.2},
        'rojo-naranja': {'rgb': (220, 80, 20), 'range': (100, 240), 'min_score': 0.2},
        
        # Naranjas
        'naranja brillante': {'rgb': (255, 165, 0), 'range': (200, 255), 'min_score': 0.6},
        'naranja': {'rgb': (200, 100, 0), 'range': (150, 200), 'min_score': 0.6},
        'naranja oscuro': {'rgb': (150, 75, 0), 'range': (100, 150), 'min_score': 0.6},
        
        # Cafés y Marrones - NUEVOS (AJUSTADOS para no competir con rojos)
        'café claro': {'rgb': (210, 180, 140), 'range': (160, 220), 'min_score': 0.5},
        'café': {'rgb': (160, 82, 45), 'range': (100, 180), 'min_score': 0.5},
        'café oscuro': {'rgb': (101, 67, 33), 'range': (60, 120), 'min_score': 0.5},
        'marrón claro': {'rgb': (205, 133, 63), 'range': (130, 210), 'min_score': 0.5},
        'marrón': {'rgb': (139, 69, 19), 'range': (80, 150), 'min_score': 0.5},
        'marrón oscuro': {'rgb': (92, 51, 23), 'range': (50, 100), 'min_score': 0.5},
        'café tostado': {'rgb': (139, 90, 43), 'range': (100, 160), 'min_score': 0.5},
        'café chocolate': {'rgb': (210, 105, 30), 'range': (150, 220), 'min_score': 0.5},
        'café caramelo': {'rgb': (160, 120, 80), 'range': (120, 180), 'min_score': 0.5},
        'café canela': {'rgb': (210, 180, 140), 'range': (160, 220), 'min_score': 0.5},
        
        # Amarillos
        'amarillo brillante': {'rgb': (255, 255, 0), 'range': (200, 255), 'min_score': 0.6},
        'amarillo': {'rgb': (200, 200, 0), 'range': (150, 200), 'min_score': 0.6},
        'amarillo oscuro': {'rgb': (150, 150, 0), 'range': (100, 150), 'min_score': 0.6},
        
        # Verdes
        'verde brillante': {'rgb': (0, 255, 0), 'range': (0, 255), 'min_score': 0.6},
        'verde': {'rgb': (0, 180, 0), 'range': (0, 200), 'min_score': 0.6},
        'verde oscuro': {'rgb': (0, 100, 0), 'range': (0, 120), 'min_score': 0.6},
        
        # Azules - MEJORADOS para detectar lona azul y otros azules
        'azul marino': {'rgb': (25, 25, 112), 'range': (20, 120), 'min_score': 0.3},
        'azul oscuro': {'rgb': (0, 0, 139), 'range': (30, 150), 'min_score': 0.3},
        'azul': {'rgb': (0, 0, 255), 'range': (80, 200), 'min_score': 0.3},
        'azul real': {'rgb': (65, 105, 225), 'range': (60, 225), 'min_score': 0.3},
        'azul brillante': {'rgb': (0, 100, 255), 'range': (100, 255), 'min_score': 0.3},
        'azul claro': {'rgb': (135, 206, 235), 'range': (130, 235), 'min_score': 0.3},
        'azul cielo': {'rgb': (173, 216, 230), 'range': (150, 230), 'min_score': 0.3},
        'azul acero': {'rgb': (70, 130, 180), 'range': (60, 180), 'min_score': 0.3},
        'azul denim': {'rgb': (21, 96, 189), 'range': (50, 190), 'min_score': 0.3},
        'azul grisáceo': {'rgb': (100, 149, 237), 'range': (90, 200), 'min_score': 0.3},
        
        # Morados
        'morado brillante': {'rgb': (128, 0, 128), 'range': (100, 150), 'min_score': 0.6},
        'morado': {'rgb': (100, 0, 100), 'range': (80, 120), 'min_score': 0.6},
        'morado oscuro': {'rgb': (60, 0, 60), 'range': (40, 80), 'min_score': 0.6},
        
        # Rosas y Magentas
        'rosa brillante': {'rgb': (255, 192, 203), 'range': (200, 255), 'min_score': 0.6},
        'rosa': {'rgb': (200, 150, 150), 'range': (150, 200), 'min_score': 0.6},
        'magenta': {'rgb': (255, 0, 255), 'range': (200, 255), 'min_score': 0.6},
        
        # Colores especiales
        'cian': {'rgb': (0, 255, 255), 'range': (0, 255), 'min_score': 0.6},
        'azul-verde': {'rgb': (0, 128, 128), 'range': (0, 150), 'min_score': 0.6},
    }
    
    # Calcular puntuaciones para todos los colores
    color_scores = {}
    for color_name, color_info in color_palette.items():
        target_r, target_g, target_b = color_info['rgb']
        score = calculate_color_score(r, g, b, target_r, target_g, target_b)
        
        # Verificar si está en el rango de brillo
        brightness = (r + g + b) / 3
        min_brightness, max_brightness = color_info['range']
        
        if min_brightness <= brightness <= max_brightness and score >= color_info['min_score']:
            color_scores[color_name] = score
    
    # Si no hay coincidencias buenas, usar análisis HSV como fallback
    if not color_scores:
        return analyze_color_hsv_fallback(r, g, b)
    
    # Devolver el color con mayor puntuación
    best_color = max(color_scores, key=color_scores.get)
    return best_color

def calculate_saturation(r, g, b):
    """Calcular saturación de un color RGB (0-1)"""
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    
    if max_val == 0:
        return 0
    return (max_val - min_val) / max_val

def analyze_color_hybrid(r, g, b):
    """Análisis híbrido RGB + HSV ULTRA-PRECISO para detección de colores reales"""
    # Normalizar valores RGB
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h_deg = h * 360
    
    # ANÁLISIS ULTRA-PRECISO CON MÚLTIPLES CRITERIOS
    
    # 1. NEGROS Y BLANCOS PUROS (casos especiales)
    if v < 0.15:  # Muy oscuro
        return 'negro'
    elif v > 0.92 and s < 0.05:  # Muy claro y sin saturación
        return 'blanco'
    
    # 2. GRISES (baja saturación, valores medios)
    elif s < 0.12:
        if v < 0.25:
            return 'gris muy oscuro'
        elif v < 0.45:
            return 'gris oscuro'
        elif v < 0.70:
            return 'gris'
        elif v < 0.88:
            return 'gris claro'
        else:
            return 'blanco'
    
    # 3. ROJOS MEJORADOS (0-25° y 340-360°) - MÁS PRECISO
    elif (h_deg <= 25 or h_deg >= 340):
        if s > 0.8 and v > 0.8:
            return 'rojo brillante'
        elif s > 0.6 and v > 0.6:
            return 'rojo'
        elif s > 0.4 and v > 0.3:
            return 'rojo oscuro'
        else:
            # Rojo con poca saturación = marrón rojizo
            return 'café rojizo' if v > 0.4 else 'café oscuro'
    
    # 4. NARANJAS Y MARRONES (25-65°) - SEPARACIÓN CLARA
    elif 25 < h_deg <= 65:
        if s > 0.7 and v > 0.7:
            return 'naranja brillante'
        elif s > 0.5 and v > 0.5:
            return 'naranja'
        elif s > 0.3:
            return 'naranja oscuro'
        else:
            # Marrones (naranja desaturado)
            if v > 0.7:
                return 'beige'
            elif v > 0.5:
                return 'café claro'
            elif v > 0.3:
                return 'café'
            else:
                return 'café oscuro'
    
    # 5. AMARILLOS (65-95°) - MEJORADO
    elif 65 < h_deg <= 95:
        if s > 0.7 and v > 0.8:
            return 'amarillo brillante'
        elif s > 0.5 and v > 0.6:
            return 'amarillo'
        elif s > 0.3:
            return 'amarillo oscuro'
        else:
            return 'beige amarillento'
    
    # 6. VERDES (95-165°) - PRECISIÓN MEJORADA
    elif 95 < h_deg <= 165:
        if s > 0.7 and v > 0.7:
            return 'verde brillante'
        elif s > 0.5 and v > 0.5:
            if h_deg < 130:
                return 'verde lima'
            else:
                return 'verde'
        elif s > 0.3:
            return 'verde oscuro'
        else:
            return 'verde grisáceo'
    
    # 7. CIANES (165-185°) - MÁS ESPECÍFICO
    elif 165 < h_deg <= 185:
        if s > 0.6:
            return 'cian'
        else:
            return 'azul verdoso'
    
    # 8. AZULES (185-265°) - ULTRA-PRECISIÓN PARA DENIM
    elif 185 < h_deg <= 265:
        if s > 0.8 and v > 0.8:
            return 'azul brillante'
        elif s > 0.6 and v > 0.7:
            if h_deg < 210:
                return 'azul cielo'
            elif h_deg < 230:
                return 'azul'
            else:
                return 'azul real'
        elif s > 0.4 and v > 0.4:
            if h_deg < 215:
                return 'azul denim'  # PERFECTO PARA JEANS
            elif h_deg < 240:
                return 'azul oscuro'
            else:
                return 'azul marino'
        elif v > 0.3:
            return 'azul marino'
        else:
            return 'azul muy oscuro'
    
    # 9. MORADOS Y VIOLETAS (265-340°) - SEPARACIÓN CLARA
    elif 265 < h_deg < 340:
        if h_deg < 295:  # Morados
            if s > 0.6 and v > 0.6:
                return 'morado'
            elif s > 0.4:
                return 'morado oscuro'
            else:
                return 'gris violáceo'
        else:  # Magentas y rosas
            if s > 0.7 and v > 0.7:
                return 'magenta'
            elif s > 0.5 and v > 0.6:
                return 'rosa'
            elif s > 0.3:
                return 'rosa oscuro'
            else:
                return 'rosa pálido'
    
    # Fallback robusto
    return 'color indefinido'

def is_similar_color_name(color_name, existing_colors):
    """Verificar si un color es muy similar a los ya detectados"""
    if not existing_colors:
        return False
    
    # Grupos de colores similares ULTRA-PRECISOS
    similar_groups = [
        # Rojos
        ['rojo', 'rojo brillante', 'rojo oscuro', 'café rojizo'],
        # Azules (separados por tipo)
        ['azul', 'azul brillante', 'azul real', 'azul cielo'],
        ['azul denim', 'azul oscuro', 'azul marino', 'azul muy oscuro'],
        ['azul verdoso', 'cian'],
        # Verdes
        ['verde', 'verde brillante', 'verde lima', 'verde oscuro', 'verde grisáceo'],
        # Amarillos
        ['amarillo', 'amarillo brillante', 'amarillo oscuro', 'beige amarillento'],
        # Naranjas
        ['naranja', 'naranja brillante', 'naranja oscuro'],
        # Marrones y cafés
        ['café', 'café claro', 'café oscuro', 'café rojizo', 'beige'],
        # Grises (más específicos)
        ['gris', 'gris claro', 'gris oscuro', 'gris muy oscuro', 'gris violáceo'],
        # Morados
        ['morado', 'morado oscuro', 'gris violáceo'],
        # Rosas y magentas
        ['rosa', 'rosa oscuro', 'rosa pálido', 'magenta'],
        # Neutros
        ['negro', 'blanco']
    ]
    
    # Verificar si el color pertenece a un grupo ya representado
    for group in similar_groups:
        if color_name in group:
            for existing in existing_colors:
                if existing in group:
                    return True
    
    return False

def calculate_color_score(r, g, b, target_r, target_g, target_b, max_distance=441):
    """Calcular puntuación de similitud de color (0-1)"""
    distance = calculate_color_distance(r, g, b, target_r, target_g, target_b)
    score = max(0, 1 - distance / max_distance)
    return score

def calculate_color_distance(r1, g1, b1, r2, g2, b2):
    """Calcular distancia euclidiana entre dos colores RGB"""
    return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)

def analyze_color_hsv_fallback(r, g, b):
    """Análisis HSV como fallback cuando el sistema de puntuación no encuentra coincidencias"""
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h_deg = h * 360
    
    # Análisis básico HSV
    if v < 0.15:
        return 'negro'
    elif v > 0.8 and s < 0.15:
        return 'blanco'
    elif s < 0.2:
        if v < 0.3:
            return 'gris muy oscuro'
        elif v < 0.5:
            return 'gris oscuro'
        elif v < 0.7:
            return 'gris'
        else:
            return 'gris claro'
    else:
        # Colores saturados básicos - MEJORADO para rojos
        if (h_deg < 25 or h_deg > 335) and s > 0.2:
            if v > 0.7:
                return 'rojo brillante'
            elif v > 0.4:
                return 'rojo'
            else:
                return 'rojo oscuro'
        elif 15 <= h_deg < 45:
            return 'naranja'
        elif 45 <= h_deg < 75:
            return 'amarillo'
        elif 75 <= h_deg < 165:
            return 'verde'
        elif 165 <= h_deg < 195:
            return 'cian'
        elif 195 <= h_deg < 255:
            if v > 0.7:
                return 'azul brillante'
            elif v > 0.4:
                return 'azul'
            else:
                return 'azul oscuro'
        elif 255 <= h_deg < 285:
            return 'morado'
        elif 285 <= h_deg < 315:
            return 'magenta'
        else:
            # Detección de cafés y marrones
            if 15 <= h_deg < 60 and s < 0.6 and v < 0.8:
                if v < 0.3:
                    return 'café oscuro'
                elif v < 0.5:
                    return 'café'
                else:
                    return 'café claro'
            else:
                return 'rosa'

# Rutas de la aplicación
@app.route('/')
def index():
    """Página principal"""
    if not models_loaded:
        return render_template('index.html', 
                             models_loaded=False,
                             loading_progress=loading_progress,
                             loading_status=loading_status)
    else:
        return render_template('index.html', 
                             models_loaded=True,
                             loading_progress=100,
                             loading_status="¡Modelos cargados exitosamente!")

@app.route('/api/loading_status')
def api_loading_status():
    """API para obtener el estado de carga"""
    return jsonify({
        'models_loaded': models_loaded,
        'progress': loading_progress,
        'status': loading_status
    })

@app.route('/api/analysis_progress')
def api_analysis_progress():
    """API para obtener el progreso del análisis"""
    return jsonify({
        'analysis_active': analysis_active,
        'progress': analysis_progress,
        'status': analysis_status
    })

@app.route('/api/analysis_result')
def api_analysis_result():
    """API para obtener el resultado del análisis"""
    try:
        logger.info(f"DEBUG: analysis_result disponible: {analysis_result is not None}")
        if analysis_result is not None:
            # Convertir numpy arrays a listas para serialización JSON
            serializable_result = {}
            for key, value in analysis_result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = np.array(value).tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable_result[key] = bool(value)
                elif isinstance(value, dict):
                    # Procesar diccionarios anidados (como cluster_info)
                    nested_dict = {}
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, np.integer):
                            nested_dict[nested_key] = int(nested_value)
                        elif isinstance(nested_value, np.floating):
                            nested_dict[nested_key] = float(nested_value)
                        elif isinstance(nested_value, np.bool_):
                            nested_dict[nested_key] = bool(nested_value)
                        else:
                            nested_dict[nested_key] = nested_value
                    serializable_result[key] = nested_dict
                else:
                    serializable_result[key] = value
            
            logger.info(f"DEBUG: Devolviendo resultado exitoso con {len(serializable_result)} campos")
            logger.info(f"DEBUG: cluster_info serializado: {serializable_result.get('cluster_info', 'NO EXISTE')}")
            return jsonify({
                'success': True,
                'result': serializable_result,
                'filename': analysis_filename
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Análisis no completado o no disponible'
            })
    except Exception as e:
        logger.error(f"Error en api_analysis_result: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error procesando resultado del análisis'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API para análisis con progreso en tiempo real"""
    if not models_loaded:
        return jsonify({'error': 'Los modelos aún se están cargando'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400
    
    try:
        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Usar análisis integral completo
        result = generate_comprehensive_analysis(filepath)
        
        # Calcular tiempo de análisis
        start_time = time.time()
        analysis_time = time.time() - start_time
        result['total_analysis_time'] = round(analysis_time, 2)
        
        # Generar interpretación adicional
        interpretation = generate_heuristic_interpretation(result)
        result['interpretation'] = interpretation
        
        # GUARDAR RESULTADO GLOBALMENTE para /api/analysis_result
        global analysis_result, analysis_active, analysis_progress, analysis_status
        analysis_result = result
        analysis_active = False
        analysis_progress = 100
        analysis_status = "Análisis completado"
        
        return jsonify({
            'success': True,
            'message': 'Análisis integral completado',
            'filename': filename,
            'result': result
        })
        
    except Exception as e:
            logger.error(f"Error en análisis integral: {e}")
            return jsonify({
                'success': False,
                'error': f'Error en análisis: {str(e)}',
                'filename': filename
            }), 500
        
    except Exception as e:
        logger.error(f"Error en API analyze: {e}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Verificar si el archivo está permitido"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_comprehensive_analysis(image_path):
    """Análisis integral completo que combina todos los componentes"""
    try:
        logger.info("Iniciando análisis integral completo...")
        
        # 1. ANÁLISIS DE COLORES
        logger.info("Paso 1: Analizando colores...")
        colors = analyze_image_colors(image_path)
        
        # 2. PREDICCIÓN DEL MODELO H5
        logger.info("Paso 2: Predicción del modelo H5...")
        embedding, predicted_class, confidence, error = predict_image(image_path)
        
        # 3. ANÁLISIS DE TENDENCIAS
        logger.info("Paso 3: Análisis de tendencias...")
        if trend_analyzer is not None:
            trend_analysis = trend_analyzer.calculate_combined_trend_score(image_path)
            logger.info(f"DEBUG: trend_analysis recibido: {trend_analysis}")
        else:
            # Fallback si no hay trend_analyzer
            h5_score = confidence if confidence else 0.0
            trend_analysis = {
                'h5_score': h5_score,
                'kmeans_score': 0.0,
                'combined_score': h5_score,
                'is_trending': h5_score > 0.3,
                'trend_label': "EN TENDENCIA" if h5_score > 0.3 else "NO EN TENDENCIA",
                'cluster_info': {'cluster_id': 0, 'cluster_size': 0},
                'formula_used': 'H5_only_fallback',
                'threshold': 0.3
            }
        
        # 4. MAPEO DE CATEGORÍAS
        category_names = {
            0: 'Casual', 1: 'Formal', 2: 'Sporty', 3: 'Elegant',
            4: 'Vintage', 5: 'Modern', 6: 'Bohemian', 7: 'Minimalist'
        }
        predicted_category = category_names.get(predicted_class, f'Categoría {predicted_class}') if predicted_class is not None else 'Desconocida'
        
        # 5. RESULTADO FINAL UNIFICADO
        logger.info(f"DEBUG: cluster_info del trend_analysis: {trend_analysis.get('cluster_info', 'NO EXISTE')}")
        result = {
            # Información básica
            'image_path': image_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_method': 'comprehensive_integral',
            
            # Colores
            'colors': colors,
            'primary_color': colors[0] if colors else 'Desconocido',
            'color_palette': colors,
            
            # Categoría y estilo
            'predicted_class': int(predicted_class) if predicted_class is not None else -1,
            'predicted_category': predicted_category,
            'style_confidence': float(confidence) if confidence else 0.0,
            'confidence': float(confidence) if confidence else 0.0,  # Para compatibilidad con frontend
            
            # Tendencias
            'trend_score': int(trend_analysis['combined_score'] * 100),
                'is_trending': bool(trend_analysis['is_trending']),
                'trend_label': str(trend_analysis['trend_label']),
                'h5_score': float(trend_analysis['h5_score']),
                'kmeans_score': float(trend_analysis['kmeans_score']),
                'combined_score': float(trend_analysis['combined_score']),
            
            # Clustering
            'cluster_id': int(trend_analysis['cluster_info'].get('cluster_id', 0)),
            'cluster_size': int(trend_analysis['cluster_info'].get('cluster_size', 0)),
            
            # Fórmulas y explicaciones
            'formula_used': str(trend_analysis.get('formula_used', 'Fórmula Inteligente: H5 + K-means + Sinergia + Sigmoide')),
            'threshold': float(trend_analysis.get('threshold', 0.3)),
            
            # Metadatos
            'total_analysis_time': 0.0,
            'models_used': ['MobileNetV2', 'K-means', 'ColorAnalysis'],
        }
        
        logger.info("Análisis integral completado exitosamente")
        return result
        
    except Exception as e:
        logger.error(f"Error en análisis integral: {e}")
        return {
            'error': str(e),
            'analysis_method': 'comprehensive_integral_failed',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def generate_heuristic_interpretation(result):
    """Generar interpretación con fórmula completa para el nuevo sistema heurístico"""
    try:
        is_trending_flag = result.get('is_trending', False)
        h5_score = result.get('h5_score', 0)
        kmeans_score = result.get('kmeans_score', 0)
        combined_score = result.get('combined_score', 0)
        cluster_info = result.get('cluster_info', {})
        if isinstance(cluster_info, str):
            cluster_info = {}
        
        # Análisis de tendencia simplificado
        if is_trending_flag:
            trend_label = "EN TENDENCIA"
        else:
            trend_label = "NO EN TENDENCIA"
        
        # Fórmula INTELIGENTE con valores reales
        synergy_bonus = 0
        if h5_score > 0.15 and kmeans_score > 0.05:
            synergy_bonus = min(0.15, (h5_score * kmeans_score) * 2)
        
        weighted_score = (h5_score * 0.7) + (kmeans_score * 0.3) + synergy_bonus
        import math
        normalized_score = 1 / (1 + math.exp(-4 * (weighted_score - 0.3)))
        
        formula_info = f"H5({h5_score:.3f}×0.7) + K-means({kmeans_score:.3f}×0.3) + Sinergia({synergy_bonus:.3f}) = {weighted_score:.3f} | Sigmoide = {normalized_score:.3f} | Final: {combined_score:.3f}"
        
        # Información del cluster con número de imágenes - Obtener del cluster_info
        cluster_id = cluster_info.get('cluster_id', 'N/A')
        cluster_size = cluster_info.get('cluster_size', 0)
        
        cluster_explanation = f"Cluster ID: {cluster_id} ({cluster_size} imágenes)"
        
        # También actualizar cluster_info para el resultado
        cluster_info = {
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'cluster_percentage': cluster_info.get('cluster_percentage', 0)
        }
        
        # Asegurar que cluster_explanation se pase al resultado
        result['cluster_explanation'] = cluster_explanation
        
        # Información de la categoría predicha
        predicted_category = result.get('predicted_category', 'Desconocida')
        predicted_class = result.get('predicted_class', 'N/A')
        category_explanation = f"Categoría: {predicted_category} (Clase {predicted_class})"
        
        # Interpretación con fórmula completa
        interpretation = {
            'trend_analysis': trend_label,
            'formula_explanation': formula_info,
            'cluster_explanation': cluster_explanation,
            'category_explanation': category_explanation,
            'confidence_analysis': "",
            'popularity_analysis': "",
            'color_analysis': "",
            'recommendations': [],
            'summary': trend_label
        }
        
        return interpretation
        
    except Exception as e:
        logger.error(f"Error generando interpretación heurística: {e}")
        return {
            'trend_analysis': "Error en análisis",
            'formula_explanation': "",
            'cluster_explanation': "",
            'category_explanation': "",
            'confidence_analysis': "",
            'popularity_analysis': "",
            'color_analysis': "",
            'recommendations': [],
            'summary': "Error en análisis"
        }

if __name__ == '__main__':
    # Iniciar carga de modelos en segundo plano
    logger.info("🚀 Iniciando Fashion Trend App SIMPLIFICADA...")
    loading_thread = threading.Thread(target=load_models_with_progress)
    loading_thread.daemon = True
    loading_thread.start()
    
    # Ejecutar aplicación
    app.run(host='0.0.0.0', port=5000, debug=False)
