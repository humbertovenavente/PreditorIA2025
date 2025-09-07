#!/usr/bin/env python3
"""
Fashion Trend Analysis App - Con Barra de Progreso
Usa el modelo entrenado y clustering con indicador de progreso
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from datetime import datetime
import threading
import time

# Importar módulos de clustering
import sys
sys.path.append('/home/jose/PreditorIA2025')
from fashion_clustering.utils.vision import load_mobilenetv2, extract_embeddings
from fashion_clustering.utils.colors import extract_dominant_colors, get_color_names

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesar imagen para el modelo"""
    try:
        from PIL import Image
        import tensorflow as tf
        
        # Cargar imagen
        img = Image.open(image_path)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize(target_size)
        
        # Convertir a array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalización simple (0-1) - más compatible con modelos entrenados
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocesando {image_path}: {e}")
        return None

# Configuración
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fashion_trend_2025_guatemala'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'

# Crear directorio de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para modelos
model = None
feature_extractor = None
clustering_data = None
cluster_centers = None
cluster_stats = None
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
    global model, feature_extractor, clustering_data, cluster_centers, cluster_stats, models_loaded, loading_progress, loading_status
    
    try:
        loading_progress = 0
        loading_status = "Iniciando carga de modelos..."
        logger.info("🔄 Iniciando carga de modelos...")
        
        # Paso 1: Cargar modelo de visión entrenado (20%)
        loading_progress = 10
        loading_status = "Cargando modelo entrenado..."
        logger.info("📦 Cargando modelo entrenado...")
        
        model_path = '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Modelo cargado desde {model_path}")
        else:
            # Usar MobileNetV2 preentrenado si no existe el modelo afinado
            loading_status = "Cargando MobileNetV2 preentrenado..."
            model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet'
            )
            logger.info("✅ Usando MobileNetV2 preentrenado")
        
        loading_progress = 30
        loading_status = "Configurando extractor de características..."
        
        # Paso 2: El modelo ya tiene la salida correcta de 11 dimensiones
        logger.info("🔧 El modelo ya tiene la salida correcta de 11 dimensiones")
        feature_extractor = model  # Usar el modelo completo
        
        loading_progress = 40
        loading_status = "Cargando datos de clustering..."
        
        # Paso 3: Cargar datos de clustering (30%)
        logger.info("📊 Cargando datos de clustering...")
        clustering_data = load_clustering_data()
        
        loading_progress = 70
        loading_status = "Calculando centros de clusters..."
        
        # Paso 4: Calcular centros de clusters (20%)
        logger.info("🎯 Calculando centros de clusters...")
        cluster_centers = calculate_cluster_centers()
        
        loading_progress = 90
        loading_status = "Cargando estadísticas..."
        
        # Paso 5: Cargar estadísticas (10%)
        logger.info("📈 Cargando estadísticas...")
        cluster_stats = load_cluster_stats()
        
        loading_progress = 100
        loading_status = "¡Modelos cargados exitosamente!"
        models_loaded = True
        
        logger.info("✅ Todos los modelos cargados exitosamente")
        
    except Exception as e:
        loading_status = f"Error: {str(e)}"
        logger.error(f"❌ Error cargando modelos: {e}")
        models_loaded = False

def load_clustering_data():
    """Cargar datos de clustering desde clustering_results/"""
    try:
        # Cargar resultados completos del clustering
        with open('/home/jose/PreditorIA2025/clustering_results/clustering_results.pkl', 'rb') as f:
            clustering_results = pickle.load(f)
        
        # Cargar modelo K-means
        with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        
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
            'n_clusters': clustering_results.get('n_clusters', 150),
            'total_images': clustering_results.get('total_images', 16959)
        }
    except Exception as e:
        logger.error(f"❌ Error cargando datos de clustering: {e}")
        return None

def calculate_cluster_centers():
    """Calcular centros de clusters usando el modelo K-means"""
    if clustering_data is None:
        return None
    
    try:
        # Usar los centros de clusters del modelo K-means entrenado
        kmeans_model = clustering_data['kmeans_model']
        cluster_centers = {}
        
        for i, center in enumerate(kmeans_model.cluster_centers_):
            cluster_centers[i] = center
        
        return cluster_centers
    except Exception as e:
        logger.error(f"❌ Error calculando centros de clusters: {e}")
        return None

def load_cluster_stats():
    """Cargar estadísticas de clusters desde los datos de clustering"""
    try:
        if clustering_data is None:
            return None
        
        # Crear estadísticas desde los datos de clustering
        cluster_counts = clustering_data['cluster_counts']
        metrics = clustering_data['metrics']
        
        # Convertir cluster_counts a formato esperado
        cluster_sizes = {}
        for cluster_id, count in cluster_counts.items():
            cluster_sizes[f'cluster_{cluster_id}'] = count
        
        stats = {
            'cluster_sizes': cluster_sizes,
            'algorithm': 'kmeans',
            'silhouette_score': metrics['silhouette'],
            'calinski_harabasz_score': metrics['calinski_harabasz'],
            'davies_bouldin_score': metrics['davies_bouldin'],
            'total_clusters': clustering_data['n_clusters'],
            'total_images': clustering_data['total_images']
        }
        
        return stats
    except Exception as e:
        logger.error(f"❌ Error cargando estadísticas: {e}")
        return None

def predict_image(image_path):
    """Realizar predicción en una imagen"""
    if not models_loaded:
        return None, None, None, "Modelos aún cargando..."
    
    try:
        # Preprocesar imagen
        image = preprocess_image(image_path, target_size=(224, 224))
        image_batch = np.expand_dims(image, axis=0)
        
        # Usar el modelo completo para predicción de clase
        prediction = model.predict(image_batch, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Extraer embedding para clustering (usar la salida del modelo como embedding)
        embedding = prediction.astype(np.float64)
        
        # Usar el modelo K-means para predecir el cluster
        if clustering_data and 'kmeans_model' in clustering_data:
            kmeans_model = clustering_data['kmeans_model']
            predicted_cluster = kmeans_model.predict([embedding])[0]
        else:
            predicted_cluster = None
        
        return embedding, predicted_class, confidence, None
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        return None, None, None, str(e)

def find_closest_cluster(embedding):
    """Encontrar el cluster más cercano usando K-means"""
    if clustering_data is None or 'kmeans_model' not in clustering_data:
        return None, None
    
    try:
        kmeans_model = clustering_data['kmeans_model']
        
        # Convertir embedding a float64 si es necesario
        embedding = embedding.astype(np.float64)
        
        # Predecir cluster usando K-means
        predicted_cluster = kmeans_model.predict([embedding])[0]
        
        # Calcular distancia al centro del cluster
        cluster_center = kmeans_model.cluster_centers_[predicted_cluster]
        distance = np.linalg.norm(embedding - cluster_center)
        
        # Normalizar distancia a score de similitud (0-100)
        max_distance = 10.0  # Ajustar según el dataset
        similarity_score = max(0, 100 - (distance / max_distance) * 100)
        
        return predicted_cluster, similarity_score
        
    except Exception as e:
        logger.error(f"❌ Error encontrando cluster: {e}")
        return None, None

def calculate_trend_score(cluster_id, similarity_score):
    """Calcular TrendScore basado en cluster y similitud"""
    if cluster_stats is None or cluster_id is None:
        return 50  # Score neutral
    
    try:
        # Obtener estadísticas del cluster
        cluster_sizes = cluster_stats.get('cluster_sizes', {})
        cluster_key = f'cluster_{cluster_id}'
        
        if cluster_key in cluster_sizes:
            cluster_size = cluster_sizes[cluster_key]
        else:
            cluster_size = 0
        
        # Calcular score base por tamaño del cluster (0-40 puntos)
        max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1
        size_score = (cluster_size / max_cluster_size) * 40
        
        # Score por similitud (0-60 puntos)
        similarity_contribution = similarity_score * 60
        
        # Bonus por cluster grande (clusters con muchas imágenes son más "trendy")
        if cluster_size > 100:  # Clusters con más de 100 imágenes
            size_bonus = min(10, cluster_size / 100)  # Bonus de hasta 10 puntos
        else:
            size_bonus = 0
        
        # Score total
        trend_score = min(100, size_score + similarity_contribution + size_bonus)
        
        return int(trend_score)
        
    except Exception as e:
        logger.error(f"❌ Error calculando trend score: {e}")
        return 50

def analyze_image_colors(image_path):
    """Analizar colores dominantes de la imagen"""
    try:
        colors = extract_dominant_colors(image_path, n_colors=5)
        color_names = get_color_names(colors)
        logger.info(f"🎨 Colores detectados: {color_names}")
        return color_names
    except Exception as e:
        logger.error(f"❌ Error analizando colores: {e}")
        return []

@app.route('/')
def index():
    """Página principal"""
    # Forzar mostrar barra de progreso si no está cargado
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
        if analysis_result is not None:
            # Convertir numpy arrays a listas para serialización JSON
            serializable_result = {}
            for key, value in analysis_result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            
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
        logger.error(f"❌ Error en api_analysis_result: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error procesando resultado del análisis'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Procesar subida de imagen"""
    if not models_loaded:
        flash('Los modelos aún se están cargando. Por favor, espera unos momentos.')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No se seleccionó archivo')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No se seleccionó archivo')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Guardar archivo
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar imagen
            result = process_image(filepath)
            
            if result:
                return render_template('result.html', 
                                    image_path=filename,
                                    result=result)
            else:
                flash('Error procesando la imagen')
                return redirect(url_for('index'))
                
        except Exception as e:
            logger.error(f"❌ Error procesando archivo: {e}")
            flash(f'Error procesando imagen: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Tipo de archivo no permitido')
        return redirect(url_for('index'))

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
        
        # Iniciar análisis en un hilo separado
        analysis_thread = threading.Thread(target=process_image, args=(filepath,))
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Análisis iniciado',
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"❌ Error en API analyze: {e}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Verificar si el archivo está permitido"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Procesar imagen completa con progreso"""
    global analysis_progress, analysis_status, analysis_active, analysis_result, analysis_filename
    
    try:
        analysis_active = True
        analysis_progress = 0
        analysis_status = "Iniciando análisis..."
        analysis_result = None
        analysis_filename = os.path.basename(image_path)
        logger.info("🔄 Procesando imagen...")
        
        # Paso 1: Preprocesamiento (20%)
        analysis_progress = 20
        analysis_status = "Preprocesando imagen..."
        logger.info("🔄 Iniciando preprocesamiento...")
        time.sleep(0.5)  # Simular tiempo de procesamiento
        
        # Predicción
        logger.info("🔄 Iniciando predicción...")
        embedding, predicted_class, confidence, error = predict_image(image_path)
        logger.info(f"🔄 Predicción completada - embedding: {embedding is not None}, error: {error}")
        
        if embedding is None:
            logger.error(f"❌ Error en predicción: {error}")
            analysis_active = False
            analysis_result = None
            return None
        
        # Paso 2: Extracción de características (40%)
        analysis_progress = 40
        analysis_status = "Extrayendo características..."
        time.sleep(0.5)
        
        logger.info("🎯 Encontrando cluster...")
        # Encontrar cluster
        cluster_id, similarity_score = find_closest_cluster(embedding)
        
        # Paso 3: Asignación de cluster (60%)
        analysis_progress = 60
        analysis_status = "Asignando cluster..."
        time.sleep(0.5)
        
        logger.info("📊 Calculando trend score...")
        # Calcular trend score
        trend_score = calculate_trend_score(cluster_id, similarity_score)
        
        # Calcular confianza combinada (modelo + clustering)
        if similarity_score is not None:
            # Combinar confianza del modelo (0-1) con similitud del clustering (0-100)
            model_confidence_normalized = confidence * 100  # Convertir a 0-100
            clustering_confidence = similarity_score  # Ya está en 0-100
            
            # Promedio ponderado: 60% modelo, 40% clustering
            combined_confidence = (model_confidence_normalized * 0.6 + clustering_confidence * 0.4) / 100
            combined_confidence = min(1.0, max(0.0, combined_confidence))  # Asegurar rango 0-1
        else:
            combined_confidence = confidence
        
        # Paso 4: Análisis de colores (80%)
        analysis_progress = 80
        analysis_status = "Analizando colores..."
        time.sleep(0.5)
        
        logger.info("🎨 Analizando colores...")
        # Analizar colores
        colors = analyze_image_colors(image_path)
        
        # Paso 5: Finalizando (100%)
        analysis_progress = 100
        analysis_status = "Finalizando análisis..."
        time.sleep(0.5)
        
        # Determinar si está en tendencia
        is_trending = trend_score >= 70
        
        # Obtener información del cluster
        cluster_info = get_cluster_info(cluster_id)
        
        result = {
            'trend_score': trend_score,
            'is_trending': is_trending,
            'cluster_id': cluster_id,
            'similarity_score': similarity_score,
            'predicted_class': predicted_class,
            'confidence': combined_confidence,  # Usar confianza combinada
            'model_confidence': confidence,  # Confianza original del modelo
            'clustering_confidence': similarity_score / 100 if similarity_score else 0,  # Confianza del clustering
            'colors': colors,
            'cluster_info': cluster_info,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generar interpretación inteligente
        interpretation = generate_fashion_interpretation(result)
        result['interpretation'] = interpretation
        
        analysis_result = result
        analysis_active = False
        analysis_progress = 0
        analysis_status = "Análisis completado"
        
        logger.info("✅ Procesamiento completado")
        return result
        
    except Exception as e:
        analysis_active = False
        analysis_progress = 0
        analysis_status = f"Error: {str(e)}"
        analysis_result = None
        logger.error(f"❌ Error procesando imagen: {e}")
        return None

def get_cluster_info(cluster_id):
    """Obtener información detallada del cluster"""
    if cluster_stats is None or cluster_id is None:
        return None
    
    try:
        cluster_sizes = cluster_stats.get('cluster_sizes', {})
        cluster_key = f'cluster_{cluster_id}'
        
        size = cluster_sizes.get(cluster_key, 0)
        
        # Obtener información de categorías dominantes en el cluster
        category_info = ""
        if clustering_data and 'category_cluster_analysis' in clustering_data:
            category_analysis = clustering_data['category_cluster_analysis']
            cluster_categories = {}
            
            for category, clusters in category_analysis.items():
                if cluster_id in clusters:
                    cluster_categories[category] = clusters[cluster_id]
            
            if cluster_categories:
                dominant_category = max(cluster_categories.items(), key=lambda x: x[1])
                category_info = f" - Dominado por: {dominant_category[0]} ({dominant_category[1]} imágenes)"
        
        return {
            'size': size,
            'description': f'Cluster {cluster_id} con {size} imágenes similares{category_info}',
            'total_clusters': cluster_stats.get('total_clusters', 150),
            'total_images': cluster_stats.get('total_images', 16959)
        }
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo info del cluster: {e}")
        return None

def generate_fashion_interpretation(result):
    """Generar interpretación inteligente de los resultados de moda"""
    try:
        trend_score = result.get('trend_score', 0)
        colors = result.get('colors', [])
        is_trending = result.get('is_trending', False)
        cluster_id = result.get('cluster_id', 0)
        similarity_score = result.get('similarity_score', 0)
        confidence = result.get('confidence', 0)
        predicted_class = result.get('predicted_class', 0)
        
        # Análisis de tendencia basado en análisis estadístico real del sistema
        # Umbrales científicos: P25=11, P75=26 (basado en fórmula corregida del sistema con 16,959 imágenes)
        if trend_score >= 26:
            trend_analysis = "EN TENDENCIA - Estilo popular y actual"
        elif trend_score >= 11:
            trend_analysis = "NEUTRO - Estilo clásico y equilibrado"
        else:
            trend_analysis = "NO EN TENDENCIA - Estilo poco popular"
        
        # Explicación del Cluster ID
        cluster_info = get_cluster_info(cluster_id)
        if cluster_info:
            cluster_explanation = f"Cluster {cluster_id} - {cluster_info['description']}. Este grupo contiene prendas con características visuales similares de nuestro dataset de {cluster_info['total_images']} imágenes organizadas en {cluster_info['total_clusters']} clusters."
        else:
            cluster_explanation = f"Cluster {cluster_id} - Este grupo contiene prendas con características visuales similares de nuestro dataset de 16,959 imágenes organizadas en 150 clusters."
        
        # Explicación de Similitud
        similarity_explanation = f"Similitud {similarity_score:.1f}% - Indica qué tan parecida es tu prenda a otras en el mismo cluster. Un porcentaje alto significa que tu estilo es muy consistente con el grupo."
        
        # Explicación de Clase Predicha - Mapeo basado en las categorías reales del dataset
        class_names = {
            0: "accesorios",
            1: "estilo", 
            2: "fashion",
            3: "general",
            4: "jewelry",
            5: "moda",
            6: "ropa",
            7: "runway",
            8: "scarves",
            9: "sombreros",
            10: "vestidos"
        }
        
        # Manejar casos donde la clase predicha está fuera del rango esperado
        if predicted_class is not None and 0 <= predicted_class <= 10:
            predicted_class_name = class_names.get(predicted_class, f"Categoría {predicted_class}")
        else:
            # Si la clase está fuera del rango, usar una descripción genérica
            predicted_class_name = f"Clase {predicted_class} (fuera del rango esperado)"
        
        class_explanation = f"Clase Predicha: {predicted_class_name.title()} - El sistema clasificó tu prenda en esta categoría basándose en sus características visuales."
        
        # Análisis de confianza
        if confidence >= 0.8:
            confidence_analysis = "MUY CONFIABLE - El análisis es muy preciso y confiable"
        elif confidence >= 0.6:
            confidence_analysis = "CONFIABLE - El análisis es preciso y confiable"
        elif confidence >= 0.4:
            confidence_analysis = "MODERADAMENTE CONFIABLE - El análisis puede variar ligeramente"
        else:
            confidence_analysis = "POCO CONFIABLE - El análisis puede ser impreciso"
        
        # Análisis de colores
        color_analysis = analyze_color_combination(colors)
        
        # Asegurar que colors es una lista de strings
        if colors and isinstance(colors[0], tuple):
            colors = [str(color) for color in colors]
        
        # Recomendaciones
        recommendations = generate_recommendations(trend_score, colors, is_trending)
        
        # Interpretación completa
        interpretation = {
            'trend_analysis': trend_analysis,
            'cluster_explanation': cluster_explanation,
            'similarity_explanation': similarity_explanation,
            'class_explanation': class_explanation,
            'color_analysis': color_analysis,
            'confidence_analysis': confidence_analysis,
            'recommendations': recommendations,
            'summary': ""
        }
        
        return interpretation
        
    except Exception as e:
        logger.error(f"Error generando interpretación: {e}")
        return {
            'trend_analysis': "Error en análisis",
            'cluster_explanation': "No disponible",
            'similarity_explanation': "No disponible",
            'class_explanation': "No disponible",
            'color_analysis': "No disponible",
            'confidence_analysis': "Error en análisis",
            'recommendations': [],
            'summary': "Error generando interpretación"
        }

def analyze_color_combination(colors):
    """Analizar combinación de colores con mayor detalle"""
    if not colors:
        return "Sin colores detectados"
    
    # Contar colores
    color_counts = {}
    for color in colors:
        color_counts[color] = color_counts.get(color, 0) + 1
    
    # Colores más frecuentes
    dominant_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Crear descripción detallada
    color_list = list(color_counts.keys())
    primary_color = dominant_colors[0][0] if dominant_colors else ""
    
    # Análisis de paleta mejorado con todos los colores
    # Colores cálidos
    warm_colors = ['rojo', 'naranja', 'amarillo', 'rosa', 'rojo-naranja', 'naranja-amarillo', 
                   'amarillo-verde', 'verde-amarillo', 'rosa-rojo', 'rojo-rosa']
    
    # Colores fríos
    cool_colors = ['azul', 'verde', 'morado', 'cian', 'azul-verde', 'verde-azul', 
                   'azul-cian', 'azul-morado', 'morado-azul', 'morado-rosa', 'rosa-morado']
    
    # Colores neutros
    neutral_colors = ['negro', 'blanco', 'gris', 'gris claro', 'gris oscuro', 'gris muy oscuro']
    
    # Colores especiales
    special_colors = ['violeta', 'verde lima', 'azul marino']
    
    # Detectar tipos de colores presentes
    has_warm = any(color in color_list for color in warm_colors)
    has_cool = any(color in color_list for color in cool_colors)
    has_neutral = any(color in color_list for color in neutral_colors)
    has_special = any(color in color_list for color in special_colors)
    
    # Análisis de paleta mejorado
    if has_warm and has_cool:
        return f"Paleta Complementaria - {primary_color} dominante con contraste cálido-frío, creando un look vibrante y equilibrado"
    elif 'rojo' in color_list and 'azul' in color_list:
        return f"Paleta Complementaria Clásica - {primary_color} con acentos azules, contraste vibrante y moderno"
    elif 'rojo' in color_list and 'verde' in color_list:
        return f"Paleta Contrastante - {primary_color} principal con toques verdes, look audaz y llamativo"
    elif 'negro' in color_list and 'blanco' in color_list:
        return f"Paleta Clásica - {primary_color} con base neutra, eternamente elegante y versátil"
    elif has_warm and not has_cool:
        if 'rojo' in color_list or any('rojo' in color for color in color_list):
            return f"Paleta Cálida - Dominada por {primary_color}, transmite pasión y energía"
        elif 'naranja' in color_list or any('naranja' in color for color in color_list):
            return f"Paleta Energética - {primary_color} dominante, transmite creatividad y entusiasmo"
        elif 'amarillo' in color_list or any('amarillo' in color for color in color_list):
            return f"Paleta Vibrante - {primary_color} principal, transmite alegría y optimismo"
        elif 'rosa' in color_list or any('rosa' in color for color in color_list):
            return f"Paleta Romántica - {primary_color} dominante, transmite dulzura y feminidad"
        else:
            return f"Paleta Cálida - {primary_color} principal, transmite calidez y vitalidad"
    elif has_cool and not has_warm:
        if 'azul' in color_list or any('azul' in color for color in color_list):
            return f"Paleta Fría - Dominada por {primary_color}, transmite calma y profesionalismo"
        elif 'verde' in color_list or any('verde' in color for color in color_list):
            return f"Paleta Natural - {primary_color} dominante, transmite frescura y vitalidad"
        elif 'morado' in color_list or any('morado' in color for color in color_list):
            return f"Paleta Real - {primary_color} principal, transmite elegancia y misterio"
        elif 'cian' in color_list or any('cian' in color for color in color_list):
            return f"Paleta Acuática - {primary_color} dominante, transmite frescura y modernidad"
        else:
            return f"Paleta Fría - {primary_color} principal, transmite serenidad y sofisticación"
    elif has_neutral:
        if 'negro' in color_list:
            return f"Paleta Oscura - Elegante y sofisticada con {primary_color}, perfecta para ocasiones formales"
        elif 'blanco' in color_list or 'gris claro' in color_list:
            return f"Paleta Clara - Fresca y luminosa con {primary_color}, ideal para looks minimalistas"
        else:
            return f"Paleta Neutra - {primary_color} con base gris, elegante y versátil"
    elif has_special:
        if 'violeta' in color_list or any('violeta' in color for color in color_list):
            return f"Paleta Real - {primary_color} principal, transmite elegancia y misterio"
        elif 'verde lima' in color_list or any('verde lima' in color for color in color_list):
            return f"Paleta Fresca - {primary_color} dominante, transmite juventud y energía"
        elif 'azul marino' in color_list or any('azul marino' in color for color in color_list):
            return f"Paleta Náutica - {primary_color} principal, transmite confianza y profesionalismo"
        else:
            return f"Paleta Especial - {primary_color} dominante, transmite originalidad y distinción"
    else:
        # Crear descripción personalizada basada en los colores detectados
        color_description = ", ".join([color for color, _ in dominant_colors[:3]])  # Top 3 colores
        return f"Paleta Personalizada - Combinación única de {color_description}, creando un look distintivo"

def generate_recommendations(trend_score, colors, is_trending):
    """Generar recomendaciones de moda"""
    recommendations = []
    
    # Recomendaciones basadas en tendencia
    if trend_score >= 70:
        recommendations.append("Mantén este estilo - Está en tendencia y te hace ver actual")
        recommendations.append("Combina con accesorios - Añade joyas o bolsos para completar el look")
    elif trend_score >= 50:
        recommendations.append("Estilo equilibrado - Clásico y atemporal, perfecto para cualquier ocasión")
        recommendations.append("Considera actualizarlo - Añade elementos modernos para darle un toque actual")
    else:
        recommendations.append("Actualiza tu guardarropa - Este estilo está pasando de moda")
        recommendations.append("Inspírate en tendencias - Busca looks más actuales en redes sociales")
    
    # Recomendaciones basadas en colores
    if 'rojo' in colors:
        recommendations.append("El rojo es atrevido - Perfecto para ocasiones especiales y para destacar")
    if 'negro' in colors:
        recommendations.append("El negro es versátil - Combina con cualquier color y es apropiado para cualquier ocasión")
    if 'azul' in colors:
        recommendations.append("El azul es confiable - Transmite confianza y profesionalismo")
    
    # Recomendaciones generales
    if len(set(colors)) > 3:
        recommendations.append("Paleta compleja - Considera simplificar con 2-3 colores principales")
    
    return recommendations

@app.route('/api/stats')
def api_stats():
    """API endpoint para estadísticas generales"""
    if not models_loaded:
        return jsonify({'error': 'Modelos aún cargando'}), 503
    
    if cluster_stats is None:
        return jsonify({'error': 'Clustering data not loaded'}), 500
    
    try:
        stats = {
            'total_clusters': cluster_stats.get('total_clusters', 150),
            'total_images': cluster_stats.get('total_images', 16959),
            'algorithm': cluster_stats.get('algorithm', 'kmeans'),
            'silhouette_score': cluster_stats.get('silhouette_score', 0.4687),
            'calinski_harabasz_score': cluster_stats.get('calinski_harabasz_score', 17223.31),
            'davies_bouldin_score': cluster_stats.get('davies_bouldin_score', 0.9283),
            'model_info': {
                'name': 'MobileNetV2 Fine-tuned',
                'checkpoint': '/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5',
                'embedding_dimensions': 11
            }
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"❌ Error en stats API: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Manejar archivos muy grandes"""
    flash('Archivo demasiado grande. Máximo 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Manejar páginas no encontradas"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Manejar errores internos"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Iniciar carga de modelos en segundo plano
    logger.info("🚀 Iniciando Fashion Trend App con progreso...")
    loading_thread = threading.Thread(target=load_models_with_progress)
    loading_thread.daemon = True
    loading_thread.start()
    
    # Ejecutar aplicación
    app.run(host='0.0.0.0', port=5000, debug=False)
