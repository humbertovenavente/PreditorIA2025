#!/usr/bin/env python3
"""
Script para debuggear la aplicación web
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import os
import pickle

def load_model_and_clustering():
    """Cargar modelo y clustering como en la aplicación web"""
    try:
        # Cargar modelo
        model_path = "data/logs/training/mobilenet_v2_final.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✅ Modelo cargado exitosamente")
        else:
            print("❌ Modelo no encontrado")
            return None, None, None
        
        # Crear extractor de características
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        print("✅ Extractor de características creado")
        
        # Cargar resultados de clustering
        clustering_path = "clustering_results/clustering_results.pkl"
        if os.path.exists(clustering_path):
            with open(clustering_path, 'rb') as f:
                clustering_results = pickle.load(f)
            print("✅ Resultados de clustering cargados")
        else:
            print("❌ Resultados de clustering no encontrados")
            clustering_results = None
        
        return model, feature_extractor, clustering_results
        
    except Exception as e:
        print(f"❌ Error cargando modelo y clustering: {e}")
        return None, None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesar imagen como en la aplicación web"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocesando imagen: {e}")
        return None

def get_class_names():
    """Obtener nombres de las clases como en la aplicación web"""
    return [
        'general', 'jewelry', 'scarves', 'shoes', 'bags', 
        'dresses', 'tops', 'pants', 'accessories', 'hats', 'other'
    ]

def predict_fashion_trend_debug(image_path, model, feature_extractor, clustering_results):
    """Función de predicción exacta como en la aplicación web"""
    try:
        if model is None:
            return None, "Modelo no cargado"
        
        # Preprocesar imagen
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None, "Error preprocesando imagen"
        
        # Hacer predicción de categoría
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        # Obtener nombre de la clase
        class_names = get_class_names()
        class_name = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
        
        print(f"🔍 PREDICCIÓN DEL MODELO:")
        print(f"   Clase predicha: {class_name}")
        print(f"   Confianza: {confidence:.1%}")
        
        # Mostrar top 3 predicciones
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        print(f"   Top 3 predicciones:")
        for i, idx in enumerate(top3_indices):
            alt_class = class_names[idx] if idx < len(class_names) else 'unknown'
            alt_confidence = float(prediction[0][idx])
            print(f"     {i+1}. {alt_class}: {alt_confidence:.1%}")
        
        # LÓGICA MEJORADA DE CORRECCIÓN
        corrected_class = class_name
        correction_reason = "Predicción original del modelo"
        
        print(f"\n🔧 APLICANDO LÓGICA DE CORRECCIÓN:")
        
        # Si la confianza es muy baja (< 60%), usar lógica alternativa
        if confidence < 0.6:
            print(f"   ⚠️  Confianza baja ({confidence:.1%}), revisando alternativas...")
            second_choice_idx = top3_indices[1]
            second_confidence = float(prediction[0][second_choice_idx])
            
            if second_confidence > confidence * 0.8:
                corrected_class = class_names[second_choice_idx] if second_choice_idx < len(class_names) else 'unknown'
                correction_reason = f"Corrección: segunda opción más confiable ({second_confidence:.1%})"
                print(f"   ✅ Corrección por confianza baja: {corrected_class}")
        
        # Reglas específicas de corrección - MÁS AGRESIVAS
        if class_name == 'scarves':
            print(f"   ⚠️  Detectada predicción problemática: 'scarves'")
            print(f"   🔍 Buscando alternativas...")
            
            for i, idx in enumerate(top3_indices[1:], 1):  # Revisar segunda y tercera opción
                alt_class = class_names[idx] if idx < len(class_names) else 'unknown'
                alt_confidence = float(prediction[0][idx])
                
                print(f"     Alternativa {i}: {alt_class} ({alt_confidence:.1%})")
                
                # Si hay una alternativa razonable (tops, dress, general) con al menos 20% de confianza
                if alt_class in ['tops', 'dress', 'general'] and alt_confidence > 0.2:
                    corrected_class = alt_class
                    correction_reason = f"Corrección: scarves → {alt_class} (confianza alternativa: {alt_confidence:.1%})"
                    print(f"     ✅ Corrección aplicada: {alt_class}")
                    break
                else:
                    print(f"     ❌ No aplicable: {alt_class} no es una alternativa válida o confianza muy baja")
        
        # Si predice 'general' con alta confianza, probablemente sea 'tops'
        if class_name == 'general' and confidence > 0.7:
            corrected_class = 'tops'
            correction_reason = "Corrección: general con alta confianza → tops"
            print(f"   ✅ Corrección general → tops")
        
        # Ajustar confianza basada en la corrección
        if corrected_class != class_name:
            final_confidence = min(confidence * 0.9, 0.85)
            print(f"   📉 Confianza ajustada: {confidence:.1%} → {final_confidence:.1%}")
        else:
            final_confidence = confidence
            print(f"   📊 Confianza sin cambios: {final_confidence:.1%}")
        
        # Análisis de clustering para determinar tendencia
        is_trendy = False
        trend_confidence = 0.0
        trend_reason = ""
        
        if clustering_results is not None and feature_extractor is not None:
            try:
                print(f"\n🔍 ANÁLISIS DE CLUSTERING:")
                
                # Cargar clusters tendenciosos
                trendy_clusters_path = "clustering_analysis/trendy_clusters.pkl"
                if os.path.exists(trendy_clusters_path):
                    with open(trendy_clusters_path, 'rb') as f:
                        trendy_data = pickle.load(f)
                    trendy_clusters = trendy_data.get('trendy_clusters', [])
                    print(f"   Clusters tendenciosos cargados: {trendy_clusters}")
                else:
                    trendy_clusters = []
                    print(f"   ⚠️  No se encontraron clusters tendenciosos")
                
                # Extraer características de la imagen
                features = feature_extractor.predict(img_array, verbose=0)
                features = features.flatten()
                print(f"   Características extraídas: {features.shape}")
                
                # Obtener características del clustering
                clustering_features = clustering_results['features']
                cluster_labels = clustering_results['cluster_labels']
                print(f"   Características de clustering: {clustering_features.shape}")
                print(f"   Labels de clustering: {len(cluster_labels)} elementos")
                
                # Calcular distancias a todos los clusters
                cluster_distances = {}
                for i, cluster_feature in enumerate(clustering_features):
                    cluster_id = cluster_labels[i]
                    if cluster_id not in cluster_distances:
                        cluster_distances[cluster_id] = []
                    dist = np.linalg.norm(features - cluster_feature)
                    cluster_distances[cluster_id].append(dist)
                
                # Calcular distancia promedio a cada cluster
                avg_distances = {}
                for cluster_id, distances in cluster_distances.items():
                    avg_distances[cluster_id] = np.mean(distances)
                
                print(f"   Distancias promedio a clusters: {avg_distances}")
                
                # Encontrar el cluster más cercano
                closest_cluster = min(avg_distances.keys(), key=lambda x: avg_distances[x])
                closest_distance = avg_distances[closest_cluster]
                
                print(f"   Cluster más cercano: {closest_cluster} (distancia: {closest_distance:.3f})")
                
                # Determinar si está de moda
                is_trendy = closest_cluster in trendy_clusters
                print(f"   ¿Está de moda? {is_trendy}")
                
                # Calcular confianza basada en distancia y coherencia
                if is_trendy:
                    trend_confidence = max(0.7, min(0.95, 1.0 - closest_distance))
                    trend_reason = f"Imagen agrupada en cluster tendencioso {closest_cluster} (distancia: {closest_distance:.3f})"
                else:
                    trend_confidence = max(0.1, min(0.6, 1.0 - closest_distance))
                    trend_reason = f"Imagen agrupada en cluster no tendencioso {closest_cluster} (distancia: {closest_distance:.3f})"
                
                print(f"   Confianza de tendencia: {trend_confidence:.1%}")
                print(f"   Razón de tendencia: {trend_reason}")
                
                # Combinar con confianza del modelo
                combined_confidence = (trend_confidence + final_confidence) / 2
                trend_confidence = combined_confidence
                print(f"   Confianza combinada: {trend_confidence:.1%}")
                
            except Exception as e:
                print(f"   ❌ Error en análisis de clustering: {e}")
                # Fallback a análisis básico
                is_trendy = final_confidence > 0.8
                trend_confidence = final_confidence
                trend_reason = f"Análisis basado en confianza del modelo ({final_confidence:.1%}) - Clustering no disponible"
        else:
            # Fallback si no hay clustering
            is_trendy = final_confidence > 0.8
            trend_confidence = final_confidence
            trend_reason = f"Análisis basado en confianza del modelo ({final_confidence:.1%}) - Clustering no disponible"
            print(f"   ⚠️  Clustering no disponible, usando análisis básico")
        
        result = {
            'is_trendy': is_trendy,
            'trend_confidence': trend_confidence,
            'trend_reason': f"{correction_reason}. {trend_reason}",
            'category': corrected_class,
            'category_confidence': final_confidence,
            'category_id': int(predicted_class),
            'original_prediction': class_name,
            'correction_applied': corrected_class != class_name
        }
        
        return result, None
        
    except Exception as e:
        print(f"❌ Error en predicción de tendencia: {e}")
        return None, str(e)

def main():
    """Función principal"""
    print("🔧 DEBUGGEANDO APLICACIÓN WEB")
    print("="*60)
    
    # Cargar modelo y clustering
    model, feature_extractor, clustering_results = load_model_and_clustering()
    if model is None:
        return
    
    # Crear imagen de prueba de t-shirt
    img = Image.new('RGB', (224, 224), (100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Dibujar forma de camiseta
    draw.ellipse([60, 40, 224-60, 80], outline=(255, 255, 255), width=3)
    draw.rectangle([70, 80, 224-70, 224-40], outline=(255, 255, 255), width=3)
    draw.text((10, 10), "T-SHIRT", fill=(255, 255, 255))
    
    # Guardar imagen de prueba
    test_path = "test_tshirt_debug.jpg"
    img.save(test_path)
    
    try:
        # Probar predicción
        result, error = predict_fashion_trend_debug(test_path, model, feature_extractor, clustering_results)
        
        if result:
            print(f"\n📊 RESULTADO FINAL:")
            print(f"   ¿Está de moda? {'Sí' if result['is_trendy'] else 'No'}")
            print(f"   Confianza de tendencia: {result['trend_confidence']:.1%}")
            print(f"   Categoría: {result['category']} ({result['category_confidence']:.1%})")
            print(f"   Predicción original: {result['original_prediction']}")
            print(f"   Corrección aplicada: {'Sí' if result['correction_applied'] else 'No'}")
            print(f"   Razón: {result['trend_reason']}")
        else:
            print(f"❌ Error en la predicción: {error}")
            
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
    
    finally:
        # Limpiar
        if os.path.exists(test_path):
            os.remove(test_path)

if __name__ == "__main__":
    main()
