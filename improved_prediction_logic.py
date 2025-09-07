
def predict_fashion_trend_improved(image_path):
    """Predicción mejorada de tendencia de moda"""
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
        
        # LÓGICA MEJORADA DE CORRECCIÓN
        corrected_class = class_name
        correction_reason = "Predicción original del modelo"
        
        # Si la confianza es muy baja (< 60%), usar lógica alternativa
        if confidence < 0.6:
            # Obtener top 3 predicciones
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            second_choice_idx = top3_indices[1]
            second_confidence = float(prediction[0][second_choice_idx])
            
            if second_confidence > confidence * 0.8:
                corrected_class = class_names[second_choice_idx] if second_choice_idx < len(class_names) else 'unknown'
                correction_reason = f"Corrección: segunda opción más confiable ({second_confidence:.1%})"
        
        # Reglas específicas de corrección
        if class_name == 'scarves' and confidence < 0.8:
            # Si predice scarves con baja confianza, probablemente sea otra cosa
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            for idx in top3_indices[1:]:  # Revisar segunda y tercera opción
                alt_class = class_names[idx] if idx < len(class_names) else 'unknown'
                if alt_class in ['tops', 'dress']:
                    corrected_class = alt_class
                    correction_reason = f"Corrección: scarves con baja confianza → {alt_class}"
                    break
        
        # Si predice 'general' con alta confianza, probablemente sea 'tops'
        if class_name == 'general' and confidence > 0.7:
            corrected_class = 'tops'
            correction_reason = "Corrección: general con alta confianza → tops"
        
        # Ajustar confianza basada en la corrección
        if corrected_class != class_name:
            final_confidence = min(confidence * 0.9, 0.85)
        else:
            final_confidence = confidence
        
        # Análisis de clustering para determinar tendencia
        is_trendy = False
        trend_confidence = 0.0
        trend_reason = ""
        
        if clustering_results is not None and feature_extractor is not None:
            try:
                # Cargar clusters tendenciosos
                trendy_clusters_path = "clustering_analysis/trendy_clusters.pkl"
                if os.path.exists(trendy_clusters_path):
                    with open(trendy_clusters_path, 'rb') as f:
                        trendy_data = pickle.load(f)
                    trendy_clusters = trendy_data.get('trendy_clusters', [])
                else:
                    trendy_clusters = []
                
                # Extraer características de la imagen
                features = feature_extractor.predict(img_array, verbose=0)
                features = features.flatten()
                
                # Obtener características del clustering
                clustering_features = clustering_results['features']
                cluster_labels = clustering_results['cluster_labels']
                
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
                
                # Encontrar el cluster más cercano
                closest_cluster = min(avg_distances.keys(), key=lambda x: avg_distances[x])
                closest_distance = avg_distances[closest_cluster]
                
                # Determinar si está de moda
                is_trendy = closest_cluster in trendy_clusters
                
                # Calcular confianza basada en distancia y coherencia
                if is_trendy:
                    trend_confidence = max(0.7, min(0.95, 1.0 - closest_distance))
                    trend_reason = f"Imagen agrupada en cluster tendencioso {closest_cluster} (distancia: {closest_distance:.3f})"
                else:
                    trend_confidence = max(0.1, min(0.6, 1.0 - closest_distance))
                    trend_reason = f"Imagen agrupada en cluster no tendencioso {closest_cluster} (distancia: {closest_distance:.3f})"
                
                # Combinar con confianza del modelo
                combined_confidence = (trend_confidence + final_confidence) / 2
                trend_confidence = combined_confidence
                
            except Exception as e:
                logger.warning(f"Error en análisis de clustering: {e}")
                # Fallback a análisis básico
                is_trendy = final_confidence > 0.8
                trend_confidence = final_confidence
                trend_reason = f"Análisis basado en confianza del modelo ({final_confidence:.1%}) - Clustering no disponible"
        else:
            # Fallback si no hay clustering
            is_trendy = final_confidence > 0.8
            trend_confidence = final_confidence
            trend_reason = f"Análisis basado en confianza del modelo ({final_confidence:.1%}) - Clustering no disponible"
        
        return {
            'is_trendy': is_trendy,
            'trend_confidence': trend_confidence,
            'trend_reason': f"{correction_reason}. {trend_reason}",
            'category': corrected_class,
            'category_confidence': final_confidence,
            'category_id': int(predicted_class),
            'original_prediction': class_name,
            'correction_applied': corrected_class != class_name
        }, None
        
    except Exception as e:
        logger.error(f"Error en predicción de tendencia: {e}")
        return None, str(e)
