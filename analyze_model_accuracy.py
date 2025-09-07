#!/usr/bin/env python3
"""
Script para analizar la precisión del modelo y identificar problemas de clasificación
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json

def load_model():
    """Cargar el modelo entrenado"""
    try:
        model_path = "data/logs/training/mobilenet_v2_final.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✅ Modelo cargado exitosamente")
            return model
        else:
            print("❌ Modelo no encontrado")
            return None
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesar imagen para el modelo"""
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
    """Obtener nombres de las clases"""
    return [
        'general', 'jewelry', 'scarves', 'shoes', 'bags', 
        'dresses', 'tops', 'pants', 'accessories', 'hats', 'other'
    ]

def analyze_model_predictions():
    """Analizar las predicciones del modelo"""
    print("🔍 ANÁLISIS DE PRECISIÓN DEL MODELO")
    print("="*50)
    
    # Cargar modelo
    model = load_model()
    if model is None:
        return
    
    # Obtener información del modelo
    print(f"📊 Información del modelo:")
    print(f"   Capas: {len(model.layers)}")
    print(f"   Parámetros totales: {model.count_params():,}")
    
    # Analizar la última capa (clasificación)
    last_layer = model.layers[-1]
    print(f"   Última capa: {last_layer.name}")
    print(f"   Unidades de salida: {last_layer.units}")
    
    # Crear imágenes de prueba para diferentes categorías
    test_images = create_test_images()
    
    print(f"\n🧪 PROBANDO CLASIFICACIÓN")
    print("="*50)
    
    class_names = get_class_names()
    results = []
    
    for category, img_array in test_images.items():
        if img_array is not None:
            # Hacer predicción
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Obtener top 3 predicciones
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            top3_predictions = []
            for idx in top3_indices:
                top3_predictions.append({
                    'class': class_names[idx] if idx < len(class_names) else f'class_{idx}',
                    'confidence': float(prediction[0][idx])
                })
            
            result = {
                'expected': category,
                'predicted': class_names[predicted_class] if predicted_class < len(class_names) else f'class_{predicted_class}',
                'confidence': confidence,
                'top3': top3_predictions
            }
            results.append(result)
            
            # Mostrar resultado
            correct = "✅" if result['expected'] == result['predicted'] else "❌"
            print(f"{correct} {category} → {result['predicted']} ({confidence:.1%})")
            
            if result['expected'] != result['predicted']:
                top3_str = [f"{p['class']} ({p['confidence']:.1%})" for p in top3_predictions]
                print(f"   Top 3: {top3_str}")
    
    # Resumen
    correct_predictions = sum(1 for r in results if r['expected'] == r['predicted'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\n📊 RESUMEN DE PRECISIÓN")
    print("="*50)
    print(f"Predicciones correctas: {correct_predictions}/{total_predictions}")
    print(f"Precisión: {accuracy:.1%}")
    
    # Identificar problemas
    print(f"\n⚠️ PROBLEMAS IDENTIFICADOS")
    print("="*50)
    for result in results:
        if result['expected'] != result['predicted']:
            print(f"❌ {result['expected']} mal clasificado como {result['predicted']}")
    
    return results

def create_test_images():
    """Crear imágenes de prueba para diferentes categorías"""
    from PIL import Image, ImageDraw
    
    test_images = {}
    size = (224, 224)
    
    # Crear imágenes de prueba simples
    categories = ['tops', 'dress', 'shoes', 'bags', 'scarves', 'jewelry']
    
    for i, category in enumerate(categories):
        try:
            # Crear imagen con forma específica para cada categoría
            img = Image.new('RGB', size, (100 + i*20, 150 + i*15, 200 - i*25))
            draw = ImageDraw.Draw(img)
            
            if category == 'tops':
                # Forma de camiseta
                draw.ellipse([60, 40, size[0]-60, 80], outline=(255, 255, 255), width=3)
                draw.rectangle([70, 80, size[0]-70, size[1]-40], outline=(255, 255, 255), width=3)
            elif category == 'dress':
                # Forma de vestido
                draw.ellipse([60, 30, size[0]-60, 70], outline=(255, 255, 255), width=3)
                draw.rectangle([50, 70, size[0]-50, size[1]-20], outline=(255, 255, 255), width=3)
            elif category == 'shoes':
                # Forma de zapato
                draw.ellipse([40, 80, size[0]-40, size[1]-40], outline=(255, 255, 255), width=3)
            elif category == 'bags':
                # Forma de bolso
                draw.rectangle([60, 60, size[0]-60, size[1]-60], outline=(255, 255, 255), width=3)
                draw.rectangle([70, 50, size[0]-70, 70], outline=(255, 255, 255), width=3)
            elif category == 'scarves':
                # Forma de bufanda
                draw.rectangle([30, 80, size[0]-30, 120], outline=(255, 255, 255), width=3)
            elif category == 'jewelry':
                # Forma de joya
                draw.ellipse([80, 80, size[0]-80, size[1]-80], outline=(255, 255, 255), width=3)
            
            # Agregar texto
            draw.text((10, 10), category, fill=(255, 255, 255))
            
            # Convertir a array
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            test_images[category] = img_array
            
        except Exception as e:
            print(f"Error creando imagen de prueba para {category}: {e}")
            test_images[category] = None
    
    return test_images

def suggest_improvements():
    """Sugerir mejoras para el modelo"""
    print(f"\n💡 SUGERENCIAS DE MEJORA")
    print("="*50)
    print("1. 🔄 Re-entrenar con más datos de 'tops' y 'shirts'")
    print("2. 📊 Balancear el dataset (más imágenes de camisetas)")
    print("3. 🎯 Usar data augmentation específico para ropa")
    print("4. 🔍 Ajustar la arquitectura del modelo")
    print("5. 📈 Implementar fine-tuning con transfer learning")
    print("6. 🏷️ Revisar las etiquetas del dataset de entrenamiento")

def main():
    """Función principal"""
    results = analyze_model_predictions()
    suggest_improvements()
    
    # Guardar resultados
    with open('model_accuracy_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados guardados en 'model_accuracy_analysis.json'")

if __name__ == "__main__":
    main()
