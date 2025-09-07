#!/usr/bin/env python3
"""
Script para probar la lógica de corrección directamente
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import os

def load_model():
    """Cargar el modelo existente"""
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
    """Preprocesar imagen"""
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

def test_correction_logic():
    """Probar la lógica de corrección directamente"""
    print("🧪 PROBANDO LÓGICA DE CORRECCIÓN DIRECTAMENTE")
    print("="*60)
    
    # Cargar modelo
    model = load_model()
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
    test_path = "test_tshirt_direct.jpg"
    img.save(test_path)
    
    try:
        # Preprocesar imagen
        img_array = preprocess_image(test_path)
        if img_array is None:
            print("❌ Error preprocesando imagen")
            return
        
        # Hacer predicción
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        # Obtener nombres de clases
        class_names = get_class_names()
        class_name = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
        
        print(f"📊 PREDICCIÓN ORIGINAL:")
        print(f"   Clase: {class_name}")
        print(f"   Confianza: {confidence:.1%}")
        
        # Mostrar top 3 predicciones
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        print(f"   Top 3 predicciones:")
        for i, idx in enumerate(top3_indices):
            alt_class = class_names[idx] if idx < len(class_names) else 'unknown'
            alt_confidence = float(prediction[0][idx])
            print(f"     {i+1}. {alt_class}: {alt_confidence:.1%}")
        
        # APLICAR LÓGICA DE CORRECCIÓN
        print(f"\n🔧 APLICANDO LÓGICA DE CORRECCIÓN:")
        corrected_class = class_name
        correction_reason = "Predicción original del modelo"
        
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
        
        # Ajustar confianza basada en la corrección
        if corrected_class != class_name:
            final_confidence = min(confidence * 0.9, 0.85)
            print(f"   📉 Confianza ajustada: {confidence:.1%} → {final_confidence:.1%}")
        else:
            final_confidence = confidence
            print(f"   📊 Confianza sin cambios: {final_confidence:.1%}")
        
        print(f"\n📋 RESULTADO FINAL:")
        print(f"   Predicción original: {class_name}")
        print(f"   Categoría corregida: {corrected_class}")
        print(f"   Confianza final: {final_confidence:.1%}")
        print(f"   Razón: {correction_reason}")
        print(f"   Corrección aplicada: {'Sí' if corrected_class != class_name else 'No'}")
        
        # Verificar si la corrección fue exitosa
        if corrected_class == 'tops' and class_name == 'scarves':
            print(f"\n✅ ¡CORRECCIÓN EXITOSA! T-shirt corregida de 'scarves' a 'tops'")
        elif corrected_class == 'tops':
            print(f"\n✅ ¡CLASIFICACIÓN CORRECTA! T-shirt clasificada como 'tops'")
        else:
            print(f"\n❌ Clasificación incorrecta: {corrected_class}")
            
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
    
    finally:
        # Limpiar
        if os.path.exists(test_path):
            os.remove(test_path)

def main():
    """Función principal"""
    print("🔧 PROBANDO LÓGICA DE CORRECCIÓN DIRECTAMENTE")
    print("="*60)
    
    test_correction_logic()
    
    print(f"\n✅ Prueba completada!")

if __name__ == "__main__":
    main()
