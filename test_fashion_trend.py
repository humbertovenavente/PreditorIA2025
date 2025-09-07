#!/usr/bin/env python3
"""
Script de prueba para la funcionalidad de tendencia de moda
"""

import requests
import os
from PIL import Image
import numpy as np
import time

def create_test_image():
    """Crea una imagen de prueba"""
    test_image_path = "test_fashion.png"
    
    # Crear imagen de prueba
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(test_image_path)
    
    return test_image_path

def test_fashion_trend():
    """Prueba la funcionalidad de tendencia de moda"""
    print("🎯 Probando funcionalidad de tendencia de moda...")
    print("="*50)
    
    # Crear imagen de prueba
    test_image_path = create_test_image()
    
    try:
        # Subir imagen
        print("📤 Subiendo imagen...")
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/upload', files=files)
        
        if response.status_code != 200:
            print(f"❌ Error subiendo imagen: {response.status_code}")
            return
        
        upload_data = response.json()
        print("✅ Imagen subida exitosamente")
        
        # Analizar tendencia
        print("🔍 Analizando tendencia de moda...")
        predict_data = {'filepath': upload_data.get('filepath')}
        response = requests.post(
            'http://localhost:5000/predict',
            json=predict_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            print(f"❌ Error en análisis: {response.status_code}")
            return
        
        result = response.json()
        prediction = result.get('prediction', {})
        
        # Mostrar resultados
        print("\n📊 RESULTADOS DEL ANÁLISIS:")
        print("-" * 30)
        
        is_trendy = prediction.get('is_trendy', False)
        trend_confidence = prediction.get('trend_confidence', 0)
        trend_reason = prediction.get('trend_reason', '')
        category = prediction.get('category', '')
        category_confidence = prediction.get('category_confidence', 0)
        
        # Mostrar resultado principal
        if is_trendy:
            print("✅ ¡SÍ ESTÁ DE MODA!")
        else:
            print("❌ NO ESTÁ DE MODA")
        
        print(f"📈 Confianza de tendencia: {trend_confidence:.1%}")
        print(f"🏷️  Categoría: {category} ({category_confidence:.1%})")
        print(f"💭 Razón: {trend_reason}")
        
        print("\n🎉 Prueba completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
    
    finally:
        # Limpiar archivo de prueba
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🗑️  Archivo de prueba eliminado")

if __name__ == "__main__":
    # Esperar a que la aplicación se inicie
    print("⏳ Esperando que la aplicación se inicie...")
    time.sleep(3)
    test_fashion_trend()
