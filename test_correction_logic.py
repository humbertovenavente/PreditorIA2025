#!/usr/bin/env python3
"""
Script para probar la lógica de corrección específicamente
"""

import requests
import json
import time
from PIL import Image, ImageDraw
import os

def create_test_tshirt():
    """Crear una imagen de prueba de t-shirt"""
    # Crear imagen simple de camiseta
    img = Image.new('RGB', (224, 224), (100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Dibujar forma de camiseta
    draw.ellipse([60, 40, 224-60, 80], outline=(255, 255, 255), width=3)
    draw.rectangle([70, 80, 224-70, 224-40], outline=(255, 255, 255), width=3)
    draw.text((10, 10), "T-SHIRT", fill=(255, 255, 255))
    
    # Guardar imagen de prueba
    test_path = "test_tshirt_correction.jpg"
    img.save(test_path)
    return test_path

def test_correction_logic():
    """Probar la lógica de corrección"""
    print("🧪 PROBANDO LÓGICA DE CORRECCIÓN")
    print("="*50)
    
    # Crear imagen de prueba
    test_path = create_test_tshirt()
    
    try:
        # Subir imagen
        with open(test_path, 'rb') as f:
            files = {'file': ('test_tshirt.jpg', f, 'image/jpeg')}
            response = requests.post('http://127.0.0.1:5000/upload', files=files)
        
        if response.status_code != 200:
            print(f"❌ Error subiendo imagen: {response.status_code}")
            return
        
        upload_data = response.json()
        if not upload_data.get('success'):
            print(f"❌ Error en upload: {upload_data.get('error')}")
            return
        
        print("✅ Imagen subida exitosamente")
        
        # Hacer predicción
        predict_data = {'filename': upload_data['filename']}
        response = requests.post('http://127.0.0.1:5000/predict', json=predict_data)
        
        if response.status_code != 200:
            print(f"❌ Error en predicción: {response.status_code}")
            return
        
        result = response.json()
        
        print(f"\n📊 RESULTADOS DETALLADOS:")
        print(f"   Predicción original: {result.get('original_prediction', 'N/A')}")
        print(f"   Categoría corregida: {result.get('category', 'N/A')}")
        print(f"   Confianza: {result.get('category_confidence', 0):.1%}")
        print(f"   Corrección aplicada: {result.get('correction_applied', False)}")
        print(f"   Razón: {result.get('trend_reason', 'N/A')}")
        
        # Verificar si la corrección funcionó
        if result.get('category') == 'tops' and result.get('original_prediction') == 'scarves':
            print("✅ ¡Corrección exitosa! T-shirt corregida de 'scarves' a 'tops'")
        elif result.get('category') == 'tops':
            print("✅ ¡Clasificación correcta! T-shirt clasificada como 'tops'")
        else:
            print(f"❌ Clasificación incorrecta: {result.get('category')}")
            
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
    
    finally:
        # Limpiar
        if os.path.exists(test_path):
            os.remove(test_path)

def test_multiple_images():
    """Probar con múltiples tipos de imágenes"""
    print(f"\n🔄 PROBANDO MÚLTIPLES TIPOS DE IMÁGENES")
    print("="*50)
    
    test_cases = [
        ("T-shirt", (100, 150, 200), "tops"),
        ("Dress", (200, 100, 150), "dress"),
        ("Shoes", (150, 200, 100), "shoes"),
        ("Bag", (200, 200, 100), "bags")
    ]
    
    for name, color, expected in test_cases:
        print(f"\n🧪 Probando {name}...")
        
        # Crear imagen de prueba
        img = Image.new('RGB', (224, 224), color)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), name.upper(), fill=(255, 255, 255))
        
        test_path = f"test_{name.lower()}.jpg"
        img.save(test_path)
        
        try:
            # Subir y predecir
            with open(test_path, 'rb') as f:
                files = {'file': (f'test_{name.lower()}.jpg', f, 'image/jpeg')}
                response = requests.post('http://127.0.0.1:5000/upload', files=files)
            
            if response.status_code == 200:
                upload_data = response.json()
                if upload_data.get('success'):
                    predict_data = {'filename': upload_data['filename']}
                    response = requests.post('http://127.0.0.1:5000/predict', json=predict_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        category = result.get('category', 'unknown')
                        confidence = result.get('category_confidence', 0)
                        correction = result.get('correction_applied', False)
                        
                        print(f"   Resultado: {category} ({confidence:.1%})")
                        if correction:
                            print(f"   Corrección aplicada: {result.get('original_prediction')} → {category}")
                        
                        if category == expected:
                            print(f"   ✅ Correcto")
                        else:
                            print(f"   ❌ Esperado: {expected}")
                    else:
                        print(f"   ❌ Error en predicción: {response.status_code}")
                else:
                    print(f"   ❌ Error en upload: {upload_data.get('error')}")
            else:
                print(f"   ❌ Error subiendo: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

def main():
    """Función principal"""
    print("🔧 PROBANDO LÓGICA DE CORRECCIÓN MEJORADA")
    print("="*60)
    
    # Esperar un momento para que la aplicación esté lista
    print("⏳ Esperando que la aplicación esté lista...")
    time.sleep(2)
    
    # Probar corrección específica
    test_correction_logic()
    
    # Probar múltiples tipos
    test_multiple_images()
    
    print(f"\n✅ Pruebas completadas!")

if __name__ == "__main__":
    main()
