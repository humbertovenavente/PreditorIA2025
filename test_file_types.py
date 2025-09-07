#!/usr/bin/env python3
"""
Script para probar diferentes tipos de archivos de imagen
"""

import requests
import os
from PIL import Image
import numpy as np

def create_test_images():
    """Crea imágenes de prueba en diferentes formatos"""
    test_images = {}
    
    # Crear imagen base
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Guardar en diferentes formatos
    formats = {
        'test.png': 'PNG',
        'test.jpg': 'JPEG', 
        'test.jpeg': 'JPEG',
        'test.gif': 'GIF',
        'test.bmp': 'BMP',
        'test.tiff': 'TIFF',
        'test.webp': 'WEBP'
    }
    
    for filename, format_type in formats.items():
        try:
            img.save(filename, format=format_type)
            test_images[filename] = format_type
            print(f"✅ Creada imagen de prueba: {filename} ({format_type})")
        except Exception as e:
            print(f"❌ Error creando {filename}: {e}")
    
    return test_images

def test_file_upload(filename, format_type):
    """Prueba la subida de un archivo específico"""
    print(f"\n🧪 Probando {filename} ({format_type})...")
    
    try:
        with open(filename, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/upload', files=files)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ {filename} subido exitosamente")
                
                # Probar análisis
                predict_data = {'filepath': result.get('filepath')}
                response = requests.post(
                    'http://localhost:5000/predict',
                    json=predict_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('prediction'):
                        pred = result['prediction']
                        print(f"   📊 Categoría: {pred.get('class_name')}")
                        print(f"   📊 Confianza: {pred.get('confidence', 0):.3f}")
                        print(f"   📊 Imágenes similares: {len(result.get('similar_images', []))}")
                    else:
                        print(f"   ❌ Error en predicción: {result.get('error')}")
                else:
                    print(f"   ❌ Error en análisis: {response.status_code}")
            else:
                print(f"❌ Error subiendo {filename}: {result.get('error')}")
        else:
            print(f"❌ Error HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error probando {filename}: {e}")

def cleanup_test_files(filenames):
    """Limpia los archivos de prueba"""
    for filename in filenames:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"🗑️  Eliminado: {filename}")
        except Exception as e:
            print(f"⚠️  Error eliminando {filename}: {e}")

def main():
    print("🎯 Probando diferentes tipos de archivos de imagen")
    print("="*50)
    
    # Crear imágenes de prueba
    test_images = create_test_images()
    
    if not test_images:
        print("❌ No se pudieron crear imágenes de prueba")
        return
    
    print(f"\n📁 Creadas {len(test_images)} imágenes de prueba")
    
    # Probar cada tipo de archivo
    for filename, format_type in test_images.items():
        test_file_upload(filename, format_type)
    
    # Limpiar archivos de prueba
    print(f"\n🧹 Limpiando archivos de prueba...")
    cleanup_test_files(test_images.keys())
    
    print("\n🎉 Pruebas de tipos de archivo completadas!")

if __name__ == "__main__":
    main()
