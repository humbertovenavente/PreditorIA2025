#!/usr/bin/env python3
"""
Script para probar el análisis de clustering con diferentes tipos de imágenes
"""

import requests
import json
import time
import os
from PIL import Image, ImageDraw
import numpy as np

def create_test_image(category, size=(224, 224), color=(100, 150, 200)):
    """Crear imagen de prueba para una categoría específica"""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Dibujar forma básica según la categoría
    if category == 'general':
        # Rectángulo simple
        draw.rectangle([50, 50, size[0]-50, size[1]-50], outline=(255, 255, 255), width=3)
    elif category == 'tops':
        # Forma de camiseta
        draw.ellipse([60, 40, size[0]-60, 80], outline=(255, 255, 255), width=3)
        draw.rectangle([70, 80, size[0]-70, size[1]-40], outline=(255, 255, 255), width=3)
    elif category == 'dress':
        # Forma de vestido
        draw.ellipse([60, 30, size[0]-60, 70], outline=(255, 255, 255), width=3)
        draw.rectangle([50, 70, size[0]-50, size[1]-20], outline=(255, 255, 255), width=3)
    elif category == 'jewelry':
        # Forma de joya
        draw.ellipse([80, 80, size[0]-80, size[1]-80], outline=(255, 255, 255), width=3)
    else:
        # Forma genérica
        draw.rectangle([40, 40, size[0]-40, size[1]-40], outline=(255, 255, 255), width=3)
    
    # Agregar texto
    draw.text((10, 10), category, fill=(255, 255, 255))
    
    return img

def test_clustering_analysis():
    """Probar análisis de clustering con diferentes imágenes"""
    base_url = "http://localhost:5000"
    
    print("🔍 PROBANDO ANÁLISIS DE CLUSTERING")
    print("="*50)
    
    # Categorías a probar
    test_categories = ['general', 'tops', 'dress', 'jewelry', 'scarves']
    
    results = []
    
    for i, category in enumerate(test_categories):
        print(f"\n📸 Probando categoría: {category}")
        
        # Crear imagen de prueba
        img = create_test_image(category, color=(100 + i*30, 150 + i*20, 200 - i*25))
        img_path = f"test_image_{category}_{i}.jpg"
        img.save(img_path)
        
        try:
            # Subir imagen
            with open(img_path, 'rb') as f:
                files = {'file': (img_path, f, 'image/jpeg')}
                response = requests.post(f"{base_url}/upload", files=files)
            
            if response.status_code == 200:
                upload_data = response.json()
                filepath = upload_data['filepath']
                
                # Analizar tendencia
                predict_data = {'filepath': filepath}
                response = requests.post(f"{base_url}/predict", 
                                       json=predict_data,
                                       headers={'Content-Type': 'application/json'})
                
                if response.status_code == 200:
                    prediction = response.json()['prediction']
                    
                    result = {
                        'category': category,
                        'is_trendy': prediction['is_trendy'],
                        'trend_confidence': prediction['trend_confidence'],
                        'trend_reason': prediction['trend_reason'],
                        'model_category': prediction['category'],
                        'model_confidence': prediction['category_confidence']
                    }
                    results.append(result)
                    
                    # Mostrar resultado
                    trend_status = "✅ TENDENCIA" if prediction['is_trendy'] else "❌ NO TENDENCIA"
                    print(f"   {trend_status}")
                    print(f"   Confianza: {prediction['trend_confidence']:.1%}")
                    print(f"   Razón: {prediction['trend_reason']}")
                    print(f"   Categoría modelo: {prediction['category']} ({prediction['category_confidence']:.1%})")
                else:
                    print(f"   ❌ Error en predicción: {response.status_code}")
            else:
                print(f"   ❌ Error en upload: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        finally:
            # Limpiar archivo
            if os.path.exists(img_path):
                os.remove(img_path)
    
    # Resumen de resultados
    print("\n" + "="*50)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*50)
    
    trendy_count = sum(1 for r in results if r['is_trendy'])
    total_count = len(results)
    
    print(f"Total de pruebas: {total_count}")
    print(f"Clasificadas como tendencia: {trendy_count}")
    print(f"Clasificadas como no tendencia: {total_count - trendy_count}")
    
    print(f"\n📋 Detalles por categoría:")
    for result in results:
        trend_icon = "✅" if result['is_trendy'] else "❌"
        print(f"   {trend_icon} {result['category']}: {result['trend_confidence']:.1%} confianza")
        print(f"      Razón: {result['trend_reason']}")
    
    return results

if __name__ == "__main__":
    print("⏳ Esperando que la aplicación se inicie...")
    time.sleep(3)  # Esperar un poco para que la app se inicie
    
    try:
        results = test_clustering_analysis()
        print(f"\n🎉 Prueba completada exitosamente!")
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
