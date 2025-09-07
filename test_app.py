#!/usr/bin/env python3
"""
Script de prueba para verificar que la aplicación funciona correctamente
"""

import requests
import time
import json

def test_app():
    """Probar la aplicación completa"""
    base_url = "http://localhost:5000"
    
    print("🧪 Probando Fashion Trend App...")
    
    # 1. Verificar estado de carga
    print("\n1. Verificando estado de carga...")
    try:
        response = requests.get(f"{base_url}/api/loading_status")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Modelos cargados: {data['models_loaded']}")
            print(f"   📊 Progreso: {data['progress']}%")
            print(f"   📝 Estado: {data['status']}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Error conectando: {e}")
        return
    
    # 2. Verificar estadísticas
    print("\n2. Verificando estadísticas...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Total clusters: {data['total_clusters']}")
            print(f"   ✅ Total imágenes: {data['total_images']}")
            print(f"   ✅ Algoritmo: {data['algorithm']}")
            print(f"   ✅ Silhouette Score: {data['silhouette_score']:.4f}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Probar análisis de imagen
    print("\n3. Probando análisis de imagen...")
    try:
        # Usar una imagen de prueba
        image_path = "/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/api/analyze", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Análisis iniciado: {data['message']}")
            print(f"   📁 Archivo: {data['filename']}")
            
            # Esperar a que termine el análisis
            print("   ⏳ Esperando análisis...")
            for i in range(10):  # Esperar hasta 10 segundos
                time.sleep(1)
                
                progress_response = requests.get(f"{base_url}/api/analysis_progress")
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    print(f"   📊 Progreso: {progress_data['progress']}% - {progress_data['status']}")
                    
                    if not progress_data['analysis_active'] and progress_data['progress'] == 0:
                        break
            
            # Obtener resultado
            result_response = requests.get(f"{base_url}/api/analysis_result")
            if result_response.status_code == 200:
                result_data = result_response.json()
                if result_data['success']:
                    result = result_data['result']
                    print(f"   ✅ Análisis completado!")
                    print(f"   📊 Trend Score: {result['trend_score']}")
                    print(f"   🔥 En tendencia: {result['is_trending']}")
                    print(f"   🎯 Cluster ID: {result['cluster_id']}")
                    print(f"   📈 Similitud: {result['similarity_score']:.1f}%")
                    print(f"   🎨 Colores: {result['colors']}")
                    print(f"   📝 Interpretación: {result['interpretation']['trend_analysis']}")
                else:
                    print(f"   ❌ Error en análisis: {result_data['message']}")
            else:
                print(f"   ❌ Error obteniendo resultado: {result_response.status_code}")
        else:
            print(f"   ❌ Error iniciando análisis: {response.status_code}")
            print(f"   📝 Respuesta: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 Prueba completada!")

if __name__ == "__main__":
    test_app()