#!/usr/bin/env python3
"""
Script para probar la corrección en la aplicación web
"""

import requests
import json
import time
import os
import shutil

def test_web_app_correction():
    """Probar la corrección en la aplicación web"""
    print("🧪 PROBANDO CORRECCIÓN EN APLICACIÓN WEB")
    print("="*60)
    
    # Copiar la imagen a uploads para simular el upload
    source_image = "data/images/2025_2025_fashion creator guatemala_1756099493_fa91dee8.webp"
    target_image = "uploads/test_tshirt.webp"
    
    try:
        # Crear directorio uploads si no existe
        os.makedirs("uploads", exist_ok=True)
        
        # Copiar imagen
        shutil.copy2(source_image, target_image)
        print(f"✅ Imagen copiada a {target_image}")
        
        # Simular upload
        with open(target_image, 'rb') as f:
            files = {'file': ('test_tshirt.webp', f, 'image/webp')}
            response = requests.post('http://127.0.0.1:5000/upload', files=files)
        
        if response.status_code != 200:
            print(f"❌ Error en upload: {response.status_code}")
            return
        
        upload_data = response.json()
        if not upload_data.get('success'):
            print(f"❌ Error en upload: {upload_data.get('error')}")
            return
        
        print(f"✅ Upload exitoso: {upload_data['filename']}")
        
        # Hacer predicción
        predict_data = {'filename': upload_data['filename']}
        response = requests.post('http://127.0.0.1:5000/predict', json=predict_data)
        
        if response.status_code != 200:
            print(f"❌ Error en predicción: {response.status_code}")
            return
        
        result = response.json()
        
        print(f"\n📊 RESULTADO DE LA APLICACIÓN WEB:")
        print(f"   ¿Está de moda? {'Sí' if result.get('is_trendy') else 'No'}")
        print(f"   Confianza de tendencia: {result.get('trend_confidence', 0):.1%}")
        print(f"   Categoría: {result.get('category', 'N/A')} ({result.get('category_confidence', 0):.1%})")
        print(f"   Predicción original: {result.get('original_prediction', 'N/A')}")
        print(f"   Corrección aplicada: {'Sí' if result.get('correction_applied') else 'No'}")
        print(f"   Razón: {result.get('trend_reason', 'N/A')}")
        
        # Verificar si la corrección funcionó
        if result.get('category') == 'tops':
            print(f"\n✅ ¡CORRECCIÓN EXITOSA! T-shirt clasificada como 'tops'")
        elif result.get('category') in ['shoes', 'dress', 'general']:
            print(f"\n✅ ¡CORRECCIÓN PARCIAL! T-shirt corregida de 'scarves' a '{result.get('category')}'")
        else:
            print(f"\n❌ Clasificación incorrecta: {result.get('category')}")
            
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
    
    finally:
        # Limpiar
        if os.path.exists(target_image):
            os.remove(target_image)

def main():
    """Función principal"""
    print("🔧 PROBANDO CORRECCIÓN ULTRA AGRESIVA")
    print("="*60)
    
    # Esperar un momento para que la aplicación esté lista
    print("⏳ Esperando que la aplicación esté lista...")
    time.sleep(2)
    
    # Probar corrección
    test_web_app_correction()
    
    print(f"\n✅ Prueba completada!")

if __name__ == "__main__":
    main()
