#!/usr/bin/env python3
"""
Script de inicio para la aplicación web PreditorIA2025
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Verifica que todos los requisitos estén disponibles"""
    print("🔍 Verificando requisitos...")
    
    # Verificar modelo entrenado
    model_path = "data/logs/training/mobilenet_v2_final.h5"
    if not os.path.exists(model_path):
        print("❌ Modelo entrenado no encontrado en:", model_path)
        print("   Ejecuta primero el entrenamiento del modelo")
        return False
    print("✅ Modelo entrenado encontrado")
    
    # Verificar resultados de clustering
    clustering_path = "clustering_results/clustering_results.pkl"
    if not os.path.exists(clustering_path):
        print("⚠️  Resultados de clustering no encontrados en:", clustering_path)
        print("   Ejecuta primero el script de clustering")
        print("   La aplicación funcionará sin clustering")
    else:
        print("✅ Resultados de clustering encontrados")
    
    # Verificar dependencias de Python
    try:
        import flask
        import tensorflow as tf
        import numpy as np
        import PIL
        print("✅ Dependencias de Python verificadas")
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Crea los directorios necesarios"""
    print("📁 Creando directorios necesarios...")
    
    directories = [
        "static/uploads",
        "static/results",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def start_application():
    """Inicia la aplicación Flask"""
    print("\n🚀 Iniciando aplicación PreditorIA2025...")
    print("="*50)
    
    try:
        # Ejecutar la aplicación
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n⏹️  Aplicación detenida por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando la aplicación: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return False
    
    return True

def open_browser():
    """Abre el navegador web"""
    print("🌐 Abriendo navegador web...")
    time.sleep(2)  # Esperar a que la aplicación se inicie
    
    try:
        webbrowser.open("http://localhost:5000")
        print("✅ Navegador abierto en http://localhost:5000")
    except Exception as e:
        print(f"⚠️  No se pudo abrir el navegador automáticamente: {e}")
        print("   Abre manualmente: http://localhost:5000")

def show_instructions():
    """Muestra instrucciones de uso"""
    print("\n" + "="*60)
    print("🎯 PREDITORIA2025 - DASHBOARD WEB")
    print("="*60)
    print()
    print("📋 INSTRUCCIONES DE USO:")
    print("1. La aplicación se ejecutará en: http://localhost:5000")
    print("2. Arrastra y suelta una imagen en el área de carga")
    print("3. Haz clic en 'Analizar Imagen' para obtener predicciones")
    print("4. Revisa la sección 'Clustering' para ver estadísticas")
    print()
    print("🔧 FUNCIONALIDADES:")
    print("• Análisis de categorías de moda en tiempo real")
    print("• Búsqueda de imágenes similares")
    print("• Visualización de clusters y métricas")
    print("• Interfaz web moderna y responsiva")
    print()
    print("⏹️  Para detener la aplicación: Ctrl+C")
    print("="*60)

def main():
    """Función principal"""
    print("🎯 PreditorIA2025 - Dashboard Web")
    print("Sistema de Análisis de Moda con IA")
    print("="*40)
    
    # Verificar requisitos
    if not check_requirements():
        print("\n❌ No se pueden cumplir todos los requisitos")
        print("   Revisa los errores anteriores y vuelve a intentar")
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Mostrar instrucciones
    show_instructions()
    
    # Preguntar si abrir navegador
    try:
        response = input("\n¿Abrir navegador automáticamente? (s/n): ").lower().strip()
        if response in ['s', 'si', 'sí', 'y', 'yes']:
            open_browser()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
        sys.exit(0)
    
    # Iniciar aplicación
    print("\n🚀 Iniciando servidor web...")
    start_application()

if __name__ == "__main__":
    main()
