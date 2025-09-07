#!/usr/bin/env python3
"""
Script de ejemplo para demostrar el análisis de clustering de imágenes de moda
"""

import os
import sys
import logging
from pathlib import Path

# Añadir el directorio actual al path
sys.path.append(str(Path(__file__).parent))

from analysis.fashion_analysis import extract_and_cluster_images

def main():
    """Función principal de ejemplo"""
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("EJEMPLO DE ANÁLISIS DE CLUSTERING DE IMÁGENES DE MODA")
    print("="*60)
    print()
    
    # Verificar que existe la base de datos
    if not os.path.exists('fashion_images.db'):
        print("❌ Error: No se encontró la base de datos 'fashion_images.db'")
        print("   Ejecuta primero el scraper para recolectar imágenes:")
        print("   python main.py --platform all --images 100")
        return
    
    print("✅ Base de datos encontrada")
    print()
    
    # Configuración del análisis
    config = {
        'model_name': 'resnet50',  # Modelo CNN a usar
        'n_clusters': 8,           # Número inicial de clusters
        'max_images': 500,         # Máximo de imágenes a procesar
        'use_pca': True,           # Usar PCA para reducción de dimensionalidad
        'pca_components': 50,      # Número de componentes PCA
        'find_optimal_k': True,    # Buscar automáticamente el K óptimo
        'max_optimal_k': 15        # Máximo K para búsqueda óptima
    }
    
    print("Configuración del análisis:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Ejecutar análisis
        print("🚀 Iniciando análisis de clustering...")
        print()
        
        extract_and_cluster_images(**config)
        
        print()
        print("✅ Análisis completado exitosamente!")
        print()
        print("Archivos generados:")
        print("  - fashion_features_resnet50.pkl (características extraídas)")
        print("  - clustering_results_resnet50.pkl (resultados del clustering)")
        print("  - cluster_visualizations/ (gráficos y visualizaciones)")
        print("  - fashion_analysis_report.txt (reporte detallado)")
        print()
        print("Para ver los resultados:")
        print("  1. Revisa el archivo 'fashion_analysis_report.txt'")
        print("  2. Mira las visualizaciones en 'cluster_visualizations/'")
        print("  3. Usa los archivos .pkl para análisis adicionales")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        print()
        print("Posibles soluciones:")
        print("  1. Verifica que tienes suficientes imágenes en la base de datos")
        print("  2. Instala las dependencias: pip install -r requirements.txt")
        print("  3. Verifica que tienes espacio suficiente en disco")
        print("  4. Revisa los logs para más detalles")

if __name__ == "__main__":
    main()

