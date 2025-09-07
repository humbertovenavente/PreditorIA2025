#!/usr/bin/env python3
"""
Ejemplo de uso del módulo Fashion Clustering
"""

import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Ejemplo completo de uso del módulo Fashion Clustering"""
    
    logger.info("🚀 Iniciando ejemplo de Fashion Clustering")
    
    # Verificar que el dataset existe
    data_root = "data/processed"
    if not os.path.exists(data_root):
        logger.error(f"❌ Dataset no encontrado en {data_root}")
        logger.info("Por favor, asegúrate de que el dataset esté en la ubicación correcta")
        return
    
    # Verificar que el modelo existe
    model_checkpoint = "data/logs/training/mobilenet_v2_final.h5"
    if not os.path.exists(model_checkpoint):
        logger.warning(f"⚠️ Modelo afinado no encontrado en {model_checkpoint}")
        logger.info("Se usará MobileNetV2 preentrenada")
        model_checkpoint = None
    
    try:
        # 1. Extraer embeddings
        logger.info("📊 Paso 1: Extrayendo embeddings...")
        os.system(f"""
        python -m fashion_clustering.extract_embeddings \
          --data_root {data_root} \
          --model_checkpoint {model_checkpoint or ''} \
          --batch_size 64 \
          --num_workers 4 \
          --resume
        """)
        
        # 2. Reducir dimensionalidad
        logger.info("📊 Paso 2: Reduciendo dimensionalidad...")
        os.system("""
        python -m fashion_clustering.reduce_dim \
          --method both \
          --n_components 50 \
          --umap_components 2
        """)
        
        # 3. Ejecutar clustering
        logger.info("📊 Paso 3: Ejecutando clustering...")
        os.system("""
        python -m fashion_clustering.run_clustering \
          --algo auto \
          --k_min 4 \
          --k_max 20 \
          --random_state 42
        """)
        
        # 4. Generar resumen
        logger.info("📊 Paso 4: Generando resumen...")
        os.system("""
        python -m fashion_clustering.summarize_clusters \
          --top_prototypes 10 \
          --n_colors 5
        """)
        
        # 5. Crear visualizaciones
        logger.info("📊 Paso 5: Creando visualizaciones...")
        os.system("""
        python -m fashion_clustering.visualize --all
        """)
        
        logger.info("🎉 ¡Proceso completado exitosamente!")
        logger.info("📁 Revisa los archivos en el directorio 'reports/'")
        
        # Mostrar archivos generados
        reports_dir = Path("reports")
        if reports_dir.exists():
            logger.info("📋 Archivos generados:")
            for file_path in sorted(reports_dir.iterdir()):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    logger.info(f"  - {file_path.name} ({size:,} bytes)")
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso: {e}")
        return
    
    # Mostrar resumen de resultados
    logger.info("\n📊 RESUMEN DE RESULTADOS:")
    
    # Cargar estadísticas si existen
    stats_path = "reports/cluster_stats.json"
    if os.path.exists(stats_path):
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        logger.info(f"  Algoritmo: {stats.get('algorithm', 'N/A')}")
        logger.info(f"  Número de clusters: {stats.get('n_clusters', 'N/A')}")
        logger.info(f"  Total de imágenes: {stats.get('total_images', 'N/A')}")
        
        metrics = stats.get('metrics', {})
        logger.info(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.3f}")
        logger.info(f"  Davies-Bouldin Index: {metrics.get('davies_bouldin_score', 'N/A'):.3f}")
    
    # Mostrar resumen de clusters si existe
    summary_path = "reports/cluster_summary.md"
    if os.path.exists(summary_path):
        logger.info(f"  Resumen detallado: {summary_path}")
    
    logger.info("\n🎯 PRÓXIMOS PASOS:")
    logger.info("  1. Revisa las visualizaciones en reports/")
    logger.info("  2. Analiza el resumen en cluster_summary.md")
    logger.info("  3. Usa los resultados para tu análisis de moda guatemalteca")
    logger.info("  4. Personaliza los parámetros en config.yaml si es necesario")

if __name__ == "__main__":
    main()


