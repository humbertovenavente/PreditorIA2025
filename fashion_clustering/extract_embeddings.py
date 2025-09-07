#!/usr/bin/env python3
"""
Script CLI para extraer embeddings con MobileNetV2
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
from typing import Optional

# Añadir el directorio padre al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from fashion_clustering.utils.data import load_dataset, create_metadata
from fashion_clustering.utils.vision import load_mobilenetv2, extract_embeddings_batch
from fashion_clustering.utils.io import save_embeddings, check_existing_files

def setup_logging(log_file: str = "reports/run.log", level: str = "INFO"):
    """Configura logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error cargando configuración: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Extraer embeddings con MobileNetV2')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Ruta raíz del dataset')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                       help='Ruta al checkpoint afinado (opcional)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Tamaño del lote')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Número de workers')
    parser.add_argument('--resume', action='store_true',
                       help='Reanudar si existen archivos')
    parser.add_argument('--config', type=str, default='fashion_clustering/config.yaml',
                       help='Archivo de configuración')
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Directorio de salida')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Splits a procesar')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Configurar logging
    log_file = config.get('logging', {}).get('file', 'reports/run.log')
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_file, log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Iniciando extracción de embeddings")
    
    # Verificar archivos existentes
    if args.resume:
        existing_files = check_existing_files(args.output_dir)
        if existing_files.get('embeddings.npy', False) and existing_files.get('metadata.csv', False):
            logger.info("✅ Archivos de embeddings ya existen. Usando --resume")
            logger.info("Para reprocesar, elimine los archivos existentes o no use --resume")
            return
    
    try:
        # Cargar dataset
        logger.info(f"📁 Cargando dataset desde {args.data_root}")
        image_paths, split_labels, category_labels = load_dataset(
            args.data_root, 
            splits=args.splits
        )
        
        if not image_paths:
            logger.error("❌ No se encontraron imágenes en el dataset")
            return
        
        logger.info(f"✅ Cargadas {len(image_paths)} imágenes")
        
        # Crear metadatos
        logger.info("📊 Creando metadatos")
        metadata = create_metadata(image_paths, split_labels, category_labels)
        
        # Cargar modelo
        logger.info("🤖 Cargando modelo MobileNetV2")
        model_config = config.get('model', {})
        model = load_mobilenetv2(
            checkpoint_path=args.model_checkpoint or model_config.get('checkpoint'),
            freeze_layers=model_config.get('freeze_layers', True),
            embedding_layer=model_config.get('embedding_layer', -2)
        )
        
        # Configurar parámetros de extracción
        data_config = config.get('data', {})
        batch_size = args.batch_size or data_config.get('batch_size', 128)
        image_size = tuple(data_config.get('image_size', [224, 224]))
        normalize = data_config.get('normalize', True)
        
        logger.info(f"🔧 Configuración: batch_size={batch_size}, image_size={image_size}, normalize={normalize}")
        
        # Callback de progreso
        def progress_callback(progress):
            logger.info(f"📈 Progreso: {progress:.1%}")
        
        # Extraer embeddings
        logger.info("🔄 Extrayendo embeddings...")
        embeddings = extract_embeddings_batch(
            model=model,
            image_paths=image_paths,
            batch_size=batch_size,
            target_size=image_size,
            normalize=normalize,
            progress_callback=progress_callback
        )
        
        if embeddings.size == 0:
            logger.error("❌ No se pudieron extraer embeddings")
            return
        
        logger.info(f"✅ Embeddings extraídos: {embeddings.shape}")
        
        # Guardar resultados
        logger.info("💾 Guardando resultados")
        saved_files = save_embeddings(embeddings, metadata, args.output_dir)
        
        logger.info("🎉 Extracción de embeddings completada exitosamente")
        logger.info(f"📁 Archivos guardados:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  - {file_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Error en extracción de embeddings: {e}")
        raise

if __name__ == "__main__":
    main()


