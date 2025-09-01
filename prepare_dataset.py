#!/usr/bin/env python3
"""
Script principal para preparar el dataset de moda guatemalteca
PASO 2: Normalización, Data Augmentation y División del Dataset
"""

import sys
import logging
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from data_preparation.dataset_preparer import DatasetPreparer

def main():
    """Función principal para preparar el dataset"""
    print(" PASO 2: PREPARACIÓN DEL DATASET DE MODA GUATEMALTECA")
    print("=" * 60)
    print(" PROCESO COMPLETO:")
    print("  • Normalización de imágenes a 224x224px (MobileNet V2)")
    print("  • Data Augmentation (3 variaciones por imagen)")
    print("  • División: 70% entrenamiento, 15% validación, 15% prueba")
    print("=" * 60)
    
    try:
        # Crear preparador del dataset
        print(" Inicializando preparador del dataset...")
        preparer = DatasetPreparer(target_size=(224, 224))
        
        # Ejecutar preparación completa
        print("Iniciando preparación completa del dataset...")
        success = preparer.run_complete_preparation()
        
        if success:
            print("\n ¡PREPARACIÓN DEL DATASET COMPLETADA EXITOSAMENTE!")
            print("=" * 60)
            
            # Obtener información final
            info = preparer.get_dataset_info()
            
            print(" ESTADÍSTICAS FINALES:")
            print(f"  • Imágenes originales: {info['statistics']['original_count']}")
            print(f"  • Imágenes normalizadas: {info['statistics']['processed_count']}")
            print(f"  • Imágenes aumentadas: {info['statistics']['augmented_count']}")
            print(f"  • Total para entrenamiento: {info['statistics']['train_count']}")
            print(f"  • Total para validación: {info['statistics']['val_count']}")
            print(f"  • Total para prueba: {info['statistics']['test_count']}")
            print(f" FORMATO: {info['target_size'][0]}x{info['target_size'][1]}px")
            
            print("\n DIRECTORIOS CREADOS:")
            print(f"  • Dataset procesado: {info['directories']['processed']}")
            print(f"  • Dataset aumentado: {info['directories']['augmented']}")
            print(f"  • Entrenamiento: {info['directories']['train']}")
            print(f"  • Validación: {info['directories']['validation']}")
            print(f"  • Prueba: {info['directories']['test']}")
            
            
            
        else:
            print("\n ERROR: La preparación del dataset falló")
            print("Revisa los logs para más detalles")
            return 1
            
    except Exception as e:
        print(f"\n ERROR CRÍTICO: {e}")
        logging.error(f"Error en la preparación del dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

