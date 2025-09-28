#!/usr/bin/env python3
"""Script para debuggear el preprocesamiento"""

import sys
sys.path.append('/home/jose/PreditorIA2025/fashion_trend_app')

from app_with_progress import preprocess_image_advanced
from PIL import Image
import numpy as np

# Probar con la imagen roja
image_path = '/home/jose/PreditorIA2025/test_red.jpg'
print(f"Imagen original: {image_path}")

# Cargar imagen original
original = Image.open(image_path)
print(f"Tamaño original: {original.size}")
print(f"Modo: {original.mode}")

# Mostrar algunos píxeles originales
original_data = np.array(original)
print(f"Primeros 5 píxeles originales: {original_data[0, :5]}")

# Aplicar preprocesamiento
processed = preprocess_image_advanced(image_path)
print(f"Tamaño procesado: {processed.size}")
print(f"Modo procesado: {processed.mode}")

# Mostrar algunos píxeles procesados
processed_data = np.array(processed)
print(f"Primeros 5 píxeles procesados: {processed_data[0, :5]}")

# Verificar si hay píxeles negros
black_pixels = np.sum((processed_data == [0, 0, 0]).all(axis=2))
total_pixels = processed_data.shape[0] * processed_data.shape[1]
print(f"Píxeles negros: {black_pixels} de {total_pixels} ({black_pixels/total_pixels*100:.1f}%)")

# Guardar imagen procesada para inspección
processed.save('/home/jose/PreditorIA2025/test_red_processed.jpg')
print("Imagen procesada guardada: test_red_processed.jpg")



