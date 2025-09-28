#!/usr/bin/env python3
"""Script para debuggear el clustering"""

import sys
sys.path.append('/home/jose/PreditorIA2025/fashion_trend_app')

from app_with_progress import preprocess_image_advanced
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Probar con la imagen roja
image_path = '/home/jose/PreditorIA2025/test_red.jpg'
print(f"Analizando imagen: {image_path}")

# Preprocesar imagen
image = preprocess_image_advanced(image_path)
data = np.array(image)
data = data.reshape(-1, 3)

print(f"Shape de datos: {data.shape}")
print(f"Primeros 5 píxeles: {data[:5]}")
print(f"Valores únicos en R: {np.unique(data[:, 0])}")
print(f"Valores únicos en G: {np.unique(data[:, 1])}")
print(f"Valores únicos en B: {np.unique(data[:, 2])}")

# Filtrar píxeles negros
black_mask = (data[:, 0] == 0) & (data[:, 1] == 0) & (data[:, 2] == 0)
filtered_data = data[~black_mask]
print(f"Píxeles después del filtrado: {len(filtered_data)}")

if len(filtered_data) < 100:
    filtered_data = data
    print("Usando todos los píxeles")

print(f"Píxeles para clustering: {len(filtered_data)}")
print(f"Primeros 5 píxeles filtrados: {filtered_data[:5]}")

# Normalizar datos
scaler = StandardScaler()
normalized_data = scaler.fit_transform(filtered_data)
print(f"Datos normalizados shape: {normalized_data.shape}")
print(f"Primeros 5 píxeles normalizados: {normalized_data[:5]}")

# Clustering simple
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=1000)
kmeans.fit(normalized_data)

# Obtener colores dominantes
colors = kmeans.cluster_centers_.astype(int)
print(f"Centros de clusters: {colors}")

# Desnormalizar para ver colores reales
colors_real = scaler.inverse_transform(kmeans.cluster_centers_)
print(f"Colores reales (desnormalizados): {colors_real.astype(int)}")

# Contar frecuencia
labels = kmeans.labels_
unique, counts = np.unique(labels, return_counts=True)
print(f"Frecuencias: {dict(zip(unique, counts))}")



