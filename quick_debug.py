#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

# Cargar modelo
print("🔍 Cargando modelo...")
model = tf.keras.models.load_model('/home/jose/PreditorIA2025/data/logs/training/mobilenet_v2_final.h5')
print(f"✅ Modelo cargado - Output: {model.output_shape}")

# Cargar clustering
print("🔍 Cargando clustering...")
with open('/home/jose/PreditorIA2025/clustering_results/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
print(f"✅ K-means cargado - Clusters: {kmeans.n_clusters}")

# Procesar imagen
print("🔍 Procesando imagen...")
img = Image.open('/home/jose/PreditorIA2025/homeImagen/imagen1.jpeg')
img = img.convert('RGB').resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Predicción del modelo
print("🔍 Predicción del modelo...")
prediction = model.predict(img_batch, verbose=0)[0]
predicted_class = np.argmax(prediction)
model_confidence = float(np.max(prediction))

print(f"📊 Predicción completa: {prediction}")
print(f"🎯 Clase predicha: {predicted_class}")
print(f"🎯 Confianza del modelo: {model_confidence:.4f} ({model_confidence*100:.1f}%)")

# Clustering
print("🔍 Clustering...")
embedding = prediction.astype(np.float64)
predicted_cluster = kmeans.predict([embedding])[0]
cluster_center = kmeans.cluster_centers_[predicted_cluster]
distance = np.linalg.norm(embedding - cluster_center)
similarity_score = max(0, 100 - (distance / 10.0) * 100)

print(f"🎯 Cluster predicho: {predicted_cluster}")
print(f"🎯 Distancia al centro: {distance:.4f}")
print(f"🎯 Similitud: {similarity_score:.1f}%")

# Confianza combinada
model_conf_100 = model_confidence * 100
combined_confidence = (model_conf_100 * 0.6 + similarity_score * 0.4) / 100
print(f"🎯 Confianza combinada: {combined_confidence:.4f} ({combined_confidence*100:.1f}%)")

# Verificar rangos
print(f"🔍 Clase en rango 0-10: {0 <= predicted_class <= 10}")
print(f"🔍 Cluster en rango 0-149: {0 <= predicted_cluster < kmeans.n_clusters}")

print("\n" + "="*50)
print("RESUMEN:")
print(f"Clase: {predicted_class}")
print(f"Cluster: {predicted_cluster}")
print(f"Confianza modelo: {model_confidence*100:.1f}%")
print(f"Confianza clustering: {similarity_score:.1f}%")
print(f"Confianza combinada: {combined_confidence*100:.1f}%")
print("="*50)


