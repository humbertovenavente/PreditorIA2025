#  REPLICACIÓN RÁPIDA - PreditorIA2025


---

## ⚡ Instalación Automática (Recomendada)

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/PreditorIA2025.git
cd PreditorIA2025

# 2. Ejecutar instalación automática
./instalar_proyecto.sh

# 3. Verificar instalación
./verificar_instalacion.sh
```

---

## Instalación Manual

### Paso 1: Preparar entorno
```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r fashion_trend_app/requirements.txt
```

### Paso 2: Configurar proyecto
```bash
# Crear directorios
mkdir -p data/{images,processed/{train,validation,test},logs,embeddings,reduced,clustering_results,analysis}
mkdir -p clustering_results credentials

# Copiar configuración
cp env.example .env
```

### Paso 3: Descargar modelo
```bash
# Descargar modelo pre-entrenado
python -c "
from tensorflow.keras.applications import MobileNetV2
import os
os.makedirs('data/logs/training', exist_ok=True)
model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model.save('data/logs/training/mobilenet_v2_pretrained.h5')
print('Modelo descargado')
"
```

---

## 🎯 Ejecución Rápida

### Opción A: Pipeline Completo
```bash
# Activar entorno
source venv/bin/activate

# Ejecutar todo el pipeline
cd fashion_clustering
python extract_embeddings.py
python reduce_dim.py
python run_clustering.py
python summarize_clusters.py
```

### Opción B: Aplicación Web
```bash
# Activar entorno
source venv/bin/activate

# Iniciar aplicación web
cd fashion_trend_app
python app_with_progress.py

# Abrir en navegador: http://localhost:5000
```

### Opción C: Con Docker
```bash
# Construir y ejecutar
docker-compose build
docker-compose up
```

---

## 📊 Verificar Resultados

```bash
# Verificar archivos generados
ls -la data/clustering_results/
ls -la clustering_results/

# Verificar métricas
python -c "
import pickle
with open('data/clustering_results/kmeans_clusters.pkl', 'rb') as f:
    results = pickle.load(f)
print(f'Clusters: {results[\"n_clusters\"]}')
print(f'Silhouette: {results[\"silhouette_score\"]:.3f}')
"
```

---

## 🆘 Solución Rápida de Problemas

### Error de memoria
```bash
# Reducir batch size en fashion_clustering/config.yaml
data:
  batch_size: 16  # Cambiar de 32 a 16
```

### Error de dependencias
```bash
# Reinstalar todo
pip install --upgrade -r requirements.txt
```

### Error de permisos
```bash
# Dar permisos
chmod -R 755 data/
chmod +x *.sh
```

---

## 📚 Documentación Completa

Para información detallada, consulta:
- **`ANEXO_A_REPLICACION_PROYECTO.md`** - Guía completa de replicación
- **`README.md`** - Documentación principal del proyecto

---

## 🎉 ¡Listo!

Si todo funciona correctamente, deberías ver:
- ✅ Archivos de clustering generados
- ✅ Visualizaciones UMAP creadas
- ✅ Aplicación web funcionando en http://localhost:5000
- ✅ Métricas de calidad del clustering

**Tiempo estimado**: 5-10 minutos (dependiendo de la velocidad de descarga)
