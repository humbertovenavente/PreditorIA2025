ANEXO 2: GUÍA DE REPLICACIÓN DEL PROYECTO

---

## ÍNDICE

1. [Descripción General del Proyecto](#1-descripción-general-del-proyecto)
2. [Requisitos del Sistema](#2-requisitos-del-sistema)
3. [Instalación y Configuración](#3-instalación-y-configuración)
4. [Estructura del Proyecto](#4-estructura-del-proyecto)
5. [Proceso de Replicación Paso a Paso](#5-proceso-de-replicación-paso-a-paso)
6. [Configuración de Entorno](#6-configuración-de-entorno)
7. [Ejecución del Sistema](#7-ejecución-del-sistema)
8. [Verificación de Resultados](#8-verificación-de-resultados)
9. [Solución de Problemas Comunes](#9-solución-de-problemas-comunes)
10. [Referencias y Dependencias](#10-referencias-y-dependencias)

---

## 1. DESCRIPCIÓN GENERAL DEL PROYECTO


### 1.2 Componentes Principales
- **Scraper de Imágenes**: Extracción automática de imágenes de moda de redes sociales y sitios web
- **Procesamiento de Datos**: Normalización, filtrado de calidad y preparación del dataset
- **Modelo de Deep Learning**: MobileNetV2 pre-entrenado para extracción de características
- **Sistema de Clustering**: K-Means con reducción de dimensionalidad (PCA + UMAP)
- **Aplicación Web**: Interfaz Flask para análisis interactivo de tendencias
- **Visualización**: Generación automática de gráficos y reportes

### 1.3 Tecnologías Utilizadas
- **Python 3.11+**
- **TensorFlow 2.16+** (Deep Learning)
- **scikit-learn 1.3+** (Machine Learning)
- **OpenCV 4.8+** (Procesamiento de Imágenes)
- **Flask 2.3+** (Aplicación Web)
- **Docker** (Contenedorización)
- **Google Cloud Storage** (Almacenamiento en la nube)

---

## 2. REQUISITOS DEL SISTEMA

### 2.1 Requisitos Mínimos
- **Sistema Operativo**: Linux (Ubuntu 20.04+), macOS (10.15+), o Windows 10+
- **RAM**: 8 GB mínimo, 16 GB recomendado
- **Almacenamiento**: 50 GB de espacio libre
- **CPU**: 4 cores mínimo, 8 cores recomendado
- **GPU**: Opcional pero recomendada (NVIDIA con CUDA)

### 2.2 Software Requerido
- **Python**: 3.11 o superior
- **Docker**: 20.10+ (opcional pero recomendado)
- **Git**: Para clonar el repositorio
- **Chrome/Chromium**: Para el scraper web

### 2.3 Cuentas de Servicios Externos
- **Google Cloud Platform**: Para almacenamiento en la nube (opcional)
- **Cuenta de Instagram**: Para scraping de redes sociales (opcional)

---

## 3. INSTALACIÓN Y CONFIGURACIÓN

### 3.1 Clonación del Repositorio

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/PreditorIA2025.git
cd PreditorIA2025

# Verificar estructura del proyecto
ls -la
```

### 3.2 Configuración del Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Verificar instalación
python --version
pip --version
```

### 3.3 Instalación de Dependencias

#### Opción A: Instalación Automática
```bash
# Ejecutar script de instalación
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### Opción B: Instalación Manual
```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias de la aplicación web
pip install -r fashion_trend_app/requirements.txt

# Instalar dependencias de entrenamiento (opcional)
pip install -r requirements_training.txt
```

### 3.4 Verificación de la Instalación

```bash
# Verificar dependencias críticas
python -c "
import tensorflow as tf
import sklearn
import cv2
import numpy as np
import matplotlib.pyplot as plt
print('Todas las dependencias instaladas correctamente')
print(f'TensorFlow: {tf.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
```

---

## 4. ESTRUCTURA DEL PROYECTO

```
PreditorIA2025/
├──  analysis/                    # Análisis de datos y clustering
│   ├── clustering.py              # Algoritmos de clustering
│   ├── fashion_analysis.py        # Análisis específico de moda
│   └── feature_extractor.py       # Extracción de características
├── config/                     # Configuraciones del sistema
│   └── settings.py               # Configuración global
├──  data/                       # Datos del proyecto
│   ├── images/                   # Imágenes originales
│   ├── processed/                # Dataset procesado
│   │   ├── train/               # Conjunto de entrenamiento
│   │   ├── validation/          # Conjunto de validación
│   │   └── test/                # Conjunto de prueba
│   └── logs/                    # Logs del sistema
├── fashion_clustering/         # Módulo principal de clustering
│   ├── config.yaml              # Configuración del clustering
│   ├── extract_embeddings.py    # Extracción de embeddings
│   ├── reduce_dim.py            # Reducción de dimensionalidad
│   ├── run_clustering.py        # Ejecución del clustering
│   └── summarize_clusters.py    # Resumen de clusters
├── fashion_trend_app/         # Aplicación web Flask
│   ├── app_with_progress.py     # Aplicación principal
│   ├── models/                  # Modelos entrenados
│   ├── static/                  # Archivos estáticos
│   └── templates/               # Plantillas HTML
├──  scrapers/                  # Módulos de scraping
│   ├── fashion_websites_scraper.py
│   └── google_images_scraper.py
├── storage/                   # Sistema de almacenamiento
│   ├── database.py              # Base de datos local
│   └── cloud_storage.py         # Almacenamiento en la nube
├── training/                  # Entrenamiento de modelos
│   └── two_phase_trainer.py     # Entrenador en dos fases
├── requirements.txt           # Dependencias principales
├── docker-compose.yml         # Configuración Docker
├── Dockerfile                 # Imagen Docker
└── README.md                  # Documentación principal
```

---

## 5. PROCESO DE REPLICACIÓN PASO A PASO

### 5.1 Preparación del Dataset

#### Paso 1: Configuración de Datos
```bash
# Crear directorios necesarios
mkdir -p data/images data/processed/{train,validation,test} data/logs

# Configurar permisos
chmod 755 data/processed/*
```

#### Paso 2: Preparación del Dataset (Opcional)
```bash
# Si tienes imágenes propias, ejecutar:
python data_preparation/dataset_preparer.py

# Esto creará:
# - División train/validation/test (80/10/10)
# - Normalización de imágenes
# - Estadísticas del dataset
```

### 5.2 Configuración del Sistema de Clustering

#### Paso 1: Configurar Parámetros
```bash
# Editar configuración del clustering
nano fashion_clustering/config.yaml
```

**Configuración recomendada para replicación:**
```yaml
data:
  root: "data/processed"
  image_size: [224, 224]
  batch_size: 32  # Reducir si hay problemas de memoria
  num_workers: 4  # Ajustar según CPU

model:
  name: "mobilenetv2"
  checkpoint: "data/logs/training/mobilenet_v2_final.h5"
  freeze_layers: true
  embedding_layer: -2

clustering:
  kmeans:
    n_clusters: 150
    random_state: 42
    max_iter: 300
    n_init: 10
```

#### Paso 2: Verificar Modelo Pre-entrenado
```bash
# Verificar que existe el modelo
ls -la data/logs/training/

# Si no existe, descargar modelo pre-entrenado
# (Ver sección 5.3)
```

### 5.3 Entrenamiento del Modelo (Si es necesario)

#### Opción A: Usar Modelo Pre-entrenado
```bash
# Descargar modelo pre-entrenado de ImageNet
python -c "
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Cargar modelo pre-entrenado
model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Guardar modelo
model.save('data/logs/training/mobilenet_v2_pretrained.h5')
print('Modelo pre-entrenado guardado')
"
```

#### Opción B: Entrenar Modelo Personalizado
```bash
# Ejecutar entrenamiento personalizado
python training/two_phase_trainer.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --output_dir data/logs/training/
```

### 5.4 Ejecución del Pipeline de Clustering

#### Paso 1: Extracción de Embeddings
```bash
cd fashion_clustering

# Extraer características de las imágenes
python extract_embeddings.py \
    --config config.yaml \
    --data_dir ../data/processed \
    --output_dir ../data/embeddings/
```

#### Paso 2: Reducción de Dimensionalidad
```bash
# Aplicar PCA y UMAP
python reduce_dim.py \
    --embeddings_file ../data/embeddings/embeddings.npy \
    --output_dir ../data/reduced/
```

#### Paso 3: Ejecutar Clustering
```bash
# Ejecutar K-Means clustering
python run_clustering.py \
    --reduced_file ../data/reduced/umap_embeddings.npy \
    --config config.yaml \
    --output_dir ../data/clustering_results/
```

#### Paso 4: Análisis y Resumen
```bash
# Generar resumen de clusters
python summarize_clusters.py \
    --clusters_file ../data/clustering_results/kmeans_clusters.pkl \
    --images_dir ../data/processed \
    --output_dir ../data/analysis/
```

### 5.5 Configuración de la Aplicación Web

#### Paso 1: Preparar Modelos para la App
```bash
# Copiar modelos entrenados a la aplicación
cp data/clustering_results/kmeans_model.pkl fashion_trend_app/models/
cp data/clustering_results/pca_model.pkl fashion_trend_app/models/
cp data/clustering_results/clustering_results.pkl fashion_trend_app/models/
```

#### Paso 2: Configurar la Aplicación
```bash
cd fashion_trend_app

# Crear archivo de configuración
cat > config.py << EOF
import os

class Config:
    SECRET_KEY = 'tu-clave-secreta-aqui'
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    MODEL_PATH = 'models/'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    @staticmethod
    def init_app(app):
        pass

config = {
    'default': Config
}
EOF
```

---

## 6. CONFIGURACIÓN DE ENTORNO

### 6.1 Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```bash
# Configuración de la base de datos
DATABASE_URL=sqlite:///data/fashion_analysis.db

# Configuración de Google Cloud (opcional)
GOOGLE_CLOUD_PROJECT=tu-proyecto-gcp
GCS_BUCKET_NAME=tu-bucket-fashion
GOOGLE_APPLICATION_CREDENTIALS=credentials/gcp-key.json

# Configuración de la aplicación
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=tu-clave-secreta-muy-segura

# Configuración del scraper
TARGET_IMAGES=5000
HEADLESS_BROWSER=true
```

### 6.2 Configuración de Google Cloud (Opcional)

#### Paso 1: Crear Proyecto en GCP
1. Ir a [Google Cloud Console](https://console.cloud.google.com/)
2. Crear nuevo proyecto
3. Habilitar Google Cloud Storage API

#### Paso 2: Crear Credenciales
```bash
# Crear directorio para credenciales
mkdir -p credentials

# Descargar archivo JSON de credenciales desde GCP Console
# Guardar como: credentials/gcp-key.json
```

#### Paso 3: Configurar Bucket
```bash
# Crear bucket en GCS
gsutil mb gs://tu-bucket-fashion

# Configurar permisos
gsutil iam ch allUsers:objectViewer gs://tu-bucket-fashion
```

---

## 7. EJECUCIÓN DEL SISTEMA

### 7.1 Ejecución Local

#### Opción A: Ejecución Completa
```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar pipeline completo
python -c "
from fashion_clustering.extract_embeddings import main as extract_main
from fashion_clustering.reduce_dim import main as reduce_main
from fashion_clustering.run_clustering import main as cluster_main
from fashion_clustering.summarize_clusters import main as summarize_main

print(' Iniciando pipeline completo...')
extract_main()
reduce_main()
cluster_main()
summarize_main()
print('Pipeline completado')
"
```

#### Opción B: Ejecución por Componentes
```bash
# 1. Extraer embeddings
cd fashion_clustering
python extract_embeddings.py

# 2. Reducir dimensionalidad
python reduce_dim.py

# 3. Ejecutar clustering
python run_clustering.py

# 4. Generar resumen
python summarize_clusters.py
```

### 7.2 Ejecución con Docker

#### Construir Imagen
```bash
# Construir imagen Docker
docker-compose build
```

#### Ejecutar Servicios
```bash
# Ejecutar solo el clustering
docker-compose up fashion-clustering

# Ejecutar aplicación web
docker-compose up fashion-app

# Ejecutar todo
docker-compose up
```

### 7.3 Ejecución de la Aplicación Web

```bash
cd fashion_trend_app

# Ejecutar aplicación Flask
python app_with_progress.py

# La aplicación estará disponible en:
# http://localhost:5000
```

#### Funcionalidades de la Aplicación Web:
- **Análisis de Imagen**: Subir imagen y obtener cluster asignado
- **Exploración de Clusters**: Ver imágenes representativas de cada cluster
- **Análisis de Tendencias**: Estadísticas de distribución de clusters
- **Visualización UMAP**: Gráfico interactivo de clusters

---

## 8. VERIFICACIÓN DE RESULTADOS

### 8.1 Verificar Archivos Generados

```bash
# Verificar estructura de resultados
ls -la data/
ls -la data/clustering_results/
ls -la data/analysis/
ls -la clustering_results/
```

**Archivos esperados:**
- `embeddings.npy` - Embeddings extraídos
- `umap_embeddings.npy` - Embeddings reducidos con UMAP
- `kmeans_clusters.pkl` - Resultados del clustering
- `cluster_summary.json` - Resumen estadístico
- `umap_visualization.png` - Visualización UMAP

### 8.2 Verificar Métricas de Calidad

```bash
# Ejecutar verificación de métricas
python -c "
import pickle
import json
import numpy as np

# Cargar resultados
with open('data/clustering_results/kmeans_clusters.pkl', 'rb') as f:
    results = pickle.load(f)

# Verificar métricas
print(' Métricas de Clustering:')
print(f'  Número de clusters: {results[\"n_clusters\"]}')
print(f'  Silhouette Score: {results[\"silhouette_score\"]:.3f}')
print(f'  Davies-Bouldin Index: {results[\"dbi_score\"]:.3f}')
print(f'  Inertia: {results[\"inertia\"]:.2f}')

# Verificar distribución
print(f'  Imágenes procesadas: {len(results[\"labels\"])}')
print(f'  Clusters no vacíos: {len(set(results[\"labels\"]))}')
"
```

### 8.3 Verificar Visualizaciones

```bash
# Verificar que se generaron las visualizaciones
ls -la clustering_results/*.png
ls -la data/analysis/*.png

# Verificar contenido de reportes
cat data/analysis/cluster_summary.json | head -20
```

---

## 9. SOLUCIÓN DE PROBLEMAS COMUNES

### 9.1 Problemas de Memoria

**Error**: `OutOfMemoryError` o `CUDA out of memory`

**Solución**:
```bash
# Reducir batch_size en config.yaml
data:
  batch_size: 16  # Reducir de 32 a 16

# O usar CPU en lugar de GPU
export CUDA_VISIBLE_DEVICES=""
```

### 9.2 Problemas de Dependencias

**Error**: `ModuleNotFoundError`

**Solución**:
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Verificar instalación
pip list | grep -E "(tensorflow|sklearn|opencv)"
```

### 9.3 Problemas de Permisos

**Error**: `PermissionError` al escribir archivos

**Solución**:
```bash
# Dar permisos de escritura
chmod -R 755 data/
chmod -R 755 clustering_results/

# O ejecutar con sudo (no recomendado)
sudo chown -R $USER:$USER data/
```

### 9.4 Problemas de Modelo

**Error**: `FileNotFoundError` para modelo pre-entrenado

**Solución**:
```bash
# Descargar modelo pre-entrenado
python -c "
from tensorflow.keras.applications import MobileNetV2
model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model.save('data/logs/training/mobilenet_v2_pretrained.h5')
"
```

### 9.5 Problemas de Docker

**Error**: `Docker build failed`

**Solución**:
```bash
# Limpiar caché de Docker
docker system prune -a

# Reconstruir imagen
docker-compose build --no-cache
```

---

## 10. REFERENCIAS Y DEPENDENCIAS

### 10.1 Dependencias Principales

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `tensorflow` | 2.16+ | Deep Learning |
| `scikit-learn` | 1.3+ | Machine Learning |
| `opencv-python` | 4.8+ | Procesamiento de Imágenes |
| `numpy` | 1.24+ | Computación Numérica |
| `pandas` | 2.0+ | Manipulación de Datos |
| `matplotlib` | 3.7+ | Visualización |
| `seaborn` | 0.12+ | Visualización Estadística |
| `umap-learn` | 0.5+ | Reducción de Dimensionalidad |
| `flask` | 2.3+ | Aplicación Web |
| `pillow` | 10.0+ | Procesamiento de Imágenes |

### 10.2 Referencias Científicas

1. **MobileNetV2**: Sandler, M., et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." CVPR 2018.

2. **UMAP**: McInnes, L., et al. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426, 2018.

3. **K-Means**: Lloyd, S. "Least squares quantization in PCM." IEEE Transactions on Information Theory, 1982.

4. **Transfer Learning**: Pan, S. J., & Yang, Q. "A survey on transfer learning." IEEE Transactions on Knowledge and Data Engineering, 2009.

### 10.3 Recursos Adicionales

- **Documentación TensorFlow**: https://tensorflow.org/docs
- **Documentación scikit-learn**: https://scikit-learn.org/stable/
- **Documentación UMAP**: https://umap-learn.readthedocs.io/
- **Documentación Flask**: https://flask.palletsprojects.com/

---

## 11. CONTACTO Y SOPORTE

### 11.1 Información del Proyecto
- **Autor**: [Tu Nombre]
- **Institución**: [Tu Universidad]
- **Año**: 2025
- **Tipo**: Tesis de Grado

### 11.2 Repositorio
- **GitHub**: https://github.com/tu-usuario/PreditorIA2025
- **Versión**: 1.0.0
- **Licencia**: MIT

### 11.3 Problemas Conocidos
- El sistema requiere al menos 8GB de RAM para procesar datasets grandes
- La primera ejecución puede tardar más tiempo debido a la descarga de modelos
- Se recomienda usar GPU para acelerar el procesamiento

---

## 12. APÉNDICES

### A.1 Comandos de Verificación Rápida

```bash
#!/bin/bash
# Script de verificación rápida

echo "Verificando instalación..."

# Verificar Python
python --version || echo " Python no encontrado"

# Verificar dependencias críticas
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)" || echo "TensorFlow"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)" || echo " scikit-learn"
python -c "import cv2; print(' OpenCV:', cv2.__version__)" || echo " OpenCV"

# Verificar estructura
[ -d "fashion_clustering" ] && echo "Módulo clustering" || echo "Módulo clustering"
[ -d "fashion_trend_app" ] && echo " App web" || echo " App web"
[ -d "data" ] && echo " Directorio datos" || echo " Directorio datos"

echo " Verificación completada"
```

### A.2 Configuración de Entorno de Desarrollo

```bash
# .bashrc o .zshrc
export PYTHONPATH="${PYTHONPATH}:/ruta/a/PreditorIA2025"
export CUDA_VISIBLE_DEVICES="0"  # Usar primera GPU
export TF_CPP_MIN_LOG_LEVEL="2"  # Reducir logs de TensorFlow
```

---

**Fin del Anexo A**

*Este documento proporciona una guía completa para replicar el proyecto PreditorIA2025. Para cualquier duda o problema, consultar la sección de solución de problemas o contactar al autor.*
