# Fashion Clustering - Módulo de Clustering de Estilos

Módulo completo para análisis de clustering de estilos de moda guatemalteca usando MobileNetV2 y algoritmos de clustering avanzados.

## 🚀 Características

- **Extracción de embeddings** con MobileNetV2 preentrenada o afinada
- **Reducción de dimensionalidad** con PCA y UMAP
- **Clustering inteligente** con K-Means y DBSCAN
- **Selección automática de parámetros** basada en métricas de calidad
- **Análisis de colores dominantes** y etiquetado automático
- **Visualizaciones completas** y reportes detallados
- **Análisis temporal** y comparación local vs global

## 📁 Estructura del Proyecto

```
fashion_clustering/
├── __init__.py
├── config.yaml                      # Configuración por defecto
├── extract_embeddings.py            # CLI: extraer embeddings
├── reduce_dim.py                    # CLI: PCA/UMAP
├── run_clustering.py                # CLI: KMeans/DBSCAN/auto
├── summarize_clusters.py            # CLI: etiquetas y resumen
├── visualize.py                     # CLI: todas las gráficas
└── utils/
    ├── __init__.py
    ├── data.py                      # Manejo de datos
    ├── vision.py                    # Visión computacional
    ├── metrics.py                   # Métricas de clustering
    ├── viz.py                       # Visualizaciones
    ├── colors.py                    # Análisis de colores
    ├── io.py                        # Entrada/salida
    └── time.py                      # Análisis temporal
```

## 🛠️ Instalación

### Dependencias

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pillow tensorflow umap-learn pyyaml
```

### Instalación del paquete

```bash
# Desde el directorio raíz del proyecto
pip install -e .
```

## 📊 Uso

### 1. Extracción de Embeddings

```bash
python -m fashion_clustering.extract_embeddings \
  --data_root /path/dataset \
  --model_checkpoint /path/mobilenetv2_finetuned.pth \
  --batch_size 128 \
  --num_workers 8 \
  --resume
```

**Parámetros:**
- `--data_root`: Ruta al dataset con estructura train/val/test
- `--model_checkpoint`: Ruta al modelo afinado (opcional)
- `--batch_size`: Tamaño del lote (default: 128)
- `--num_workers`: Número de workers (default: 8)
- `--resume`: Reanudar si existen archivos

### 2. Reducción de Dimensionalidad

```bash
# PCA
python -m fashion_clustering.reduce_dim \
  --method pca \
  --n_components 50

# UMAP
python -m fashion_clustering.reduce_dim \
  --method umap \
  --umap_components 2 \
  --n_neighbors 15 \
  --min_dist 0.1

# Ambos
python -m fashion_clustering.reduce_dim \
  --method both \
  --n_components 50 \
  --umap_components 2
```

### 3. Clustering

```bash
# K-Means con búsqueda automática de K
python -m fashion_clustering.run_clustering \
  --algo kmeans \
  --k_min 4 \
  --k_max 30

# DBSCAN con búsqueda automática de parámetros
python -m fashion_clustering.run_clustering \
  --algo dbscan \
  --dbscan_eps_grid 0.3 0.5 0.7 0.9 \
  --dbscan_min_samples_grid 5 10 15 20

# Selección automática del mejor algoritmo
python -m fashion_clustering.run_clustering \
  --algo auto \
  --k_min 4 \
  --k_max 30
```

### 4. Resumen y Etiquetado

```bash
python -m fashion_clustering.summarize_clusters \
  --top_prototypes 10 \
  --n_colors 5
```

### 5. Visualizaciones

```bash
# Todas las visualizaciones
python -m fashion_clustering.visualize --all

# Visualizaciones específicas
python -m fashion_clustering.visualize \
  --umap --elbow --silhouette --sizes --prototypes
```

## 📈 Flujo de Trabajo Completo

```bash
# 1. Extraer embeddings
python -m fashion_clustering.extract_embeddings \
  --data_root data/processed \
  --batch_size 128

# 2. Reducir dimensionalidad
python -m fashion_clustering.reduce_dim \
  --method both \
  --n_components 50 \
  --umap_components 2

# 3. Ejecutar clustering
python -m fashion_clustering.run_clustering \
  --algo auto \
  --k_min 4 \
  --k_max 30

# 4. Generar resumen
python -m fashion_clustering.summarize_clusters

# 5. Crear visualizaciones
python -m fashion_clustering.visualize --all
```

## 📋 Archivos de Salida

### Datos
- `embeddings.npy`: Embeddings extraídos (N × 1280)
- `metadata.csv`: Metadatos de las imágenes
- `pca.npy`: Embeddings PCA (N × n_components)
- `umap.npy`: Embeddings UMAP (N × 2)
- `cluster_assignments.csv`: Asignaciones de clusters
- `cluster_stats.json`: Estadísticas del clustering

### Análisis
- `cluster_trends.csv`: Tendencias temporales
- `local_global_stats.json`: Comparación local vs global
- `cluster_summary.md`: Resumen detallado de clusters

### Visualizaciones
- `umap_clusters.png`: Clusters en espacio UMAP
- `elbow.png`: Curva del codo para K-Means
- `silhouette_analysis.png`: Análisis de silhouette
- `cluster_sizes.png`: Tamaños de clusters
- `cluster_{id}_prototypes.jpg`: Mosaicos de prototipos
- `cluster_trends.png`: Tendencias temporales
- `clustering_summary.png`: Gráfico resumen

## ⚙️ Configuración

Edita `fashion_clustering/config.yaml` para personalizar parámetros:

```yaml
# Parámetros del dataset
data:
  root: "data/processed"
  image_size: [224, 224]
  batch_size: 128
  num_workers: 8

# Parámetros de clustering
clustering:
  kmeans:
    k_min: 4
    k_max: 30
    random_state: 42
  dbscan:
    eps_grid: [0.3, 0.5, 0.7, 0.9]
    min_samples_grid: [5, 10, 15, 20]
```

## 🧪 Testing

```bash
# Ejecutar tests
python -m pytest tests/test_fashion_clustering.py -v

# Tests específicos
python -m pytest tests/test_fashion_clustering.py::TestDataUtils -v
```

## 📊 Métricas de Calidad

El módulo calcula automáticamente:

- **Silhouette Score**: Cohesión y separación de clusters
- **Davies-Bouldin Index**: Compacidad y separación
- **Calinski-Harabasz Index**: Ratio de varianza entre/within clusters

## 🎨 Análisis de Colores

- Extracción de colores dominantes por cluster
- Nombres de colores en español
- Análisis de temperatura de color (cálido/frío)
- Paletas de colores visuales

## 📅 Análisis Temporal

- Tendencias mensuales de clusters
- Comparación local vs global
- Detección de clusters tendenciosos
- Análisis de patrones estacionales

## 🔧 Personalización

### Añadir Nuevos Algoritmos

```python
# En utils/metrics.py
def run_custom_clustering(embeddings, parameters):
    # Implementar algoritmo personalizado
    pass
```

### Añadir Nuevas Visualizaciones

```python
# En utils/viz.py
def plot_custom_visualization(data, save_path):
    # Implementar visualización personalizada
    pass
```

## 🐛 Solución de Problemas

### Error de Memoria
```bash
# Reducir batch_size
python -m fashion_clustering.extract_embeddings --batch_size 64
```

### Error de GPU
```bash
# Forzar CPU
export CUDA_VISIBLE_DEVICES=""
```

### Archivos Existentes
```bash
# Limpiar archivos existentes
rm reports/*.npy reports/*.csv reports/*.json
```

## 📚 Referencias

- MobileNetV2: [Paper](https://arxiv.org/abs/1801.04381)
- UMAP: [Paper](https://arxiv.org/abs/1802.03426)
- Scikit-learn Clustering: [Documentación](https://scikit-learn.org/stable/modules/clustering.html)

## 🤝 Contribuciones

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Autores

- **PreditorIA2025** - Desarrollo inicial

## 📞 Soporte

Para preguntas o problemas, abre un issue en el repositorio o contacta al equipo de desarrollo.


