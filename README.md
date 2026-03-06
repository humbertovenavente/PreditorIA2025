# PreditorIA2025 - Sistema de Análisis de Tendencias de Moda

## Descripción
Sistema completo de análisis de tendencias de moda mediante clustering de imágenes utilizando deep learning. Incluye scraper de imágenes, procesamiento de datos, modelo de machine learning, clustering y aplicación web interactiva.

## Características
- **Scraper de Imágenes**: Extracción automática de imágenes de moda
-  **Procesamiento de Datos**: Normalización y filtrado de calidad
- **Deep Learning**: MobileNetV2 para extracción de características
- **Clustering Inteligente**: K-Means con reducción de dimensionalidad (PCA + UMAP)
-  **Aplicación Web**: Interfaz Flask para análisis interactivo
- **Visualización**: Gráficos UMAP y análisis de tendencias
- **Contenedorización**: Docker para fácil despliegue
- **Almacenamiento Híbrido**: Local + Google Cloud Storage

##  Instalación Rápida

### Instalación Automática (Recomendada)
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/PreditorIA2025.git
cd PreditorIA2025

# Instalación automática
./instalar_proyecto.sh

# Verificar instalación
./verificar_instalacion.sh
```

### Instalación Manual
```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r fashion_trend_app/requirements.txt

# Configurar proyecto
mkdir -p data/{images,processed,logs,embeddings,clustering_results}
cp env.example .env
```

### Con Docker
```bash
# Construir y ejecutar
docker-compose build
docker-compose up
```

## Documentación de Replicación

- **[REPLICACION_RAPIDA.md](REPLICACION_RAPIDA.md)** - Guía de 5 minutos
- **[ANEXO_A_REPLICACION_PROYECTO.md](ANEXO_A_REPLICACION_PROYECTO.md)** - Documento completo para tesis

## Configuración de Google Cloud

1. Sigue las instrucciones en `setup_gcp.md`
2. Configura las credenciales en `credentials/gcp-key.json`
3. Actualiza `.env` con tu proyecto y bucket

##  Uso del Sistema

### Pipeline de Clustering
```bash
# Activar entorno
source venv/bin/activate

# Ejecutar pipeline completo
cd fashion_clustering
python extract_embeddings.py    # Extraer características
python reduce_dim.py            # Reducir dimensionalidad
python run_clustering.py        # Ejecutar clustering
python summarize_clusters.py    # Generar resumen
```

### Aplicación Web
```bash
# Iniciar aplicación Flask
cd fashion_trend_app
python app_with_progress.py

# Abrir en navegador: http://localhost:5000
```

### Scraper de Imágenes (Opcional)
```bash
# Scrapear imágenes de moda
python scrapers/fashion_websites_scraper.py
python scrapers/google_images_scraper.py
```

## Estructura del Proyecto
```
PreditorIA2025/
├──  fashion_clustering/     # Módulo principal de clustering
│   ├── config.yaml           # Configuración del clustering
│   ├── extract_embeddings.py # Extracción de características
│   ├── reduce_dim.py         # Reducción de dimensionalidad
│   ├── run_clustering.py     # Ejecución del clustering
│   └── summarize_clusters.py # Análisis de resultados
├── fashion_trend_app/     # Aplicación web Flask
│   ├── app_with_progress.py  # Aplicación principal
│   ├── models/               # Modelos entrenados
│   ├── static/               # Archivos estáticos
│   └── templates/            # Plantillas HTML
├──  analysis/              # Análisis de datos
│   ├── clustering.py         # Algoritmos de clustering
│   └── fashion_analysis.py   # Análisis específico
├──  scrapers/              # Módulos de scraping
│   └── fashion_websites_scraper.py
├──  data/                  # Datos del proyecto
│   ├── images/               # Imágenes originales
│   ├── processed/            # Dataset procesado
│   └── logs/                 # Logs del sistema
├──  storage/               # Sistema de almacenamiento
├──  requirements.txt       # Dependencias principales
├──  docker-compose.yml     # Configuración Docker
└── ANEXO_A_REPLICACION_PROYECTO.md # Documento de replicación
```

## Estimación de Recursos

### Almacenamiento
- **5,000 imágenes**: ~8-12 GB
- **Metadatos**: ~50 MB
- **Logs**: ~100 MB

### Google Cloud Storage
- **Costo mensual**: ~$0.25 USD
- **Transferencia**: Incluida en tier gratuito

### Recursos del Sistema
- **RAM**: 2-4 GB recomendado
- **CPU**: 2 cores mínimo
- **Tiempo estimado**: 6-12 horas

## Consideraciones Éticas
- Respeta términos de servicio de plataformas
-  Implementa delays para evitar sobrecarga
   Solo descarga contenido público
-  Filtra por relevancia de moda
- Almacena metadatos para atribución

## Monitoreo
- Logs detallados en `data/logs/`
- Estadísticas en tiempo real
- Progreso guardado en base de datos
- Respaldo automático en la nube
