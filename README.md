# PreditorIA2025 - Fashion Image Scraper

## Descripción
Scraper de imágenes de moda para tesis que extrae automáticamente 5,000 imágenes de redes sociales y catálogos digitales en Guatemala, aplicando filtros de calidad y almacenando metadatos.

## Características
- ✅ Extracción de imágenes de Instagram y sitios web de moda
- ✅ Filtrado automático de calidad de imágenes
- ✅ Extracción de metadatos (hashtags, descripciones)
- ✅ Almacenamiento híbrido: local + Google Cloud Storage
- ✅ Contenedorización con Docker
- ✅ Base de datos SQLite para metadatos
- ✅ Logging detallado y estadísticas

## Instalación

### Opción 1: Local
```bash
pip install -r requirements.txt
cp .env.example .env
# Configurar variables en .env
```

### Opción 2: Docker (Recomendado)
```bash
# Construir imagen
docker-compose build

# Ejecutar scraper
docker-compose up fashion-scraper

# Ver estadísticas
docker-compose --profile stats up stats-viewer
```

## Configuración de Google Cloud

1. Sigue las instrucciones en `setup_gcp.md`
2. Configura las credenciales en `credentials/gcp-key.json`
3. Actualiza `.env` con tu proyecto y bucket

## Uso

### Comandos Principales
```bash
# Scrapear todas las plataformas
python main.py

# Solo Instagram
python main.py --instagram-only

# Solo sitios web
python main.py --web-only

# Ver estadísticas
python main.py --stats

# Número específico de imágenes
python main.py --images 1000
```

### Con Docker
```bash
# Scraping completo
docker-compose up fashion-scraper

# Solo estadísticas
docker-compose --profile stats up stats-viewer
```

## Estructura del Proyecto
```
├── scrapers/           # Módulos de scraping
│   ├── instagram_scraper.py
│   └── web_scraper.py
├── filters/           # Filtros de calidad de imagen
│   └── image_quality.py
├── storage/           # Sistema de almacenamiento
│   ├── database.py
│   └── cloud_storage.py
├── config/            # Configuraciones
│   └── settings.py
├── data/              # Datos locales
│   ├── images/
│   ├── metadata/
│   └── logs/
├── credentials/       # Credenciales GCP
├── Dockerfile         # Imagen Docker
├── docker-compose.yml # Orquestación
└── setup_gcp.md      # Guía de configuración
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
- ✅ Respeta términos de servicio de plataformas
- ✅ Implementa delays para evitar sobrecarga
- ✅ Solo descarga contenido público
- ✅ Filtra por relevancia de moda
- ✅ Almacena metadatos para atribución

## Monitoreo
- Logs detallados en `data/logs/`
- Estadísticas en tiempo real
- Progreso guardado en base de datos
- Respaldo automático en la nube
