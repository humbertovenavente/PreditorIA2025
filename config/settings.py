import os
from dotenv import load_dotenv

load_dotenv()

# Configuración general
TARGET_IMAGES = 5000
MAX_IMAGES_PER_SOURCE = 1000

# Configuración de navegador
BROWSER_CONFIG = {
    'headless': True,
    'window_size': (1920, 1080),
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'implicit_wait': 10
}

# Configuración de delays (para ser respetuoso con los servidores)
DELAYS = {
    'between_requests': 2,  # segundos
    'between_pages': 5,
    'on_error': 10
}

# Filtros de calidad de imagen
IMAGE_QUALITY_FILTERS = {
    'min_width': 300,
    'min_height': 300,
    'max_file_size_mb': 10,
    'allowed_formats': ['jpg', 'jpeg', 'png', 'webp'],
    'min_quality_score': 0.7  # basado en análisis de nitidez
}

# Hashtags relacionados con moda en Guatemala
FASHION_HASHTAGS = [
    '#modaguatemala', '#fashionguatemala', '#ropaguatemala',
    '#estilaguatemala', '#tendenciasgt', '#modagt',
    '#fashiongt', '#guatemalafashion', '#diseñoguatemala',
    '#boutiqueguatemala', '#tiendaguatemala'
]

# Sitios web de moda guatemaltecos
GUATEMALA_FASHION_SITES = [
    'https://www.instagram.com/explore/tags/modaguatemala/',
    'https://www.instagram.com/explore/tags/fashionguatemala/',
    # Agregar más sitios específicos aquí
]

# Configuración de base de datos
DATABASE_CONFIG = {
    'type': 'sqlite',
    'name': 'fashion_images.db',
    'path': 'data/'
}

# Directorios
DIRECTORIES = {
    'images': 'data/images/',
    'metadata': 'data/metadata/',
    'logs': 'data/logs/',
    'temp': 'data/temp/'
}
