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

# Hashtags relacionados con moda en Guatemala (en español)
FASHION_HASHTAGS = [
    # Hashtags principales de moda GT
    '#modaguatemala', '#fashionguatemala', '#ropaguatemala',
    '#estilaguatemala', '#tendenciasgt', '#modagt',
    '#fashiongt', '#guatemalafashion', '#diseñoguatemala',
    '#boutiqueguatemala', '#tiendaguatemala',
    
    # Hashtags específicos de ciudades
    '#antiguafashion', '#quetzalfashion', '#chimaltenangofashion',
    '#sacatepequezfashion', '#guatecityfashion', '#xelajufashion',
    
    # Hashtags de estilos locales
    '#modachapina', '#chapinfashion', '#guatefashion',
    '#indigenafashion', '#mayafashion', '#tradicionalgt',
    '#artesaniagt', '#localgt', '#guatemalteco',
    
    # Hashtags de eventos y pasarelas (en español)
    '#semanadelamoda', '#fashionweek', '#modaweek',
    '#eventosmoda', '#desfiles', '#pasarelas',
    '#modaespañola', '#modalatina', '#modahispana',
    
    # Hashtags de diseñadores y boutiques
    '#diseñadoresgt', '#boutiquesgt', '#tiendasmodagt',
    '#estilistasgt', '#modistosgt', '#costurerasgt'
]

# Sitios web de moda guatemaltecos expandidos
GUATEMALA_FASHION_SITES = [
    # Instagram hashtags
    'https://www.instagram.com/explore/tags/modaguatemala/',
    'https://www.instagram.com/explore/tags/fashionguatemala/',
    'https://www.instagram.com/explore/tags/antiguafashion/',
    'https://www.instagram.com/explore/tags/quetzalfashion/',
    'https://www.instagram.com/explore/tags/modachapina/',
    
    # Sitios web de moda GT
    'https://www.guatemala.com/estilo-y-moda/',
    'https://www.revistamujer.com.gt/categoria/moda/',
    'https://www.guatemala.com/antigua-guatemala/',
    
    # Catálogos de stock photos con filtros de Guatemala
    'https://unsplash.com/s/photos/guatemala-fashion',
    'https://unsplash.com/s/photos/guatemala-clothing',
    'https://pixabay.com/images/search/guatemala%20fashion/',
    'https://pexels.com/search/guatemala%20fashion/',
    
    # Redes sociales locales
    'https://www.facebook.com/groups/modaguatemala/',
    'https://www.facebook.com/groups/fashionguatemala/',
    'https://www.facebook.com/groups/antiguafashion/'
]

# Configuración optimizada para 5000 imágenes
SCRAPING_TARGETS = {
    'guatemala_fashion': 1500,    # Scraper específico GT
    'instagram': 1500,            # Instagram con hashtags GT
    'web_catalog': 1000,          # Sitios web y catálogos
    'fast_generation': 1000       # Generación rápida
}

# Configuración de calidad mejorada
IMAGE_QUALITY_FILTERS = {
    'min_width': 300,
    'min_height': 300,
    'max_file_size_mb': 10,
    'allowed_formats': ['jpg', 'jpeg', 'png', 'webp'],
    'min_quality_score': 0.7,  # basado en análisis de nitidez
    'guatemala_relevance_bonus': 0.2,  # Bonus para contenido GT
    'fashion_relevance_bonus': 0.15    # Bonus para contenido de moda
}

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
    'temp': 'data/temp/',
    'processed': 'data/processed/',
    'augmented': 'data/augmented/'
}
