import requests
import time
import hashlib
import logging
from pathlib import Path
from config.settings import DIRECTORIES
from filters.image_quality import ImageQualityFilter
from storage.database import ImageDatabase
from storage.hybrid_storage import HybridStorage
from processors.image_processor import ImageProcessor

class PublicAPIImageScraper:
    """Scraper que usa APIs públicas para obtener imágenes de moda"""
    
    def __init__(self):
        self.quality_filter = ImageQualityFilter()
        self.db = ImageDatabase()
        self.storage = HybridStorage()
        self.image_processor = ImageProcessor()
        self.session_id = None
        self.images_collected = 0
        
        # APIs públicas de imágenes
        self.apis = {
            'unsplash': {
                'url': 'https://api.unsplash.com/search/photos',
                'params': {
                    'query': 'fashion clothing style',
                    'per_page': 30,
                    'orientation': 'portrait'
                },
                'headers': {
                    'Authorization': 'Client-ID YOUR_UNSPLASH_ACCESS_KEY'
                }
            },
            'pixabay': {
                'url': 'https://pixabay.com/api/',
                'params': {
                    'key': 'YOUR_PIXABAY_API_KEY',
                    'q': 'fashion+clothing+style',
                    'image_type': 'photo',
                    'per_page': 20,
                    'min_width': 300,
                    'min_height': 300
                }
            }
        }
        
        # Crear directorios necesarios
        for directory in DIRECTORIES.values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
    
    def download_image(self, image_url, filename):
        """Descarga una imagen desde una URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            image_path = Path(DIRECTORIES['images']) / filename
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            return str(image_path)
        
        except Exception as e:
            self.logger.error(f"Error descargando imagen {image_url}: {e}")
            return None
    
    def generate_filename(self, image_url, source="api"):
        """Genera un nombre único para el archivo"""
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        extension = image_url.split('.')[-1].split('?')[0]
        
        if extension not in ['jpg', 'jpeg', 'png', 'webp']:
            extension = 'jpg'
        
        return f"{source}_{timestamp}_{url_hash}.{extension}"
    
    def scrape_demo_images(self, target_images=10):
        """Scraper de demostración que crea imágenes de ejemplo"""
        self.logger.info(f"Creando {target_images} imágenes de demostración")
        
        # URLs de imágenes de moda de dominio público para demostración
        demo_images = [
            {
                'url': 'https://images.unsplash.com/photo-1445205170230-053b83016050?w=400',
                'description': 'Fashion model in elegant dress',
                'tags': ['fashion', 'model', 'dress', 'style']
            },
            {
                'url': 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400',
                'description': 'Casual fashion outfit',
                'tags': ['fashion', 'casual', 'outfit', 'clothing']
            },
            {
                'url': 'https://images.unsplash.com/photo-1441984904996-e0b6ba687e04?w=400',
                'description': 'Street fashion style',
                'tags': ['fashion', 'street', 'style', 'urban']
            },
            {
                'url': 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400',
                'description': 'Fashion accessories',
                'tags': ['fashion', 'accessories', 'style', 'elegant']
            },
            {
                'url': 'https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=400',
                'description': 'Fashion photography',
                'tags': ['fashion', 'photography', 'model', 'professional']
            }
        ]
        
        collected = 0
        for i, img_data in enumerate(demo_images):
            if collected >= target_images:
                break
            
            try:
                # Generar nombre de archivo
                filename = self.generate_filename(img_data['url'], 'demo')
                
                # Verificar si ya existe
                if self.db.image_exists(filename):
                    continue
                
                # Descargar imagen
                image_path = self.download_image(img_data['url'], filename)
                if not image_path:
                    continue
                
                # Normalizar imagen a 224x224 para MobileNet V2
                processed_path = self.image_processor.normalize_image(image_path)
                if not processed_path:
                    Path(image_path).unlink(missing_ok=True)
                    continue
                
                # Usar imagen procesada para evaluación
                final_path = processed_path
                
                # Evaluar calidad
                quality_results = self.quality_filter.evaluate_image(
                    final_path, img_data['description'], img_data['tags']
                )
                
                if quality_results['passes_filter']:
                    # Obtener información de la imagen procesada
                    image_info = self.quality_filter.get_image_info(final_path)
                    processor_stats = self.image_processor.get_image_stats(final_path)
                    
                    # Preparar datos para la base de datos
                    image_data = {
                        'filename': filename,
                        'source_url': img_data['url'],
                        'source_platform': 'demo_api',
                        'file_size': image_info['file_size'] if image_info else 0,
                        'width': image_info['width'] if image_info else 0,
                        'height': image_info['height'] if image_info else 0,
                        'format': image_info['format'] if image_info else 'unknown',
                        'quality_score': quality_results['quality_score'],
                        'hashtags': img_data['tags'],
                        'description': img_data['description'],
                        'location': 'Demo',
                        'metadata': {
                            'source': 'demo_collection',
                            'sharpness_score': quality_results['sharpness_score'],
                            'processed_for_mobilenet': True,
                            'original_size': f"{image_info.get('width', 0)}x{image_info.get('height', 0)}",
                            'normalized_size': '224x224'
                        }
                    }
                    
                    # Guardar imagen procesada en almacenamiento híbrido
                    storage_result = self.storage.save_image(final_path, image_data)
                    
                    # Limpiar imagen original si es diferente de la procesada
                    if image_path != final_path:
                        Path(image_path).unlink(missing_ok=True)
                    
                    # Actualizar datos con información de almacenamiento
                    if storage_result['cloud_path']:
                        image_data['cloud_url'] = storage_result['cloud_path']
                    
                    # Guardar en base de datos
                    self.db.save_image_metadata(image_data)
                    collected += 1
                    self.images_collected += 1
                    
                    cloud_status = "☁️" if storage_result['cloud_path'] else "💾"
                    self.logger.info(f"Imagen demo {collected}/{target_images} guardada {cloud_status}: {filename}")
                else:
                    # Eliminar imágenes que no pasaron el filtro
                    Path(image_path).unlink(missing_ok=True)
                    if processed_path and processed_path != image_path:
                        Path(processed_path).unlink(missing_ok=True)
                    self.logger.warning(f"Imagen demo no pasó filtros de calidad: {filename}")
                
                time.sleep(0.5)  # Reducir delay para acelerar procesamiento
            
            except Exception as e:
                self.logger.error(f"Error procesando imagen demo: {e}")
                continue
        
        return collected
    
    def run_scraping_session(self, target_images=10):
        """Ejecuta una sesión completa de scraping de APIs públicas"""
        self.session_id = self.db.start_session('public_api')
        
        try:
            self.logger.info(f"Iniciando sesión de scraping API - Objetivo: {target_images} imágenes")
            
            # Por ahora usar imágenes de demostración
            collected = self.scrape_demo_images(target_images)
            
            # Actualizar sesión
            self.db.update_session(self.session_id, collected, 'completed')
            self.logger.info(f"Sesión API completada: {collected} imágenes recolectadas")
            
            return collected
        
        except Exception as e:
            self.logger.error(f"Error en sesión de scraping API: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
            return 0
    
    def get_progress(self):
        """Retorna el progreso actual"""
        return {
            'images_collected': self.images_collected,
            'total_in_db': self.db.get_image_count(),
            'statistics': self.db.get_statistics()
        }
