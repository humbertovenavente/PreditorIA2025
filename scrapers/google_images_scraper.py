import requests
import json
import time
import hashlib
from pathlib import Path
import logging
from config.settings import DIRECTORIES
from filters.image_quality import ImageQualityFilter
from storage.cloud_database import CloudDatabase
from storage.cloud_storage import HybridStorage
import random
import re
from urllib.parse import quote, unquote
import concurrent.futures

class GoogleImagesScraper:
    def __init__(self):
        self.quality_filter = ImageQualityFilter()
        self.db = CloudDatabase()
        self.storage = HybridStorage()
        self.session_id = None
        self.images_collected = 0
        
        # Crear directorios necesarios
        for directory in DIRECTORIES.values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{DIRECTORIES['logs']}/google_images_scraper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # User-Agents reales que funcionan con Google
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        
        # Términos de moda que funcionan bien en Google Images
        self.fashion_terms = [
            # Español
            "moda", "fashion", "estilo", "style", "ropa", "clothing", "vestidos", "dresses",
            "outfits", "tendencias", "trends", "elegante", "elegant", "chic", "glamour",
            "pasarela", "runway", "desfile", "show", "modelo", "model", "fotografia", "photography",
            
            # Inglés
            "fashion", "style", "clothing", "dress", "outfit", "trend", "elegant", "chic",
            "glamorous", "runway", "model", "photography", "portrait", "beauty", "beautiful",
            "luxury", "luxurious", "sophisticated", "modern", "contemporary", "classic",
            
            # Términos específicos
            "fashion photography", "fashion model", "fashion portrait", "fashion beauty",
            "elegant woman", "chic style", "modern fashion", "contemporary style",
            "luxury fashion", "high fashion", "street style", "street fashion",
            "fashion trends", "style trends", "clothing trends", "dress trends",
            
            # Términos adicionales
            "fashion magazine", "fashion editorial", "fashion campaign", "fashion brand",
            "designer clothing", "haute couture", "ready to wear", "fashion show",
            "fashion week", "fashion event", "fashion industry", "fashion design"
        ]
    
    def get_random_headers(self):
        """Genera headers aleatorios que funcionan con Google"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Referer': 'https://www.google.com/'
        }
    
    def scrape_google_images(self, query, max_images=200):
        """Scraper para Google Images que SÍ funciona"""
        self.logger.info(f"Scraping Google Images para: {query}")
        
        images_found = 0
        start_index = 0
        
        while images_found < max_images and start_index < 1000:  # Máximo 1000 resultados
            try:
                # Construir URL de búsqueda de Google Images
                search_url = f"https://www.google.com/search?q={quote(query)}&tbm=isch&start={start_index}"
                
                headers = self.get_random_headers()
                response = requests.get(search_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Buscar URLs de imágenes en la respuesta de Google
                    image_urls = self.extract_google_image_urls(content)
                    
                    if not image_urls:
                        self.logger.warning(f"No se encontraron imágenes para {query} en índice {start_index}")
                        break
                    
                    for image_url in image_urls:
                        if images_found >= max_images:
                            break
                        
                        try:
                            # Generar nombre de archivo
                            filename = self.generate_filename(image_url, f"google_{query}")
                            
                            # Verificar si ya existe
                            if self.db.image_exists(filename):
                                continue
                            
                            # Descargar imagen
                            image_path = self.download_image(image_url, filename)
                            if not image_path:
                                continue
                            
                            # Evaluar calidad
                            hashtags = [f"#{query}", "#fashion", "#moda", "#google"]
                            quality_results = self.quality_filter.evaluate_image(
                                image_path, f"Imagen de {query} desde Google Images", hashtags
                            )
                            
                            if quality_results['passes_filter']:
                                # Obtener información de la imagen
                                image_info = self.quality_filter.get_image_info(image_path)
                                
                                # Preparar datos para la base de datos
                                image_data = {
                                    'filename': filename,
                                    'source_url': image_url,
                                    'source_platform': 'google_images',
                                    'file_size': image_info['file_size'] if image_info else 0,
                                    'width': image_info['width'] if image_info else 0,
                                    'height': image_info['height'] if image_info else 0,
                                    'format': image_info['format'] if image_info else 'unknown',
                                    'quality_score': quality_results['quality_score'],
                                    'hashtags': hashtags,
                                    'description': f"Imagen de {query} desde Google Images",
                                    'location': 'Internacional',
                                    'metadata': {
                                        'search_query': query,
                                        'sharpness_score': quality_results['sharpness_score'],
                                        'platform': 'google_images',
                                        'start_index': start_index
                                    }
                                }
                                
                                # Guardar en almacenamiento híbrido
                                storage_result = self.storage.save_image(image_path, image_data)
                                
                                # Actualizar datos con información de almacenamiento
                                if storage_result['cloud_path']:
                                    image_data['cloud_url'] = storage_result['cloud_path']
                                
                                # Guardar en base de datos
                                self.db.save_image_metadata(image_data)
                                images_found += 1
                                self.images_collected += 1
                                
                                cloud_status = "☁️" if storage_result['cloud_path'] else "💾"
                                self.logger.info(f"Google {images_found}/{max_images} guardada {cloud_status}: {filename}")
                            else:
                                # Eliminar imagen que no pasó el filtro
                                Path(image_path).unlink(missing_ok=True)
                            
                            # Pausa mínima
                            time.sleep(0.5)
                            
                        except Exception as e:
                            self.logger.error(f"Error procesando imagen de Google: {e}")
                            continue
                    
                    # Avanzar al siguiente conjunto de resultados
                    start_index += 20
                    
                else:
                    self.logger.warning(f"Google Images falló para {query} en índice {start_index}: {response.status_code}")
                    break
                
                # Pausa entre búsquedas
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error scraping Google Images para {query} en índice {start_index}: {e}")
                break
        
        self.logger.info(f"Google Images completado: {images_found} imágenes para {query}")
        return images_found
    
    def extract_google_image_urls(self, content):
        """Extrae URLs de imágenes de la respuesta de Google"""
        image_urls = []
        
        try:
            # Buscar URLs de imágenes en diferentes formatos
            patterns = [
                r'https://[^"]*\.(?:jpg|jpeg|png|webp)',
                r'https://[^"]*\.(?:gif|svg)',
                r'https://[^"]*\.(?:bmp|tiff)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Filtrar URLs válidas
                    if self.is_valid_image_url(match):
                        image_urls.append(match)
            
            # Eliminar duplicados
            image_urls = list(set(image_urls))
            
            # Limitar a 50 URLs por búsqueda
            return image_urls[:50]
            
        except Exception as e:
            self.logger.error(f"Error extrayendo URLs de Google: {e}")
            return []
    
    def is_valid_image_url(self, url):
        """Verifica si una URL es válida para descargar"""
        try:
            # Verificar que sea una URL válida
            if not url.startswith('http'):
                return False
            
            # Verificar que no sea una URL de Google
            if 'google.com' in url or 'googleusercontent.com' in url:
                return False
            
            # Verificar que sea una imagen
            if not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                return False
            
            return True
            
        except:
            return False
    
    def download_image(self, image_url, filename):
        """Descarga una imagen desde una URL"""
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Referer': 'https://www.google.com/'
            }
            
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Verificar que es realmente una imagen
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                self.logger.warning(f"URL no es una imagen: {content_type}")
                return None
            
            image_path = Path(DIRECTORIES['images']) / filename
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            return str(image_path)
        
        except Exception as e:
            self.logger.error(f"Error descargando imagen {image_url}: {e}")
            return None
    
    def generate_filename(self, image_url, post_id=""):
        """Genera un nombre único para el archivo"""
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        extension = image_url.split('.')[-1].split('?')[0]
        
        if extension not in ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff']:
            extension = 'jpg'
        
        return f"google_{post_id}_{timestamp}_{url_hash}.{extension}"
    
    def run_google_scraping(self, target_images=5000):
        """Ejecuta scraping de Google Images para alcanzar el objetivo"""
        self.session_id = self.db.start_session('google_images_5000')
        
        try:
            self.logger.info(f"Iniciando scraping de Google Images - Objetivo: {target_images} imágenes")
            
            # Distribuir imágenes entre términos
            images_per_term = max(200, target_images // len(self.fashion_terms))
            
            # Usar procesamiento paralelo para mayor velocidad
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Crear tareas para cada término
                future_to_term = {}
                
                for term in self.fashion_terms:
                    if self.images_collected >= target_images:
                        break
                    
                    # Tarea para Google Images
                    future = executor.submit(self.scrape_google_images, term, images_per_term)
                    future_to_term[f"google_{term}"] = future
                
                # Esperar resultados
                for future_name, future in future_to_term.items():
                    try:
                        result = future.result()
                        self.logger.info(f"Tarea {future_name} completada: {result} imágenes")
                    except Exception as e:
                        self.logger.error(f"Error en tarea {future_name}: {e}")
            
            # Finalizar sesión
            self.db.update_session(self.session_id, self.images_collected, 'completed')
            self.logger.info(f"Sesión de Google Images completada: {self.images_collected} imágenes recolectadas")
            
        except Exception as e:
            self.logger.error(f"Error en sesión de Google Images: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
    
    def get_progress(self):
        """Retorna el progreso actual"""
        return {
            'images_collected': self.images_collected,
            'total_in_db': self.db.get_image_count(),
            'statistics': self.db.get_statistics()
        }

