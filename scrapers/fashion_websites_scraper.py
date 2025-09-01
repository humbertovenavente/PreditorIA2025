import requests
import json
import time
import hashlib
from pathlib import Path
import logging
from config.settings import BROWSER_CONFIG, DELAYS, FASHION_HASHTAGS, DIRECTORIES
from filters.image_quality import ImageQualityFilter
from storage.cloud_database import CloudDatabase
from storage.cloud_storage import HybridStorage

class FashionWebsitesScraper:
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
                logging.FileHandler(f"{DIRECTORIES['logs']}/fashion_websites_scraper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Headers para evitar bloqueos
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        
        # Sitios web de moda principales
        self.fashion_sites = [
            {
                'name': 'Vogue',
                'base_url': 'https://www.vogue.com',
                'search_url': 'https://www.vogue.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.vogue\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.vogue\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'Elle',
                'base_url': 'https://www.elle.com',
                'search_url': 'https://www.elle.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.elle\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.elle\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'Harper\'s Bazaar',
                'base_url': 'https://www.harpersbazaar.com',
                'search_url': 'https://www.harpersbazaar.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.harpersbazaar\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.harpersbazaar\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'Cosmopolitan',
                'base_url': 'https://www.cosmopolitan.com',
                'search_url': 'https://www.cosmopolitan.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.cosmopolitan\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.cosmopolitan\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'Marie Claire',
                'base_url': 'https://www.marieclaire.com',
                'search_url': 'https://www.marieclaire.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.marieclaire\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.marieclaire\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'InStyle',
                'base_url': 'https://www.instyle.com',
                'search_url': 'https://www.instyle.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.instyle\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.instyle\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'Glamour',
                'base_url': 'https://www.glamour.com',
                'search_url': 'https://www.glamour.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.glamour\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.glamour\.com[^"]*\.(?:gif|svg)'
                ]
            },
            {
                'name': 'Allure',
                'base_url': 'https://www.allure.com',
                'search_url': 'https://www.allure.com/search?q={query}',
                'image_patterns': [
                    r'https://[^"]*\.allure\.com[^"]*\.(?:jpg|jpeg|png|webp)',
                    r'https://[^"]*\.allure\.com[^"]*\.(?:gif|svg)'
                ]
            }
        ]
        
        # Términos de moda en español para búsquedas
        self.fashion_terms = [
            "semanadelamoda",
            "fashionweek", 
            "modaespañola",
            "modalatina",
            "modahispana",
            "eventosmoda",
            "desfiles",
            "pasarelas",
            "tendenciasmoda",
            "estilomoderno",
            "ropaelegante",
            "modaactual",
            "fashiontrends",
            "runway",
            "couture",
            "haute",
            "luxury",
            "style",
            "trendy",
            "elegant",
            "chic",
            "sophisticated",
            "glamorous",
            "fashionable",
            "stylish"
        ]
    
    def scrape_site_for_query(self, site, query, max_images=100):
        """Scraper para un sitio específico con una consulta"""
        self.logger.info(f"Scraping {site['name']} para: {query}")
        
        try:
            # Construir URL de búsqueda
            search_url = site['search_url'].format(query=query)
            
            response = requests.get(search_url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                content = response.text
                
                images_found = 0
                
                # Buscar imágenes usando patrones del sitio
                import re
                for pattern in site['image_patterns']:
                    matches = re.findall(pattern, content)
                    for match in matches[:max_images]:
                        if images_found >= max_images:
                            break
                            
                        try:
                            # Filtrar solo imágenes de moda (tamaño razonable)
                            if any(keyword in match.lower() for keyword in ['fashion', 'style', 'model', 'clothing', 'dress', 'outfit']):
                                # Generar nombre de archivo
                                filename = self.generate_filename(match, f"{site['name'].lower()}_{query}")
                                
                                # Verificar si ya existe
                                if self.db.image_exists(filename):
                                    continue
                                
                                # Descargar imagen
                                image_path = self.download_image(match, filename)
                                if not image_path:
                                    continue
                                
                                # Evaluar calidad
                                hashtags = [f"#{query}", "#fashion", "#moda", f"#{site['name'].lower()}"]
                                quality_results = self.quality_filter.evaluate_image(
                                    image_path, f"Imagen de {query} desde {site['name']}", hashtags
                                )
                                
                                if quality_results['passes_filter']:
                                    # Obtener información de la imagen
                                    image_info = self.quality_filter.get_image_info(image_path)
                                    
                                    # Preparar datos para la base de datos
                                    image_data = {
                                        'filename': filename,
                                        'source_url': match,
                                        'source_platform': 'fashion_website',
                                        'file_size': image_info['file_size'] if image_info else 0,
                                        'width': image_info['width'] if image_info else 0,
                                        'height': image_info['height'] if image_info else 0,
                                        'format': image_info['format'] if image_info else 'unknown',
                                        'quality_score': quality_results['quality_score'],
                                        'hashtags': hashtags,
                                        'description': f"Imagen de {query} desde {site['name']}",
                                        'location': 'Internacional',
                                        'metadata': {
                                            'search_query': query,
                                            'sharpness_score': quality_results['sharpness_score'],
                                            'platform': 'fashion_website',
                                            'source_site': site['name'],
                                            'source_url': search_url
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
                                    self.logger.info(f"Imagen {images_found}/{max_images} guardada {cloud_status}: {filename}")
                                else:
                                    # Eliminar imagen que no pasó el filtro
                                    Path(image_path).unlink(missing_ok=True)
                                
                                # Pausa entre descargas
                                time.sleep(DELAYS['between_requests'])
                                
                        except Exception as e:
                            self.logger.error(f"Error procesando imagen de {site['name']}: {e}")
                            continue
                
                self.logger.info(f"{site['name']} completado: {images_found} imágenes para {query}")
                return images_found
                
        except Exception as e:
            self.logger.error(f"Error scraping {site['name']} para {query}: {e}")
        
        return 0
    
    def scrape_fashion_terms(self, max_images_per_term=200):
        """Scraper para términos de moda usando todos los sitios"""
        self.logger.info("Iniciando scraping de términos de moda en español usando sitios web")
        
        total_images = 0
        
        for term in self.fashion_terms:
            if self.images_collected >= 5000:  # Límite total
                break
                
            self.logger.info(f"Procesando término: {term}")
            
            # Distribuir imágenes entre sitios
            images_per_site = max_images_per_term // len(self.fashion_sites)
            
            for site in self.fashion_sites:
                if self.images_collected >= 5000:
                    break
                    
                site_images = self.scrape_site_for_query(site, term, images_per_site)
                total_images += site_images
                
                self.logger.info(f"  • {site['name']}: {site_images} imágenes")
                
                # Pausa entre sitios
                time.sleep(DELAYS['between_pages'])
            
            self.logger.info(f"Término {term}: {total_images} imágenes totales")
            self.logger.info(f"Progreso general: {self.images_collected}/5000 imágenes recolectadas")
            
            # Pausa entre términos
            time.sleep(DELAYS['between_pages'])
        
        return total_images
    
    def download_image(self, image_url, filename):
        """Descarga una imagen desde una URL"""
        try:
            headers = {
                'User-Agent': self.headers['User-Agent'],
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
        
        if extension not in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
            extension = 'jpg'
        
        return f"fashion_{post_id}_{timestamp}_{url_hash}.{extension}"
    
    def run_scraping_session(self, target_images=5000):
        """Ejecuta una sesión completa de scraping para alcanzar 5,000 imágenes"""
        self.session_id = self.db.start_session('fashion_websites_5000')
        
        try:
            self.logger.info(f"Iniciando sesión de scraping para 5,000 imágenes - Objetivo: {target_images} imágenes")
            
            # Scraping de términos de moda
            total_collected = self.scrape_fashion_terms(target_images // len(self.fashion_terms))
            
            # Finalizar sesión
            self.db.update_session(self.session_id, self.images_collected, 'completed')
            self.logger.info(f"Sesión completada: {self.images_collected} imágenes recolectadas")
            
        except Exception as e:
            self.logger.error(f"Error en sesión de scraping: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
    
    def get_progress(self):
        """Retorna el progreso actual"""
        return {
            'images_collected': self.images_collected,
            'total_in_db': self.db.get_image_count(),
            'statistics': self.db.get_statistics()
        }

