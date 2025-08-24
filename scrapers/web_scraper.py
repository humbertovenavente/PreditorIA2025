import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from pathlib import Path
import hashlib
import logging
from urllib.parse import urljoin, urlparse
from config.settings import BROWSER_CONFIG, DELAYS, DIRECTORIES
from filters.image_quality import ImageQualityFilter
from storage.database import ImageDatabase

class WebScraper:
    def __init__(self):
        self.driver = None
        self.quality_filter = ImageQualityFilter()
        self.db = ImageDatabase()
        self.session_id = None
        self.images_collected = 0
        
        # Sitios web de moda y catálogos públicos
        self.fashion_sites = [
            'https://unsplash.com/s/photos/fashion',
            'https://unsplash.com/s/photos/clothing',
            'https://unsplash.com/s/photos/style',
            'https://pixabay.com/images/search/fashion/',
            'https://pixabay.com/images/search/clothing/',
            'https://pexels.com/search/fashion/',
            'https://pexels.com/search/clothing/',
            # Sitios de moda con imágenes públicas
        ]
        
        # Crear directorios necesarios
        for directory in DIRECTORIES.values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{DIRECTORIES['logs']}/web_scraper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_driver(self):
        """Configura el driver de Chrome"""
        chrome_options = Options()
        
        if BROWSER_CONFIG['headless']:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument(f"--window-size={BROWSER_CONFIG['window_size'][0]},{BROWSER_CONFIG['window_size'][1]}")
        chrome_options.add_argument(f"--user-agent={BROWSER_CONFIG['user_agent']}")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(BROWSER_CONFIG['implicit_wait'])
    
    def extract_images_from_page(self, url, max_images=50):
        """Extrae imágenes de una página web específica"""
        try:
            self.logger.info(f"Extrayendo imágenes de: {url}")
            self.driver.get(url)
            time.sleep(DELAYS['between_pages'])
            
            # Scroll para cargar contenido dinámico
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Obtener HTML de la página
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Buscar imágenes con diferentes selectores para sitios de stock photos
            image_selectors = [
                'img[src*="unsplash"]',
                'img[src*="pixabay"]',
                'img[src*="pexels"]',
                'img[alt*="fashion"]',
                'img[alt*="clothing"]',
                'img[alt*="style"]',
                'img[data-src]',
                '.photo img',
                '.image img',
                '[data-testid="photo-grid"] img'
            ]
            
            images_found = []
            for selector in image_selectors:
                imgs = soup.select(selector)
                for img in imgs:
                    src = img.get('src') or img.get('data-src')
                    if src:
                        # Convertir URL relativa a absoluta
                        full_url = urljoin(url, src)
                        if self.is_valid_image_url(full_url):
                            images_found.append({
                                'url': full_url,
                                'alt': img.get('alt', ''),
                                'title': img.get('title', ''),
                                'source_page': url
                            })
            
            # Limitar número de imágenes
            images_found = images_found[:max_images]
            
            collected = 0
            for img_data in images_found:
                if collected >= max_images:
                    break
                
                try:
                    # Generar nombre de archivo
                    filename = self.generate_filename(img_data['url'], urlparse(url).netloc)
                    
                    # Verificar si ya existe
                    if self.db.image_exists(filename):
                        continue
                    
                    # Descargar imagen
                    image_path = self.download_image(img_data['url'], filename)
                    if not image_path:
                        continue
                    
                    # Evaluar calidad
                    description = f"{img_data['alt']} {img_data['title']}".strip()
                    quality_results = self.quality_filter.evaluate_image(
                        image_path, description, []
                    )
                    
                    if quality_results['passes_filter']:
                        # Obtener información de la imagen
                        image_info = self.quality_filter.get_image_info(image_path)
                        
                        # Preparar datos para la base de datos
                        image_data = {
                            'filename': filename,
                            'source_url': img_data['url'],
                            'source_platform': 'web_catalog',
                            'file_size': image_info['file_size'] if image_info else 0,
                            'width': image_info['width'] if image_info else 0,
                            'height': image_info['height'] if image_info else 0,
                            'format': image_info['format'] if image_info else 'unknown',
                            'quality_score': quality_results['quality_score'],
                            'hashtags': [],
                            'description': description,
                            'location': 'Guatemala',
                            'metadata': {
                                'source_page': img_data['source_page'],
                                'alt_text': img_data['alt'],
                                'title': img_data['title']
                            }
                        }
                        
                        # Guardar en base de datos
                        self.db.save_image_metadata(image_data)
                        collected += 1
                        self.images_collected += 1
                        
                        self.logger.info(f"Imagen guardada: {filename}")
                    else:
                        # Eliminar imagen que no pasó el filtro
                        Path(image_path).unlink(missing_ok=True)
                    
                    time.sleep(DELAYS['between_requests'])
                
                except Exception as e:
                    self.logger.error(f"Error procesando imagen: {e}")
                    continue
            
            return collected
        
        except Exception as e:
            self.logger.error(f"Error extrayendo imágenes de {url}: {e}")
            return 0
    
    def is_valid_image_url(self, url):
        """Verifica si una URL es válida para una imagen"""
        if not url:
            return False
        
        # Verificar extensiones de imagen
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        url_lower = url.lower()
        
        # Verificar si tiene extensión válida o contiene palabras clave de imagen
        has_extension = any(ext in url_lower for ext in valid_extensions)
        has_image_keywords = any(keyword in url_lower for keyword in ['image', 'img', 'photo', 'picture'])
        
        return has_extension or has_image_keywords
    
    def download_image(self, image_url, filename):
        """Descarga una imagen desde una URL"""
        try:
            headers = {
                'User-Agent': BROWSER_CONFIG['user_agent'],
                'Referer': image_url
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
    
    def generate_filename(self, image_url, domain):
        """Genera un nombre único para el archivo"""
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        
        # Limpiar nombre de dominio
        clean_domain = domain.replace('.', '_').replace('-', '_')
        
        # Obtener extensión
        extension = image_url.split('.')[-1].split('?')[0]
        if extension not in ['jpg', 'jpeg', 'png', 'webp']:
            extension = 'jpg'
        
        return f"web_{clean_domain}_{timestamp}_{url_hash}.{extension}"
    
    def scrape_fashion_websites(self, max_images_per_site=100):
        """Scraper para sitios web de moda guatemaltecos"""
        self.setup_driver()
        
        try:
            total_collected = 0
            
            for site_url in self.fashion_sites:
                if total_collected >= max_images_per_site * len(self.fashion_sites):
                    break
                
                collected = self.extract_images_from_page(site_url, max_images_per_site)
                total_collected += collected
                
                self.logger.info(f"Sitio {site_url}: {collected} imágenes recolectadas")
                time.sleep(DELAYS['between_pages'])
            
            return total_collected
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def run_scraping_session(self, target_images=500):
        """Ejecuta una sesión completa de scraping web"""
        self.session_id = self.db.start_session('web_catalog')
        
        try:
            self.logger.info(f"Iniciando scraping web - Objetivo: {target_images} imágenes")
            
            images_per_site = target_images // len(self.fashion_sites)
            collected = self.scrape_fashion_websites(images_per_site)
            
            # Actualizar sesión
            self.db.update_session(self.session_id, collected, 'completed')
            self.logger.info(f"Sesión web completada: {collected} imágenes recolectadas")
            
            return collected
        
        except Exception as e:
            self.logger.error(f"Error en sesión de scraping web: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
            return 0
