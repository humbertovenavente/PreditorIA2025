import time
import requests
import hashlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from pathlib import Path
import json
import logging
from config.settings import BROWSER_CONFIG, DELAYS, FASHION_HASHTAGS, DIRECTORIES
from filters.image_quality import ImageQualityFilter
from storage.database import ImageDatabase
from storage.cloud_storage import HybridStorage

class InstagramScraper:
    def __init__(self):
        self.driver = None
        self.quality_filter = ImageQualityFilter()
        self.db = ImageDatabase()
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
                logging.FileHandler(f"{DIRECTORIES['logs']}/instagram_scraper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_driver(self):
        """Configura el driver de Chrome con las opciones necesarias"""
        chrome_options = Options()
        
        if BROWSER_CONFIG['headless']:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument(f"--window-size={BROWSER_CONFIG['window_size'][0]},{BROWSER_CONFIG['window_size'][1]}")
        chrome_options.add_argument(f"--user-agent={BROWSER_CONFIG['user_agent']}")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-extensions')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(BROWSER_CONFIG['implicit_wait'])
        
        self.logger.info("Driver de Chrome configurado correctamente")
    
    def extract_hashtags_from_text(self, text):
        """Extrae hashtags de un texto"""
        if not text:
            return []
        
        hashtags = []
        words = text.split()
        for word in words:
            if word.startswith('#'):
                hashtags.append(word.lower())
        
        return hashtags
    
    def download_image(self, image_url, filename):
        """Descarga una imagen desde una URL"""
        try:
            headers = {
                'User-Agent': BROWSER_CONFIG['user_agent'],
                'Referer': 'https://www.instagram.com/'
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
    
    def generate_filename(self, image_url, post_id=""):
        """Genera un nombre único para el archivo"""
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        extension = image_url.split('.')[-1].split('?')[0]
        
        if extension not in ['jpg', 'jpeg', 'png', 'webp']:
            extension = 'jpg'
        
        return f"instagram_{post_id}_{timestamp}_{url_hash}.{extension}"
    
    def scrape_hashtag(self, hashtag, max_images=200):
        """Scraper para un hashtag específico de Instagram"""
        if not hashtag.startswith('#'):
            hashtag = f"#{hashtag}"
        
        url = f"https://www.instagram.com/explore/tags/{hashtag[1:]}/"
        
        try:
            self.logger.info(f"Iniciando scraping para hashtag: {hashtag}")
            self.driver.get(url)
            time.sleep(DELAYS['between_pages'])
            
            # Scroll para cargar más contenido
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            images_found = 0
            scroll_attempts = 0
            max_scrolls = 10
            
            while images_found < max_images and scroll_attempts < max_scrolls:
                # Buscar elementos de imagen
                image_elements = self.driver.find_elements(By.CSS_SELECTOR, "article img")
                
                for img_element in image_elements[images_found:]:
                    if images_found >= max_images:
                        break
                    
                    try:
                        # Obtener URL de la imagen
                        img_src = img_element.get_attribute('src')
                        if not img_src or 'instagram.com' not in img_src:
                            continue
                        
                        # Hacer clic en la imagen para obtener más detalles
                        try:
                            img_element.click()
                            time.sleep(2)
                            
                            # Extraer metadatos del post
                            post_text = ""
                            hashtags = [hashtag]
                            
                            try:
                                # Buscar texto del post
                                text_elements = self.driver.find_elements(By.CSS_SELECTOR, "article div[data-testid='post-text'] span")
                                if text_elements:
                                    post_text = text_elements[0].text
                                    hashtags.extend(self.extract_hashtags_from_text(post_text))
                            except:
                                pass
                            
                            # Cerrar modal
                            close_button = self.driver.find_element(By.CSS_SELECTOR, "svg[aria-label='Close']")
                            close_button.click()
                            time.sleep(1)
                            
                        except:
                            # Si no se puede hacer clic, continuar con la imagen básica
                            hashtags = [hashtag]
                            post_text = ""
                        
                        # Generar nombre de archivo
                        filename = self.generate_filename(img_src, f"hashtag_{hashtag[1:]}")
                        
                        # Verificar si ya existe
                        if self.db.image_exists(filename):
                            continue
                        
                        # Descargar imagen
                        image_path = self.download_image(img_src, filename)
                        if not image_path:
                            continue
                        
                        # Evaluar calidad
                        quality_results = self.quality_filter.evaluate_image(
                            image_path, post_text, hashtags
                        )
                        
                        if quality_results['passes_filter']:
                            # Obtener información de la imagen
                            image_info = self.quality_filter.get_image_info(image_path)
                            
                            # Preparar datos para la base de datos
                            image_data = {
                                'filename': filename,
                                'source_url': img_src,
                                'source_platform': 'instagram',
                                'file_size': image_info['file_size'] if image_info else 0,
                                'width': image_info['width'] if image_info else 0,
                                'height': image_info['height'] if image_info else 0,
                                'format': image_info['format'] if image_info else 'unknown',
                                'quality_score': quality_results['quality_score'],
                                'hashtags': list(set(hashtags)),
                                'description': post_text,
                                'location': 'Guatemala',
                                'metadata': {
                                    'search_hashtag': hashtag,
                                    'sharpness_score': quality_results['sharpness_score']
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
                        
                        time.sleep(DELAYS['between_requests'])
                    
                    except Exception as e:
                        self.logger.error(f"Error procesando imagen: {e}")
                        continue
                
                # Scroll hacia abajo
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(DELAYS['between_pages'])
                
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                last_height = new_height
            
            self.logger.info(f"Completado scraping para {hashtag}: {images_found} imágenes")
            return images_found
        
        except Exception as e:
            self.logger.error(f"Error en scraping de hashtag {hashtag}: {e}")
            return 0
    
    def run_scraping_session(self, target_images=1000):
        """Ejecuta una sesión completa de scraping"""
        self.session_id = self.db.start_session('instagram')
        self.setup_driver()
        
        try:
            self.logger.info(f"Iniciando sesión de scraping - Objetivo: {target_images} imágenes")
            
            # Distribuir imágenes entre hashtags
            images_per_hashtag = target_images // len(FASHION_HASHTAGS)
            
            for hashtag in FASHION_HASHTAGS:
                if self.images_collected >= target_images:
                    break
                
                remaining_images = target_images - self.images_collected
                images_to_collect = min(images_per_hashtag, remaining_images)
                
                collected = self.scrape_hashtag(hashtag, images_to_collect)
                
                # Actualizar progreso en base de datos
                self.db.update_session(self.session_id, self.images_collected)
                
                self.logger.info(f"Progreso: {self.images_collected}/{target_images} imágenes")
                
                if collected == 0:
                    self.logger.warning(f"No se pudieron obtener imágenes para {hashtag}")
                
                time.sleep(DELAYS['between_pages'])
            
            # Finalizar sesión
            self.db.update_session(self.session_id, self.images_collected, 'completed')
            self.logger.info(f"Sesión completada: {self.images_collected} imágenes recolectadas")
            
        except Exception as e:
            self.logger.error(f"Error en sesión de scraping: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def get_progress(self):
        """Retorna el progreso actual"""
        return {
            'images_collected': self.images_collected,
            'total_in_db': self.db.get_image_count(),
            'statistics': self.db.get_statistics()
        }
