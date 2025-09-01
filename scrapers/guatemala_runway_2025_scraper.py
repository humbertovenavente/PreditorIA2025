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

class GuatemalaRunway2025Scraper:
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
                logging.FileHandler(f"{DIRECTORIES['logs']}/guatemala_runway_2025.log"),
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
        
        # TÉRMINOS ULTRA-ESPECIALIZADOS EN PASARELAS Y DISEÑADORES GUATEMALA 2025
        self.guatemala_runway_terms = [
            # PASARELAS DE GUATEMALA 2025
            "pasarela guatemala 2025", "runway guatemala 2025", "catwalk guatemala 2025",
            "desfile guatemala 2025", "fashion show guatemala 2025", "show moda guatemala 2025",
            "presentacion moda guatemala 2025", "presentation fashion guatemala 2025",
            "exhibicion moda guatemala 2025", "fashion exhibition guatemala 2025",
            "lanzamiento coleccion guatemala 2025", "collection launch guatemala 2025",
            
            # FASHION WEEK GUATEMALA 2025
            "fashion week guatemala 2025", "semana moda guatemala 2025", "semanadelamoda guatemala 2025",
            "guatemala fashion week 2025", "guate fashion week 2025", "guatecityfashion 2025",
            "guate city fashion 2025", "fashion week guate 2025", "semana moda guate 2025",
            "fashion week antigua guatemala 2025", "fashion week ciudad guatemala 2025",
            "fashion week xela 2025", "fashion week quetzaltenango 2025",
            
            # DISEÑADORES DE MODA GUATEMALTECOS 2025
            "diseñador moda guatemala 2025", "fashion designer guatemala 2025",
            "diseñador ropa guatemala 2025", "clothing designer guatemala 2025",
            "creador moda guatemala 2025", "fashion creator guatemala 2025",
            "estilista guatemala 2025", "stylist guatemala 2025", "modista guatemala 2025",
            "seamstress guatemala 2025", "sastre guatemala 2025", "tailor guatemala 2025",
            "costurera guatemala 2025", "dressmaker guatemala 2025",
            
            # MARCAS DE DISEÑADORES GUATEMALTECOS 2025
            "marca diseñador guatemala 2025", "designer brand guatemala 2025",
            "casa moda guatemala 2025", "fashion house guatemala 2025",
            "atelier guatemala 2025", "taller moda guatemala 2025",
            "estudio moda guatemala 2025", "fashion studio guatemala 2025",
            "laboratorio moda guatemala 2025", "fashion lab guatemala 2025",
            
            # EVENTOS DE MODA GUATEMALA 2025
            "evento moda guatemala 2025", "fashion event guatemala 2025",
            "evento pasarela guatemala 2025", "runway event guatemala 2025",
            "evento diseñadores guatemala 2025", "designer event guatemala 2025",
            "concurso moda guatemala 2025", "fashion contest guatemala 2025",
            "competencia moda guatemala 2025", "fashion competition guatemala 2025",
            "festival moda guatemala 2025", "fashion festival guatemala 2025",
            
            # TENDENCIAS DE MODA EN PASARELAS GUATEMALA 2025
            "tendencias pasarela guatemala 2025", "runway trends guatemala 2025",
            "tendencias moda guatemala 2025", "fashion trends guatemala 2025",
            "estilos pasarela guatemala 2025", "runway styles guatemala 2025",
            "looks pasarela guatemala 2025", "runway looks guatemala 2025",
            "outfits pasarela guatemala 2025", "runway outfits guatemala 2025",
            "colecciones pasarela guatemala 2025", "runway collections guatemala 2025",
            
            # ROPA DE PASARELA GUATEMALA 2025
            "ropa pasarela guatemala 2025", "runway clothing guatemala 2025",
            "vestidos pasarela guatemala 2025", "runway dresses guatemala 2025",
            "trajes pasarela guatemala 2025", "runway suits guatemala 2025",
            "conjuntos pasarela guatemala 2025", "runway ensembles guatemala 2025",
            "accesorios pasarela guatemala 2025", "runway accessories guatemala 2025",
            "zapatos pasarela guatemala 2025", "runway shoes guatemala 2025",
            
            # MODELOS DE PASARELA GUATEMALA 2025
            "modelos pasarela guatemala 2025", "runway models guatemala 2025",
            "modelos desfile guatemala 2025", "fashion show models guatemala 2025",
            "modelos fashion week guatemala 2025", "fashion week models guatemala 2025",
            "modelos diseñadores guatemala 2025", "designer models guatemala 2025",
            
            # FOTOGRAFÍA DE PASARELA GUATEMALA 2025
            "fotografia pasarela guatemala 2025", "runway photography guatemala 2025",
            "fotografia desfile guatemala 2025", "fashion show photography guatemala 2025",
            "fotografia fashion week guatemala 2025", "fashion week photography guatemala 2025",
            "fotografo pasarela guatemala 2025", "runway photographer guatemala 2025",
            "fotografo moda guatemala 2025", "fashion photographer guatemala 2025",
            
            # VIDEOS Y MEDIA DE PASARELA GUATEMALA 2025
            "video pasarela guatemala 2025", "runway video guatemala 2025",
            "video desfile guatemala 2025", "fashion show video guatemala 2025",
            "video fashion week guatemala 2025", "fashion week video guatemala 2025",
            "streaming pasarela guatemala 2025", "runway streaming guatemala 2025",
            "live pasarela guatemala 2025", "live runway guatemala 2025",
            
            # BACKSTAGE Y PREPARACIÓN GUATEMALA 2025
            "backstage pasarela guatemala 2025", "runway backstage guatemala 2025",
            "backstage desfile guatemala 2025", "fashion show backstage guatemala 2025",
            "preparacion pasarela guatemala 2025", "runway preparation guatemala 2025",
            "maquillaje pasarela guatemala 2025", "runway makeup guatemala 2025",
            "peinado pasarela guatemala 2025", "runway hairstyle guatemala 2025",
            
            # INVITADOS Y CELEBRIDADES GUATEMALA 2025
            "invitados pasarela guatemala 2025", "runway guests guatemala 2025",
            "celebrities pasarela guatemala 2025", "runway celebrities guatemala 2025",
            "influencers pasarela guatemala 2025", "runway influencers guatemala 2025",
            "bloggers pasarela guatemala 2025", "runway bloggers guatemala 2025",
            "prensa pasarela guatemala 2025", "runway press guatemala 2025",
            
            # LOCACIONES DE PASARELA GUATEMALA 2025
            "lugar pasarela guatemala 2025", "runway venue guatemala 2025",
            "lugar desfile guatemala 2025", "fashion show venue guatemala 2025",
            "hotel pasarela guatemala 2025", "runway hotel guatemala 2025",
            "centro eventos pasarela guatemala 2025", "runway event center guatemala 2025",
            "museo pasarela guatemala 2025", "runway museum guatemala 2025",
            "galeria pasarela guatemala 2025", "runway gallery guatemala 2025",
            
            # TEMPORADAS DE PASARELA GUATEMALA 2025
            "primavera verano guatemala 2025", "spring summer guatemala 2025",
            "otoño invierno guatemala 2025", "autumn winter guatemala 2025",
            "coleccion primavera guatemala 2025", "spring collection guatemala 2025",
            "coleccion verano guatemala 2025", "summer collection guatemala 2025",
            "coleccion otoño guatemala 2025", "autumn collection guatemala 2025",
            "coleccion invierno guatemala 2025", "winter collection guatemala 2025",
            
            # ESTILOS DE PASARELA GUATEMALA 2025
            "alta costura guatemala 2025", "haute couture guatemala 2025",
            "pret a porter guatemala 2025", "ready to wear guatemala 2025",
            "moda calle guatemala 2025", "street fashion guatemala 2025",
            "moda urbana guatemala 2025", "urban fashion guatemala 2025",
            "moda contemporanea guatemala 2025", "contemporary fashion guatemala 2025",
            "moda vanguardista guatemala 2025", "avant-garde fashion guatemala 2025",
            
            # MATERIALES Y TÉCNICAS GUATEMALA 2025
            "materiales pasarela guatemala 2025", "runway materials guatemala 2025",
            "telas pasarela guatemala 2025", "runway fabrics guatemala 2025",
            "bordados pasarela guatemala 2025", "runway embroidery guatemala 2025",
            "tejidos pasarela guatemala 2025", "runway textiles guatemala 2025",
            "técnicas pasarela guatemala 2025", "runway techniques guatemala 2025",
            "innovacion pasarela guatemala 2025", "runway innovation guatemala 2025",
            
            # SUSTENTABILIDAD EN PASARELA GUATEMALA 2025
            "moda sustentable guatemala 2025", "sustainable fashion guatemala 2025",
            "moda ecologica guatemala 2025", "eco fashion guatemala 2025",
            "moda reciclada guatemala 2025", "recycled fashion guatemala 2025",
            "moda organica guatemala 2025", "organic fashion guatemala 2025",
            "moda vegana guatemala 2025", "vegan fashion guatemala 2025",
            "moda cruelty free guatemala 2025", "cruelty free fashion guatemala 2025",
            
            # CULTURA MAYA EN PASARELA GUATEMALA 2025
            "moda maya guatemala 2025", "maya fashion guatemala 2025",
            "huipil pasarela guatemala 2025", "runway huipil guatemala 2025",
            "traje tipico pasarela guatemala 2025", "runway traditional dress guatemala 2025",
            "indigena pasarela guatemala 2025", "runway indigenous guatemala 2025",
            "ancestral pasarela guatemala 2025", "runway ancestral guatemala 2025",
            "cultural pasarela guatemala 2025", "runway cultural guatemala 2025",
            
            # FUSIÓN CULTURAL EN PASARELA GUATEMALA 2025
            "fusion cultural pasarela guatemala 2025", "cultural fusion runway guatemala 2025",
            "moda mestiza guatemala 2025", "mestizo fashion guatemala 2025",
            "moda latina guatemala 2025", "latina fashion guatemala 2025",
            "moda hispana guatemala 2025", "hispana fashion guatemala 2025",
            "moda centroamericana guatemala 2025", "central american fashion guatemala 2025",
            
            # TECNOLOGÍA EN PASARELA GUATEMALA 2025
            "moda digital guatemala 2025", "digital fashion guatemala 2025",
            "moda virtual guatemala 2025", "virtual fashion guatemala 2025",
            "moda aumentada guatemala 2025", "augmented fashion guatemala 2025",
            "moda inteligente guatemala 2025", "smart fashion guatemala 2025",
            "moda conectada guatemala 2025", "connected fashion guatemala 2025",
            
            # INFLUENCIAS INTERNACIONALES GUATEMALA 2025
            "moda internacional guatemala 2025", "international fashion guatemala 2025",
            "moda europea guatemala 2025", "european fashion guatemala 2025",
            "moda americana guatemala 2025", "american fashion guatemala 2025",
            "moda asiatica guatemala 2025", "asian fashion guatemala 2025",
            "moda africana guatemala 2025", "african fashion guatemala 2025",
            
            # COLABORACIONES INTERNACIONALES GUATEMALA 2025
            "colaboracion internacional guatemala 2025", "international collaboration guatemala 2025",
            "diseñador extranjero guatemala 2025", "foreign designer guatemala 2025",
            "marca extranjera guatemala 2025", "foreign brand guatemala 2025",
            "partnership internacional guatemala 2025", "international partnership guatemala 2025",
            
            # PREMIOS Y RECONOCIMIENTOS GUATEMALA 2025
            "premio moda guatemala 2025", "fashion award guatemala 2025",
            "reconocimiento diseñador guatemala 2025", "designer recognition guatemala 2025",
            "mejor diseñador guatemala 2025", "best designer guatemala 2025",
            "diseñador del año guatemala 2025", "designer of the year guatemala 2025",
            "excelencia moda guatemala 2025", "fashion excellence guatemala 2025",
            
            # EDUCACIÓN Y FORMACIÓN GUATEMALA 2025
            "escuela moda guatemala 2025", "fashion school guatemala 2025",
            "universidad moda guatemala 2025", "fashion university guatemala 2025",
            "carrera moda guatemala 2025", "fashion career guatemala 2025",
            "curso moda guatemala 2025", "fashion course guatemala 2025",
            "taller moda guatemala 2025", "fashion workshop guatemala 2025",
            "maestria moda guatemala 2025", "fashion master guatemala 2025"
        ]
        
        # Términos adicionales específicos de pasarelas y eventos
        self.additional_runway_terms = [
            # EVENTOS ESPECÍFICOS DE GUATEMALA 2025
            "guatemala fashion week 2025", "guate fashion week 2025", "guatecityfashion 2025",
            "guate city fashion 2025", "fashion week guate 2025", "semana moda guate 2025",
            "fashion week antigua guatemala 2025", "fashion week ciudad guatemala 2025",
            "fashion week xela 2025", "fashion week quetzaltenango 2025",
            "fashion week peten 2025", "fashion week huehuetenango 2025",
            
            # BÚSQUEDAS EN ESPAÑOL ESPECÍFICAS
            "pasarela guatemala", "runway guatemala", "desfile guatemala", "fashion show guatemala",
            "semanadelamoda guatemala", "fashion week guatemala", "diseñador guatemala",
            "designer guatemala", "modista guatemala", "seamstress guatemala",
            "sastre guatemala", "tailor guatemala", "costurera guatemala",
            
            # TÉRMINOS DE MODA ACTUAL 2025
            "streetwear guatemala 2025", "street wear guatemala 2025", "ropa calle guatemala 2025",
            "athleisure guatemala 2025", "ropa deportiva casual guatemala 2025", "casual sport guatemala 2025",
            "normcore guatemala 2025", "estilo normal guatemala 2025", "minimalismo guatemala 2025",
            "minimalist style guatemala 2025", "maximalismo guatemala 2025", "maximalist style guatemala 2025",
            "grunge guatemala 2025", "estilo grunge guatemala 2025", "preppy guatemala 2025",
            "estilo preppy guatemala 2025", "hipster guatemala 2025", "estilo hipster guatemala 2025"
        ]
        
        # Combinar todos los términos
        self.all_runway_terms = self.guatemala_runway_terms + self.additional_runway_terms
    
    def get_random_headers(self):
        """Genera headers aleatorios que funcionan con Google"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-GT,es;q=0.9,en;q=0.8',
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
    
    def scrape_google_images_runway(self, query, max_images=60):
        """Scraper para Google Images específico para pasarelas y diseñadores de Guatemala 2025"""
        self.logger.info(f"Scraping Google Images PASARELA GUATEMALA 2025 para: {query}")
        
        images_found = 0
        start_index = 0
        
        while images_found < max_images and start_index < 500:  # Máximo 500 resultados por término
            try:
                # Construir URL de búsqueda de Google Images con término de pasarela guatemalteca 2025
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
                            filename = self.generate_filename(image_url, f"runway_{query}")
                            
                            # Verificar si ya existe
                            if self.db.image_exists(filename):
                                continue
                            
                            # Descargar imagen
                            image_path = self.download_image(image_url, filename)
                            if not image_path:
                                continue
                            
                            # Evaluar calidad
                            hashtags = [f"#{query}", "#guatemala", "#pasarela2025", "#runway2025", "#chapin"]
                            quality_results = self.quality_filter.evaluate_image(
                                image_path, f"Imagen de pasarela 2025 {query} en Guatemala", hashtags
                            )
                            
                            if quality_results['passes_filter']:
                                # Obtener información de la imagen
                                image_info = self.quality_filter.get_image_info(image_path)
                                
                                # Preparar datos para la base de datos
                                image_data = {
                                    'filename': filename,
                                    'source_url': image_url,
                                    'source_platform': 'google_images_pasarela_2025_guatemala',
                                    'file_size': image_info['file_size'] if image_info else 0,
                                    'width': image_info['width'] if image_info else 0,
                                    'height': image_info['height'] if image_info else 0,
                                    'format': image_info['format'] if image_info else 'unknown',
                                    'quality_score': quality_results['quality_score'],
                                    'hashtags': hashtags,
                                    'description': f"Imagen de pasarela 2025 {query} en Guatemala",
                                    'location': 'Guatemala',
                                    'metadata': {
                                        'search_query': query,
                                        'sharpness_score': quality_results['sharpness_score'],
                                        'platform': 'google_images_pasarela_2025_guatemala',
                                        'start_index': start_index,
                                        'guatemala_specific': True,
                                        'runway_2025': True,
                                        'category': 'pasarela_2025_guatemala'
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
                                self.logger.info(f"Pasarela 2025 Guatemala {images_found}/{max_images} guardada {cloud_status}: {filename}")
                            else:
                                # Eliminar imagen que no pasó el filtro
                                Path(image_path).unlink(missing_ok=True)
                            
                            # Pausa mínima
                            time.sleep(0.5)
                            
                        except Exception as e:
                            self.logger.error(f"Error procesando imagen de pasarela 2025 de Guatemala: {e}")
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
        
        self.logger.info(f"Google Images Pasarela 2025 Guatemala completado: {images_found} imágenes para {query}")
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
        
        return f"runway_{post_id}_{timestamp}_{url_hash}.{extension}"
    
    def run_guatemala_runway_2025_scraping(self, target_images=5000):
        """Ejecuta scraping de Google Images para pasarelas y diseñadores de Guatemala 2025"""
        self.session_id = self.db.start_session('guatemala_runway_2025_google_5000')
        
        try:
            self.logger.info(f"Iniciando scraping de PASARELAS GUATEMALA 2025 - Objetivo: {target_images} imágenes")
            
            # Distribuir imágenes entre términos
            images_per_term = max(30, target_images // len(self.all_runway_terms))
            
            # Usar procesamiento paralelo para mayor velocidad
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                # Crear tareas para cada término
                future_to_term = {}
                
                for term in self.all_runway_terms:
                    if self.images_collected >= target_images:
                        break
                    
                    # Tarea para Google Images Pasarela 2025 Guatemala
                    future = executor.submit(self.scrape_google_images_runway, term, images_per_term)
                    future_to_term[f"runway_{term}"] = future
                
                # Esperar resultados
                for future_name, future in future_to_term.items():
                    try:
                        result = future.result()
                        self.logger.info(f"Tarea {future_name} completada: {result} imágenes")
                    except Exception as e:
                        self.logger.error(f"Error en tarea {future_name}: {e}")
            
            # Finalizar sesión
            self.db.update_session(self.session_id, self.images_collected, 'completed')
            self.logger.info(f"Sesión de PASARELAS GUATEMALA 2025 completada: {self.images_collected} imágenes recolectadas")
            
        except Exception as e:
            self.logger.error(f"Error en sesión de PASARELAS GUATEMALA 2025: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
    
    def get_progress(self):
        """Retorna el progreso actual"""
        return {
            'images_collected': self.images_collected,
            'total_in_db': self.db.get_image_count(),
            'statistics': self.db.get_statistics()
        }

