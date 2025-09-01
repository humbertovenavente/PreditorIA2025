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

class Guatemala2025FashionScraper:
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
                logging.FileHandler(f"{DIRECTORIES['logs']}/guatemala_2025_fashion.log"),
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
        
        # FILTROS ESPECÍFICOS PARA 2025 Y MARCAS GUATEMALTECAS
        self.guatemala_2025_fashion_terms = [
            # MODA GUATEMALA 2025 - TÉRMINOS ACTUALES
            "moda guatemala 2025", "fashion guatemala 2025", "estilo guatemala 2025",
            "style guatemala 2025", "tendencias guatemala 2025", "trends guatemala 2025",
            "ropa guatemala 2025", "clothing guatemala 2025", "vestidos guatemala 2025",
            "dresses guatemala 2025", "outfits guatemala 2025", "looks guatemala 2025",
            
            # MARCAS GUATEMALTECAS ESPECÍFICAS
            "marca guatemala", "brand guatemala", "marcas guatemaltecas", "guatemalan brands",
            "ropa marca guatemala", "clothing brand guatemala", "diseñador guatemala",
            "designer guatemala", "modista guatemala", "seamstress guatemala",
            "sastre guatemala", "tailor guatemala", "costurera guatemala",
            
            # MARCAS ESPECÍFICAS DE GUATEMALA (reales)
            "guatecityfashion", "guate city fashion", "guatemala fashion week",
            "semanadelamoda guatemala", "fashion week guatemala", "desfiles guatemala",
            "pasarelas guatemala", "runway guatemala", "catwalk guatemala",
            
            # TIENDAS Y BOUTIQUES GUATEMALTECAS
            "tienda ropa guatemala", "clothing store guatemala", "boutique guatemala",
            "boutique ropa guatemala", "centro comercial guatemala", "mall guatemala",
            "mercado ropa guatemala", "clothing market guatemala", "plaza ropa guatemala",
            "shopping center guatemala", "comercial ropa guatemala",
            
            # DISEÑADORES GUATEMALTECOS ESPECÍFICOS
            "diseñador moda guatemala", "fashion designer guatemala", "creador moda guatemala",
            "fashion creator guatemala", "estilista guatemala", "stylist guatemala",
            "consultor moda guatemala", "fashion consultant guatemala",
            
            # ROPA POR TEMPORADA 2025
            "ropa primavera guatemala 2025", "spring clothing guatemala 2025",
            "ropa verano guatemala 2025", "summer clothing guatemala 2025",
            "ropa otoño guatemala 2025", "autumn clothing guatemala 2025",
            "ropa invierno guatemala 2025", "winter clothing guatemala 2025",
            
            # ROPA POR OCASIÓN 2025
            "ropa fiesta guatemala 2025", "party clothing guatemala 2025",
            "ropa boda guatemala 2025", "wedding clothing guatemala 2025",
            "ropa graduacion guatemala 2025", "graduation clothing guatemala 2025",
            "ropa quinceañera guatemala 2025", "quinceañera clothing guatemala 2025",
            "ropa carnaval guatemala 2025", "carnival clothing guatemala 2025",
            
            # ROPA POR ESTILO 2025
            "ropa vintage guatemala 2025", "vintage clothing guatemala 2025",
            "ropa retro guatemala 2025", "retro clothing guatemala 2025",
            "ropa bohemia guatemala 2025", "bohemian clothing guatemala 2025",
            "ropa minimalista guatemala 2025", "minimalist clothing guatemala 2025",
            "ropa sostenible guatemala 2025", "sustainable clothing guatemala 2025",
            
            # ROPA POR MATERIAL 2025
            "ropa algodon guatemala 2025", "cotton clothing guatemala 2025",
            "ropa lino guatemala 2025", "linen clothing guatemala 2025",
            "ropa seda guatemala 2025", "silk clothing guatemala 2025",
            "ropa cuero guatemala 2025", "leather clothing guatemala 2025",
            "ropa denim guatemala 2025", "denim clothing guatemala 2025",
            
            # ROPA POR GÉNERO 2025
            "ropa mujer guatemala 2025", "women clothing guatemala 2025",
            "ropa hombre guatemala 2025", "men clothing guatemala 2025",
            "ropa niños guatemala 2025", "children clothing guatemala 2025",
            "ropa bebes guatemala 2025", "baby clothing guatemala 2025",
            "ropa unisex guatemala 2025", "unisex clothing guatemala 2025",
            
            # ROPA POR EDAD 2025
            "ropa adolescente guatemala 2025", "teen clothing guatemala 2025",
            "ropa joven guatemala 2025", "young clothing guatemala 2025",
            "ropa adulto guatemala 2025", "adult clothing guatemala 2025",
            "ropa senior guatemala 2025", "senior clothing guatemala 2025",
            
            # ROPA POR PROFESIÓN 2025
            "ropa doctor guatemala 2025", "doctor clothing guatemala 2025",
            "ropa abogado guatemala 2025", "lawyer clothing guatemala 2025",
            "ropa ingeniero guatemala 2025", "engineer clothing guatemala 2025",
            "ropa profesor guatemala 2025", "teacher clothing guatemala 2025",
            "ropa enfermera guatemala 2025", "nurse clothing guatemala 2025",
            "ropa ejecutivo guatemala 2025", "executive clothing guatemala 2025",
            
            # ROPA POR DEPORTE 2025
            "ropa futbol guatemala 2025", "soccer clothing guatemala 2025",
            "ropa basquetbol guatemala 2025", "basketball clothing guatemala 2025",
            "ropa tenis guatemala 2025", "tennis clothing guatemala 2025",
            "ropa natacion guatemala 2025", "swimming clothing guatemala 2025",
            "ropa gym guatemala 2025", "gym clothing guatemala 2025",
            "ropa yoga guatemala 2025", "yoga clothing guatemala 2025",
            "ropa running guatemala 2025", "running clothing guatemala 2025",
            
            # ROPA POR ACCESORIOS 2025
            "accesorios ropa guatemala 2025", "clothing accessories guatemala 2025",
            "bolsos guatemala 2025", "bags guatemala 2025", "carteras guatemala 2025",
            "wallets guatemala 2025", "cinturones guatemala 2025", "belts guatemala 2025",
            "sombreros guatemala 2025", "hats guatemala 2025", "bufandas guatemala 2025",
            "scarves guatemala 2025", "guantes guatemala 2025", "gloves guatemala 2025",
            "joyeria guatemala 2025", "jewelry guatemala 2025",
            
            # ROPA POR COLOR 2025
            "ropa azul guatemala 2025", "blue clothing guatemala 2025",
            "ropa roja guatemala 2025", "red clothing guatemala 2025",
            "ropa verde guatemala 2025", "green clothing guatemala 2025",
            "ropa negra guatemala 2025", "black clothing guatemala 2025",
            "ropa blanca guatemala 2025", "white clothing guatemala 2025",
            "ropa multicolor guatemala 2025", "multicolor clothing guatemala 2025",
            "ropa pastel guatemala 2025", "pastel clothing guatemala 2025",
            "ropa neón guatemala 2025", "neon clothing guatemala 2025",
            
            # ROPA POR PATRÓN 2025
            "ropa estampada guatemala 2025", "printed clothing guatemala 2025",
            "ropa rayada guatemala 2025", "striped clothing guatemala 2025",
            "ropa cuadros guatemala 2025", "plaid clothing guatemala 2025",
            "ropa flores guatemala 2025", "floral clothing guatemala 2025",
            "ropa geometrica guatemala 2025", "geometric clothing guatemala 2025",
            "ropa animal print guatemala 2025", "animal print clothing guatemala 2025",
            "ropa tie dye guatemala 2025", "tie dye clothing guatemala 2025",
            
            # ROPA POR TAMAÑO 2025
            "ropa talla grande guatemala 2025", "plus size clothing guatemala 2025",
            "ropa talla pequeña guatemala 2025", "petite clothing guatemala 2025",
            "ropa talla mediana guatemala 2025", "medium size clothing guatemala 2025",
            "ropa talla extra grande guatemala 2025", "extra large clothing guatemala 2025",
            "ropa talla xs guatemala 2025", "xs clothing guatemala 2025",
            "ropa talla xl guatemala 2025", "xl clothing guatemala 2025",
            "ropa talla xxl guatemala 2025", "xxl clothing guatemala 2025",
            
            # ROPA POR PRECIO 2025
            "ropa barata guatemala 2025", "cheap clothing guatemala 2025",
            "ropa economica guatemala 2025", "affordable clothing guatemala 2025",
            "ropa lujo guatemala 2025", "luxury clothing guatemala 2025",
            "ropa premium guatemala 2025", "premium clothing guatemala 2025",
            "ropa outlet guatemala 2025", "outlet clothing guatemala 2025",
            "ropa descuento guatemala 2025", "discount clothing guatemala 2025",
            "ropa oferta guatemala 2025", "offer clothing guatemala 2025",
            
            # ROPA POR ORIGEN 2025
            "ropa importada guatemala 2025", "imported clothing guatemala 2025",
            "ropa nacional guatemala 2025", "national clothing guatemala 2025",
            "ropa artesanal guatemala 2025", "artisan clothing guatemala 2025",
            "ropa hecha a mano guatemala 2025", "handmade clothing guatemala 2025",
            "ropa local guatemala 2025", "local clothing guatemala 2025",
            "ropa regional guatemala 2025", "regional clothing guatemala 2025",
            
            # ROPA POR CIUDAD 2025
            "ropa antigua guatemala 2025", "antigua clothing guatemala 2025",
            "ropa ciudad guatemala 2025", "guatemala city clothing 2025",
            "ropa quetzaltenango 2025", "quetzaltenango clothing 2025",
            "ropa xela 2025", "xela clothing 2025", "ropa peten 2025",
            "peten clothing 2025", "ropa huehuetenango 2025", "huehuetenango clothing 2025",
            "ropa chiquimula 2025", "chiquimula clothing 2025", "ropa escuintla 2025",
            "escuintla clothing 2025", "ropa sacatepequez 2025", "sacatepequez clothing 2025",
            
            # ROPA POR EVENTO 2025
            "ropa fashion week guatemala 2025", "fashion week clothing guatemala 2025",
            "ropa desfile guatemala 2025", "runway clothing guatemala 2025",
            "ropa pasarela guatemala 2025", "catwalk clothing guatemala 2025",
            "ropa evento moda guatemala 2025", "fashion event clothing guatemala 2025",
            "ropa show guatemala 2025", "fashion show clothing guatemala 2025",
            "ropa presentacion guatemala 2025", "presentation clothing guatemala 2025",
            "ropa lanzamiento guatemala 2025", "launch clothing guatemala 2025",
            
            # ROPA POR TEMÁTICA 2025
            "ropa ecologica guatemala 2025", "eco clothing guatemala 2025",
            "ropa sostenible guatemala 2025", "sustainable clothing guatemala 2025",
            "ropa reciclada guatemala 2025", "recycled clothing guatemala 2025",
            "ropa organica guatemala 2025", "organic clothing guatemala 2025",
            "ropa vegana guatemala 2025", "vegan clothing guatemala 2025",
            "ropa cruelty free guatemala 2025", "cruelty free clothing guatemala 2025",
            
            # ROPA POR ESTILO DE VIDA 2025
            "ropa urbana guatemala 2025", "urban clothing guatemala 2025",
            "ropa rural guatemala 2025", "rural clothing guatemala 2025",
            "ropa playera guatemala 2025", "beach clothing guatemala 2025",
            "ropa montaña guatemala 2025", "mountain clothing guatemala 2025",
            "ropa ciudad guatemala 2025", "city clothing guatemala 2025",
            "ropa campo guatemala 2025", "country clothing guatemala 2025"
        ]
        
        # Términos adicionales específicos de marcas y tiendas
        self.additional_brand_terms = [
            # MARCAS Y TIENDAS ESPECÍFICAS DE GUATEMALA
            "guatemala clothing store", "tienda ropa guatemala", "guatemala clothing shop",
            "guatemala clothing market", "mercado ropa guatemala", "guatemala clothing mall",
            "centro comercial ropa guatemala", "guatemala clothing boutique", "boutique ropa guatemala",
            "guatemala clothing designer", "diseñador ropa guatemala", "guatemala clothing brand",
            "marca ropa guatemala", "guatemala clothing factory", "fabrica ropa guatemala",
            "guatemala clothing manufacturer", "fabricante ropa guatemala", "guatemala clothing supplier",
            "proveedor ropa guatemala", "guatemala clothing wholesaler", "mayorista ropa guatemala",
            "guatemala clothing retailer", "minorista ropa guatemala", "guatemala clothing distributor",
            "distribuidor ropa guatemala", "guatemala clothing importer", "importador ropa guatemala",
            "guatemala clothing exporter", "exportador ropa guatemala",
            
            # BÚSQUEDAS EN ESPAÑOL ESPECÍFICAS
            "almacen ropa guatemala", "deposito ropa guatemala", "distribuidor ropa guatemala",
            "importador ropa guatemala", "exportador ropa guatemala", "comerciante ropa guatemala",
            "vendedor ropa guatemala", "comprador ropa guatemala", "cliente ropa guatemala",
            "consumidor ropa guatemala", "usuario ropa guatemala", "comprador ropa guatemala",
            "proveedor ropa guatemala", "suministrador ropa guatemala", "abastecedor ropa guatemala",
            
            # TÉRMINOS DE MODA ACTUAL 2025
            "streetwear guatemala 2025", "street wear guatemala 2025", "ropa calle guatemala 2025",
            "athleisure guatemala 2025", "ropa deportiva casual guatemala 2025", "casual sport guatemala 2025",
            "normcore guatemala 2025", "estilo normal guatemala 2025", "minimalismo guatemala 2025",
            "minimalist style guatemala 2025", "maximalismo guatemala 2025", "maximalist style guatemala 2025",
            "grunge guatemala 2025", "estilo grunge guatemala 2025", "preppy guatemala 2025",
            "estilo preppy guatemala 2025", "hipster guatemala 2025", "estilo hipster guatemala 2025"
        ]
        
        # Combinar todos los términos
        self.all_fashion_terms = self.guatemala_2025_fashion_terms + self.additional_brand_terms
    
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
    
    def scrape_google_images_2025(self, query, max_images=80):
        """Scraper para Google Images específico para moda 2025 de Guatemala"""
        self.logger.info(f"Scraping Google Images MODA 2025 GUATEMALA para: {query}")
        
        images_found = 0
        start_index = 0
        
        while images_found < max_images and start_index < 400:  # Máximo 400 resultados por término
            try:
                # Construir URL de búsqueda de Google Images con término de moda 2025 guatemalteca
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
                            filename = self.generate_filename(image_url, f"2025_{query}")
                            
                            # Verificar si ya existe
                            if self.db.image_exists(filename):
                                continue
                            
                            # Descargar imagen
                            image_path = self.download_image(image_url, filename)
                            if not image_path:
                                continue
                            
                            # Evaluar calidad
                            hashtags = [f"#{query}", "#guatemala", "#moda2025", "#fashion2025", "#chapin"]
                            quality_results = self.quality_filter.evaluate_image(
                                image_path, f"Imagen de moda 2025 {query} en Guatemala", hashtags
                            )
                            
                            if quality_results['passes_filter']:
                                # Obtener información de la imagen
                                image_info = self.quality_filter.get_image_info(image_path)
                                
                                # Preparar datos para la base de datos
                                image_data = {
                                    'filename': filename,
                                    'source_url': image_url,
                                    'source_platform': 'google_images_moda_2025_guatemala',
                                    'file_size': image_info['file_size'] if image_info else 0,
                                    'width': image_info['width'] if image_info else 0,
                                    'height': image_info['height'] if image_info else 0,
                                    'format': image_info['format'] if image_info else 'unknown',
                                    'quality_score': quality_results['quality_score'],
                                    'hashtags': hashtags,
                                    'description': f"Imagen de moda 2025 {query} en Guatemala",
                                    'location': 'Guatemala',
                                    'metadata': {
                                        'search_query': query,
                                        'sharpness_score': quality_results['sharpness_score'],
                                        'platform': 'google_images_moda_2025_guatemala',
                                        'start_index': start_index,
                                        'guatemala_specific': True,
                                        'fashion_2025': True,
                                        'category': 'moda_2025_guatemala'
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
                                self.logger.info(f"Moda 2025 Guatemala {images_found}/{max_images} guardada {cloud_status}: {filename}")
                            else:
                                # Eliminar imagen que no pasó el filtro
                                Path(image_path).unlink(missing_ok=True)
                            
                            # Pausa mínima
                            time.sleep(0.5)
                            
                        except Exception as e:
                            self.logger.error(f"Error procesando imagen de moda 2025 de Guatemala: {e}")
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
        
        self.logger.info(f"Google Images Moda 2025 Guatemala completado: {images_found} imágenes para {query}")
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
        
        return f"2025_{post_id}_{timestamp}_{url_hash}.{extension}"
    
    def run_guatemala_2025_fashion_scraping(self, target_images=5000):
        """Ejecuta scraping de Google Images para moda 2025 de Guatemala"""
        self.session_id = self.db.start_session('guatemala_2025_fashion_google_5000')
        
        try:
            self.logger.info(f"Iniciando scraping de MODA 2025 GUATEMALA - Objetivo: {target_images} imágenes")
            
            # Distribuir imágenes entre términos
            images_per_term = max(40, target_images // len(self.all_fashion_terms))
            
            # Usar procesamiento paralelo para mayor velocidad
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Crear tareas para cada término
                future_to_term = {}
                
                for term in self.all_fashion_terms:
                    if self.images_collected >= target_images:
                        break
                    
                    # Tarea para Google Images Moda 2025 Guatemala
                    future = executor.submit(self.scrape_google_images_2025, term, images_per_term)
                    future_to_term[f"2025_{term}"] = future
                
                # Esperar resultados
                for future_name, future in future_to_term.items():
                    try:
                        result = future.result()
                        self.logger.info(f"Tarea {future_name} completada: {result} imágenes")
                    except Exception as e:
                        self.logger.error(f"Error en tarea {future_name}: {e}")
            
            # Finalizar sesión
            self.db.update_session(self.session_id, self.images_collected, 'completed')
            self.logger.info(f"Sesión de MODA 2025 GUATEMALA completada: {self.images_collected} imágenes recolectadas")
            
        except Exception as e:
            self.logger.error(f"Error en sesión de MODA 2025 GUATEMALA: {e}")
            if self.session_id:
                self.db.update_session(self.session_id, self.images_collected, 'error')
    
    def get_progress(self):
        """Retorna el progreso actual"""
        return {
            'images_collected': self.images_collected,
            'total_in_db': self.db.get_image_count(),
            'statistics': self.db.get_statistics()
        }

