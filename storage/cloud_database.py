"""
Base de datos en Cloud Storage para persistencia
Reemplaza SQLite local con almacenamiento en la nube
"""

import json
import logging
import time
from pathlib import Path
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import os

class CloudDatabase:
    """Base de datos basada en Cloud Storage para persistencia total"""
    
    def __init__(self):
        self.bucket_name = os.getenv('GCS_BUCKET_NAME', 'fashion-images-preditoria2025-1755994382-1755994702')
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'preditoria2025-1755994382')
        self.client = None
        self.bucket = None
        self.logger = logging.getLogger(__name__)
        
        # Archivos de metadatos en Cloud Storage
        self.metadata_file = 'database/images_metadata.json'
        self.sessions_file = 'database/sessions.json'
        self.stats_file = 'database/statistics.json'
        
        self.initialize_client()
    
    def initialize_client(self):
        """Inicializar cliente de Google Cloud Storage"""
        try:
            # Usar credenciales del archivo gcp-key.json
            import os
            # Asegurar que las credenciales estén configuradas
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/gcp-key.json'
            
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Verificar que el bucket existe
            if not self.bucket.exists():
                self.logger.error(f"Bucket {self.bucket_name} no existe")
                self.client = None
                self.bucket = None
                return
                
            self.logger.info(f"CloudDatabase inicializada: {self.bucket_name}")
            self.ensure_database_files()
            
        except Exception as e:
            self.logger.error(f"Error inicializando CloudDatabase: {e}")
            self.logger.error(f"Bucket: {self.bucket_name}, Project: {self.project_id}")
            self.client = None
            self.bucket = None
    
    def ensure_database_files(self):
        """Crear archivos de base de datos si no existen"""
        try:
            # Metadata de imágenes
            if not self.file_exists(self.metadata_file):
                self.save_json(self.metadata_file, {"images": [], "last_updated": time.time()})
            
            # Sesiones
            if not self.file_exists(self.sessions_file):
                self.save_json(self.sessions_file, {"sessions": [], "last_updated": time.time()})
            
            # Estadísticas
            if not self.file_exists(self.stats_file):
                self.save_json(self.stats_file, {
                    "total_images": 0,
                    "by_platform": {},
                    "avg_quality": 0.0,
                    "last_updated": time.time()
                })
                
        except Exception as e:
            self.logger.error(f"Error creando archivos de base de datos: {e}")
    
    def file_exists(self, file_path):
        """Verificar si archivo existe en Cloud Storage"""
        try:
            if not self.is_enabled():
                return False
            
            blob = self.bucket.blob(file_path)
            return blob.exists()
        except:
            return False
    
    def load_json(self, file_path):
        """Cargar JSON desde Cloud Storage"""
        try:
            if not self.is_enabled():
                return {}
            
            blob = self.bucket.blob(file_path)
            if blob.exists():
                content = blob.download_as_text()
                return json.loads(content)
            return {}
        except Exception as e:
            self.logger.error(f"Error cargando {file_path}: {e}")
            return {}
    
    def save_json(self, file_path, data):
        """Guardar JSON en Cloud Storage"""
        try:
            if not self.is_enabled():
                return False
            
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(json.dumps(data, indent=2))
            return True
        except Exception as e:
            self.logger.error(f"Error guardando {file_path}: {e}")
            return False
    
    def save_image_metadata(self, image_data):
        """Guardar metadatos de imagen en la nube"""
        try:
            if not self.is_enabled():
                self.logger.warning("CloudDatabase no habilitada, usando fallback local")
                return self.save_local_fallback(image_data)
            
            # Cargar datos existentes
            metadata = self.load_json(self.metadata_file)
            
            # Agregar nueva imagen
            image_data['id'] = len(metadata.get('images', [])) + 1
            image_data['timestamp'] = time.time()
            
            if 'images' not in metadata:
                metadata['images'] = []
            
            metadata['images'].append(image_data)
            metadata['last_updated'] = time.time()
            
            # Guardar de vuelta
            success = self.save_json(self.metadata_file, metadata)
            
            if success:
                # Actualizar estadísticas
                self.update_statistics(image_data)
                self.logger.info(f"Metadatos guardados en nube: {image_data.get('filename', 'unknown')}")
            else:
                self.logger.warning("Fallo guardado en nube, usando fallback local")
                return self.save_local_fallback(image_data)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error guardando metadatos: {e}")
            return self.save_local_fallback(image_data)
    
    def save_local_fallback(self, image_data):
        """Fallback: guardar en SQLite local si Cloud Storage falla"""
        try:
            from storage.database import ImageDatabase
            local_db = ImageDatabase()
            return local_db.save_image_metadata(image_data)
        except Exception as e:
            self.logger.error(f"Error en fallback local: {e}")
            return False
    
    def update_statistics(self, image_data):
        """Actualizar estadísticas en tiempo real"""
        try:
            stats = self.load_json(self.stats_file)
            
            # Actualizar contadores
            stats['total_images'] = stats.get('total_images', 0) + 1
            
            # Por plataforma
            platform = image_data.get('source_platform', 'unknown')
            if 'by_platform' not in stats:
                stats['by_platform'] = {}
            stats['by_platform'][platform] = stats['by_platform'].get(platform, 0) + 1
            
            # Calidad promedio
            quality = image_data.get('quality_score', 0)
            current_avg = stats.get('avg_quality', 0)
            total = stats['total_images']
            stats['avg_quality'] = ((current_avg * (total - 1)) + quality) / total
            
            stats['last_updated'] = time.time()
            
            self.save_json(self.stats_file, stats)
            
        except Exception as e:
            self.logger.error(f"Error actualizando estadísticas: {e}")
    
    def get_image_count(self):
        """Obtener número total de imágenes"""
        try:
            stats = self.load_json(self.stats_file)
            return stats.get('total_images', 0)
        except:
            return 0
    
    def get_statistics(self):
        """Obtener estadísticas completas"""
        try:
            return self.load_json(self.stats_file)
        except:
            return {"total_images": 0, "by_platform": {}, "avg_quality": 0.0}
    
    def image_exists(self, filename):
        """Verificar si imagen ya existe"""
        try:
            metadata = self.load_json(self.metadata_file)
            images = metadata.get('images', [])
            return any(img.get('filename') == filename for img in images)
        except:
            return False
    
    def start_session(self, method):
        """Iniciar sesión de scraping"""
        try:
            sessions = self.load_json(self.sessions_file)
            
            session_id = f"{method}_{int(time.time())}"
            session_data = {
                'id': session_id,
                'method': method,
                'start_time': time.time(),
                'status': 'active',
                'images_collected': 0
            }
            
            if 'sessions' not in sessions:
                sessions['sessions'] = []
            
            sessions['sessions'].append(session_data)
            sessions['last_updated'] = time.time()
            
            self.save_json(self.sessions_file, sessions)
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error iniciando sesión: {e}")
            return None
    
    def update_session(self, session_id, images_collected, status='active'):
        """Actualizar sesión de scraping"""
        try:
            sessions = self.load_json(self.sessions_file)
            
            for session in sessions.get('sessions', []):
                if session['id'] == session_id:
                    session['images_collected'] = images_collected
                    session['status'] = status
                    session['last_updated'] = time.time()
                    break
            
            sessions['last_updated'] = time.time()
            self.save_json(self.sessions_file, sessions)
            
        except Exception as e:
            self.logger.error(f"Error actualizando sesión: {e}")
    
    def is_enabled(self):
        """Verificar si la base de datos está habilitada"""
        return self.client is not None and self.bucket is not None
