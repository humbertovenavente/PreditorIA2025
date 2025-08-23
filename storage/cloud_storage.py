import os
import json
from pathlib import Path
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import logging
from config.settings import DIRECTORIES

class GoogleCloudStorage:
    def __init__(self, bucket_name=None, project_id=None):
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.client = None
        self.bucket = None
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Intentar inicializar cliente
        self.initialize_client()
    
    def initialize_client(self):
        """Inicializa el cliente de Google Cloud Storage"""
        try:
            # Verificar credenciales
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and Path(credentials_path).exists():
                self.client = storage.Client(project=self.project_id)
                self.logger.info("Cliente GCS inicializado con archivo de credenciales")
            else:
                # Intentar con credenciales por defecto
                self.client = storage.Client(project=self.project_id)
                self.logger.info("Cliente GCS inicializado con credenciales por defecto")
            
            # Verificar bucket
            if self.bucket_name:
                self.bucket = self.client.bucket(self.bucket_name)
                self.logger.info(f"Bucket configurado: {self.bucket_name}")
            
        except DefaultCredentialsError:
            self.logger.warning("No se encontraron credenciales de GCS. Funcionando en modo local.")
            self.client = None
        except Exception as e:
            self.logger.error(f"Error inicializando cliente GCS: {e}")
            self.client = None
    
    def is_enabled(self):
        """Verifica si GCS está habilitado y configurado"""
        return self.client is not None and self.bucket is not None
    
    def upload_image(self, local_path, remote_path=None):
        """Sube una imagen a Google Cloud Storage"""
        if not self.is_enabled():
            self.logger.debug("GCS no habilitado, manteniendo archivo local")
            return local_path
        
        try:
            if not remote_path:
                # Generar path remoto basado en el archivo local
                filename = Path(local_path).name
                remote_path = f"images/{filename}"
            
            # Subir archivo
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            
            # Configurar metadatos
            blob.metadata = {
                'uploaded_from': 'fashion_scraper',
                'local_path': str(local_path)
            }
            blob.patch()
            
            self.logger.info(f"Imagen subida a GCS: {remote_path}")
            
            # Retornar URL pública o path del bucket
            return f"gs://{self.bucket_name}/{remote_path}"
        
        except Exception as e:
            self.logger.error(f"Error subiendo imagen a GCS: {e}")
            return local_path
    
    def upload_metadata(self, metadata_dict, remote_path):
        """Sube metadatos como archivo JSON a GCS"""
        if not self.is_enabled():
            return None
        
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_string(
                json.dumps(metadata_dict, indent=2, ensure_ascii=False),
                content_type='application/json'
            )
            
            self.logger.info(f"Metadatos subidos a GCS: {remote_path}")
            return f"gs://{self.bucket_name}/{remote_path}"
        
        except Exception as e:
            self.logger.error(f"Error subiendo metadatos a GCS: {e}")
            return None
    
    def download_image(self, remote_path, local_path):
        """Descarga una imagen desde GCS"""
        if not self.is_enabled():
            return False
        
        try:
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            
            self.logger.info(f"Imagen descargada desde GCS: {remote_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error descargando imagen desde GCS: {e}")
            return False
    
    def list_images(self, prefix="images/"):
        """Lista todas las imágenes en el bucket"""
        if not self.is_enabled():
            return []
        
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        
        except Exception as e:
            self.logger.error(f"Error listando imágenes en GCS: {e}")
            return []
    
    def get_storage_stats(self):
        """Obtiene estadísticas de almacenamiento"""
        if not self.is_enabled():
            return {"enabled": False, "total_files": 0, "total_size": 0}
        
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix="images/")
            total_files = 0
            total_size = 0
            
            for blob in blobs:
                total_files += 1
                total_size += blob.size or 0
            
            return {
                "enabled": True,
                "bucket_name": self.bucket_name,
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas de GCS: {e}")
            return {"enabled": True, "error": str(e)}
    
    def create_bucket_if_not_exists(self, location="us-central1"):
        """Crea el bucket si no existe"""
        if not self.client:
            return False
        
        try:
            # Verificar si el bucket existe
            try:
                self.client.get_bucket(self.bucket_name)
                self.logger.info(f"Bucket {self.bucket_name} ya existe")
                return True
            except:
                pass
            
            # Crear bucket
            bucket = self.client.bucket(self.bucket_name)
            bucket = self.client.create_bucket(bucket, location=location)
            
            self.logger.info(f"Bucket {self.bucket_name} creado en {location}")
            self.bucket = bucket
            return True
        
        except Exception as e:
            self.logger.error(f"Error creando bucket: {e}")
            return False

class HybridStorage:
    """Clase que maneja almacenamiento híbrido: local + nube"""
    
    def __init__(self, bucket_name=None, project_id=None):
        self.local_storage = True
        self.cloud_storage = GoogleCloudStorage(bucket_name, project_id)
        self.logger = logging.getLogger(__name__)
    
    def save_image(self, local_path, metadata=None):
        """Guarda imagen localmente y opcionalmente en la nube"""
        results = {
            'local_path': local_path,
            'cloud_path': None,
            'metadata_cloud_path': None
        }
        
        # Siempre mantener copia local
        if Path(local_path).exists():
            results['local_path'] = local_path
            
            # Intentar subir a la nube si está habilitado
            if self.cloud_storage.is_enabled():
                cloud_path = self.cloud_storage.upload_image(local_path)
                if cloud_path != local_path:  # Se subió exitosamente
                    results['cloud_path'] = cloud_path
                    
                    # Subir metadatos si se proporcionan
                    if metadata:
                        filename = Path(local_path).stem
                        metadata_path = f"metadata/{filename}.json"
                        metadata_cloud_path = self.cloud_storage.upload_metadata(
                            metadata, metadata_path
                        )
                        results['metadata_cloud_path'] = metadata_cloud_path
        
        return results
    
    def get_storage_info(self):
        """Obtiene información del almacenamiento"""
        info = {
            'local_enabled': self.local_storage,
            'cloud_enabled': self.cloud_storage.is_enabled()
        }
        
        if self.cloud_storage.is_enabled():
            info['cloud_stats'] = self.cloud_storage.get_storage_stats()
        
        return info
