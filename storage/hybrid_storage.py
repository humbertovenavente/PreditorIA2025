"""
Sistema de almacenamiento híbrido optimizado
Combina almacenamiento local y Google Cloud Storage
"""

import os
import json
import logging
from pathlib import Path
from storage.cloud_storage import GoogleCloudStorage

class HybridStorage:
    """Sistema de almacenamiento híbrido local + GCS"""
    
    def __init__(self):
        self.gcs = GoogleCloudStorage()
        self.logger = logging.getLogger(__name__)
    
    def save_image(self, image_path, metadata):
        """
        Guarda imagen en almacenamiento híbrido
        
        Args:
            image_path: Ruta local de la imagen
            metadata: Metadatos de la imagen
            
        Returns:
            dict con información del almacenamiento
        """
        result = {
            'local_path': image_path,
            'cloud_path': None,
            'metadata_path': None,
            'success': True
        }
        
        try:
            filename = Path(image_path).name
            
            # Subir a Google Cloud Storage si está disponible
            if self.gcs.is_enabled():
                # Subir imagen
                cloud_image_path = f"images/{filename}"
                if self.gcs.upload_image(image_path, cloud_image_path):
                    result['cloud_path'] = cloud_image_path
                
                # Subir metadatos
                metadata_filename = filename.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json')
                metadata_path = f"metadata/{metadata_filename}"
                
                # Crear archivo temporal de metadatos
                temp_metadata_path = Path(image_path).parent / f"temp_{metadata_filename}"
                with open(temp_metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                if self.gcs.upload_metadata(str(temp_metadata_path), metadata_path):
                    result['metadata_path'] = metadata_path
                
                # Limpiar archivo temporal
                temp_metadata_path.unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en almacenamiento híbrido: {e}")
            result['success'] = False
            return result
