"""
Procesador de imágenes para MobileNet V2
Normaliza imágenes a 224x224 px y optimiza para deep learning
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import logging
from typing import Tuple, Optional
import hashlib

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Procesador de imágenes optimizado para MobileNet V2"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
    def normalize_image(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Normaliza imagen a 224x224 px para MobileNet V2
        
        Args:
            image_path: Ruta de la imagen original
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta de la imagen procesada o None si falla
        """
        try:
            # Cargar imagen
            with Image.open(image_path) as img:
                # Convertir a RGB si es necesario
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionar manteniendo aspecto y rellenando
                img = ImageOps.fit(img, self.target_size, Image.Resampling.LANCZOS)
                
                # Generar nombre de salida si no se proporciona
                if output_path is None:
                    path_obj = Path(image_path)
                    output_path = str(path_obj.parent / f"processed_{path_obj.stem}.jpg")
                
                # Guardar imagen procesada
                img.save(output_path, 'JPEG', quality=95, optimize=True)
                
                self.logger.info(f"Imagen normalizada: {output_path}")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Error procesando imagen {image_path}: {e}")
            return None
    
    def batch_process(self, input_dir: str, output_dir: str) -> int:
        """
        Procesa todas las imágenes de un directorio
        
        Args:
            input_dir: Directorio de imágenes originales
            output_dir: Directorio de salida
            
        Returns:
            Número de imágenes procesadas
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                output_file = output_path / f"processed_{image_file.stem}.jpg"
                
                if self.normalize_image(str(image_file), str(output_file)):
                    processed_count += 1
        
        self.logger.info(f"Procesadas {processed_count} imágenes en lote")
        return processed_count
    
    def get_image_stats(self, image_path: str) -> dict:
        """
        Obtiene estadísticas de una imagen
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Diccionario con estadísticas
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas de {image_path}: {e}")
            return {}
    
    def validate_for_mobilenet(self, image_path: str) -> bool:
        """
        Valida si una imagen es adecuada para MobileNet V2
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            True si es válida, False en caso contrario
        """
        stats = self.get_image_stats(image_path)
        
        if not stats:
            return False
        
        # Validaciones para MobileNet V2
        min_size = 32  # Tamaño mínimo recomendado
        max_size = 2048  # Tamaño máximo práctico
        max_file_size = 10 * 1024 * 1024  # 10MB
        
        return (
            stats.get('width', 0) >= min_size and
            stats.get('height', 0) >= min_size and
            stats.get('width', 0) <= max_size and
            stats.get('height', 0) <= max_size and
            stats.get('size_bytes', 0) <= max_file_size and
            stats.get('mode') in ['RGB', 'RGBA', 'L']
        )
