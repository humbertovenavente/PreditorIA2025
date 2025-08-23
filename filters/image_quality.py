import cv2
import numpy as np
from PIL import Image
import os
from config.settings import IMAGE_QUALITY_FILTERS

class ImageQualityFilter:
    def __init__(self):
        self.min_width = IMAGE_QUALITY_FILTERS['min_width']
        self.min_height = IMAGE_QUALITY_FILTERS['min_height']
        self.max_file_size = IMAGE_QUALITY_FILTERS['max_file_size_mb'] * 1024 * 1024
        self.allowed_formats = IMAGE_QUALITY_FILTERS['allowed_formats']
        self.min_quality_score = IMAGE_QUALITY_FILTERS['min_quality_score']
    
    def calculate_sharpness(self, image_path):
        """Calcula la nitidez de una imagen usando la varianza del Laplaciano"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0
            
            # Calcular la varianza del Laplaciano
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # Normalizar el score (valores típicos van de 0 a 1000+)
            normalized_score = min(laplacian_var / 1000, 1.0)
            return normalized_score
        except Exception:
            return 0
    
    def check_dimensions(self, image_path):
        """Verifica si las dimensiones de la imagen cumplen los requisitos"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return width >= self.min_width and height >= self.min_height
        except Exception:
            return False
    
    def check_file_size(self, image_path):
        """Verifica si el tamaño del archivo está dentro de los límites"""
        try:
            file_size = os.path.getsize(image_path)
            return file_size <= self.max_file_size
        except Exception:
            return False
    
    def check_format(self, image_path):
        """Verifica si el formato de la imagen es permitido"""
        try:
            with Image.open(image_path) as img:
                format_lower = img.format.lower() if img.format else ''
                return format_lower in self.allowed_formats
        except Exception:
            return False
    
    def is_fashion_related(self, image_path, description="", hashtags=[]):
        """Análisis básico para determinar si la imagen está relacionada con moda"""
        fashion_keywords = [
            'moda', 'fashion', 'ropa', 'clothes', 'outfit', 'style',
            'vestido', 'dress', 'camisa', 'shirt', 'pantalon', 'pants',
            'zapatos', 'shoes', 'accesorios', 'accessories', 'bolsa', 'bag'
        ]
        
        # Buscar palabras clave en descripción y hashtags
        text_content = (description + ' ' + ' '.join(hashtags)).lower()
        
        for keyword in fashion_keywords:
            if keyword in text_content:
                return True
        
        # Aquí se podría agregar análisis de imagen con ML en el futuro
        return True  # Por ahora asumimos que todas las imágenes son relevantes
    
    def evaluate_image(self, image_path, description="", hashtags=[]):
        """Evalúa la calidad general de una imagen"""
        results = {
            'passes_filter': False,
            'quality_score': 0,
            'dimensions_ok': False,
            'file_size_ok': False,
            'format_ok': False,
            'sharpness_score': 0,
            'fashion_related': False
        }
        
        # Verificar dimensiones
        results['dimensions_ok'] = self.check_dimensions(image_path)
        
        # Verificar tamaño de archivo
        results['file_size_ok'] = self.check_file_size(image_path)
        
        # Verificar formato
        results['format_ok'] = self.check_format(image_path)
        
        # Calcular nitidez
        results['sharpness_score'] = self.calculate_sharpness(image_path)
        
        # Verificar relevancia de moda
        results['fashion_related'] = self.is_fashion_related(image_path, description, hashtags)
        
        # Calcular score general
        if all([results['dimensions_ok'], results['file_size_ok'], results['format_ok']]):
            results['quality_score'] = results['sharpness_score']
            results['passes_filter'] = (
                results['quality_score'] >= self.min_quality_score and
                results['fashion_related']
            )
        
        return results
    
    def get_image_info(self, image_path):
        """Obtiene información básica de la imagen"""
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.size[0],
                    'height': img.size[1],
                    'format': img.format.lower() if img.format else 'unknown',
                    'file_size': os.path.getsize(image_path)
                }
        except Exception:
            return None
