import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import shutil
import logging
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.settings import DIRECTORIES

class DatasetPreparer:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.original_images_dir = Path(DIRECTORIES['images'])
        self.processed_dir = Path(DIRECTORIES['processed'])
        self.augmented_dir = Path(DIRECTORIES['augmented'])
        
        # Configurar logging primero
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Crear directorio de logs si no existe
        Path(DIRECTORIES['logs']).mkdir(parents=True, exist_ok=True)
        
        # Crear handler para archivo
        file_handler = logging.FileHandler(f"{DIRECTORIES['logs']}/dataset_preparation.log")
        file_handler.setLevel(logging.INFO)
        
        # Crear handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Crear formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Agregar handlers al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Crear directorios necesarios
        self.setup_directories()
        
        # Estadísticas del dataset
        self.dataset_stats = {
            'original_count': 0,
            'processed_count': 0,
            'augmented_count': 0,
            'train_count': 0,
            'val_count': 0,
            'test_count': 0
        }
    
    def setup_directories(self):
        """Crear estructura de directorios para el dataset procesado"""
        directories = [
            self.processed_dir / 'normalized',
            self.processed_dir / 'train',
            self.processed_dir / 'validation',
            self.processed_dir / 'test',
            self.augmented_dir / 'train',
            self.augmented_dir / 'validation',
            self.augmented_dir / 'test'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directorio creado: {directory}")
    
    def normalize_image(self, image_path, output_path):
        """Normalizar una imagen a 224x224px con preservación de aspecto"""
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Obtener dimensiones originales
            h, w = image_rgb.shape[:2]
            
            # Calcular proporción de aspecto
            aspect_ratio = w / h
            
            # Calcular nuevas dimensiones manteniendo proporción
            if aspect_ratio > 1:  # Imagen más ancha que alta
                new_w = self.target_size[0]
                new_h = int(new_w / aspect_ratio)
            else:  # Imagen más alta que ancha
                new_h = self.target_size[1]
                new_w = int(new_h * aspect_ratio)
            
            # Redimensionar imagen
            resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Crear imagen final 224x224 con padding negro
            final_image = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            
            # Calcular posición para centrar la imagen
            y_offset = (self.target_size[1] - new_h) // 2
            x_offset = (self.target_size[0] - new_w) // 2
            
            # Colocar imagen redimensionada en el centro
            final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Guardar imagen normalizada
            cv2.imwrite(str(output_path), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error normalizando imagen {image_path}: {e}")
            return False
    
    def apply_data_augmentation(self, image_path, output_dir, filename_base, augmentations_per_image=3):
        """Aplicar data augmentation a una imagen"""
        try:
            # Leer imagen
            image = Image.open(image_path)
            
            # Lista de transformaciones de data augmentation
            augmentations = [
                # Rotaciones
                lambda img: img.rotate(random.uniform(-15, 15)),
                lambda img: img.rotate(random.uniform(-30, 30)),
                
                # Cambios de brillo
                lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
                lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
                
                # Cambios de contraste
                lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
                lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1)),
                
                # Cambios de saturación
                lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
                lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2)),
                
                # Cambios de nitidez
                lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.2)),
                lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.9, 1.1)),
                
                # Filtros
                lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5)),
                lambda img: img.filter(ImageFilter.UnsharpMask(radius=1, percent=150)),
                
                # Flip horizontal (espejo)
                lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
                
                # Cambios de color
                lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5)),
            ]
            
            # Aplicar transformaciones aleatorias
            for i in range(augmentations_per_image):
                # Seleccionar transformación aleatoria
                transform = random.choice(augmentations)
                
                # Aplicar transformación
                augmented_image = transform(image)
                
                # Generar nombre de archivo único
                augmented_filename = f"{filename_base}_aug_{i+1}.jpg"
                output_path = output_dir / augmented_filename
                
                # Guardar imagen aumentada
                augmented_image.save(output_path, 'JPEG', quality=95)
                
                self.logger.debug(f"Imagen aumentada guardada: {augmented_filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error aplicando data augmentation a {image_path}: {e}")
            return False
    
    def process_all_images(self):
        """Procesar todas las imágenes: normalización y data augmentation"""
        self.logger.info("Iniciando procesamiento de todas las imágenes...")
        
        # Obtener lista de imágenes originales
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        original_images = []
        
        for ext in image_extensions:
            original_images.extend(self.original_images_dir.glob(f"*{ext}"))
            original_images.extend(self.original_images_dir.glob(f"*{ext.upper()}"))
        
        self.dataset_stats['original_count'] = len(original_images)
        self.logger.info(f"Total de imágenes encontradas: {self.dataset_stats['original_count']}")
        
        # Procesar cada imagen
        processed_count = 0
        augmented_count = 0
        
        for image_path in tqdm(original_images, desc="Procesando imágenes"):
            try:
                # Generar nombre de archivo base
                filename_base = image_path.stem
                normalized_filename = f"{filename_base}_norm.jpg"
                normalized_path = self.processed_dir / 'normalized' / normalized_filename
                
                # Normalizar imagen
                if self.normalize_image(image_path, normalized_path):
                    processed_count += 1
                    
                    # Aplicar data augmentation
                    if self.apply_data_augmentation(
                        normalized_path, 
                        self.augmented_dir / 'train', 
                        filename_base
                    ):
                        augmented_count += 3  # 3 imágenes aumentadas por imagen original
                
            except Exception as e:
                self.logger.error(f"Error procesando {image_path}: {e}")
                continue
        
        self.dataset_stats['processed_count'] = processed_count
        self.dataset_stats['augmented_count'] = augmented_count
        
        self.logger.info(f"Procesamiento completado:")
        self.logger.info(f"  - Imágenes normalizadas: {processed_count}")
        self.logger.info(f"  - Imágenes aumentadas: {augmented_count}")
        
        return processed_count, augmented_count
    
    def 
    split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Dividir el dataset en entrenamiento, validación y prueba"""
        self.logger.info("Dividiendo dataset en subconjuntos...")
        
        # Verificar proporciones
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"
        
        # Obtener todas las imágenes procesadas (normalizadas + aumentadas)
        all_images = []
        
        # Imágenes normalizadas
        normalized_dir = self.processed_dir / 'normalized'
        if normalized_dir.exists():
            all_images.extend(list(normalized_dir.glob("*.jpg")))
        
        # Imágenes aumentadas
        augmented_train_dir = self.augmented_dir / 'train'
        if augmented_train_dir.exists():
            all_images.extend(list(augmented_train_dir.glob("*.jpg")))
        
        total_images = len(all_images)
        self.logger.info(f"Total de imágenes para dividir: {total_images}")
        
        if total_images == 0:
            self.logger.error("No hay imágenes procesadas para dividir")
            return False
        
        # Dividir dataset
        # Primero separar entrenamiento del resto
        train_images, temp_images = train_test_split(
            all_images, 
            train_size=train_ratio, 
            random_state=42,
            shuffle=True
        )
        
        # Luego dividir el resto entre validación y prueba
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_images, test_images = train_test_split(
            temp_images,
            train_size=val_ratio_adjusted,
            random_state=42,
            shuffle=True
        )
        
        # Actualizar estadísticas
        self.dataset_stats['train_count'] = len(train_images)
        self.dataset_stats['val_count'] = len(val_images)
        self.dataset_stats['test_count'] = len(test_images)
        
        # Copiar imágenes a directorios correspondientes
        self._copy_images_to_split(train_images, self.processed_dir / 'train', 'entrenamiento')
        self._copy_images_to_split(val_images, self.processed_dir / 'validation', 'validación')
        self._copy_images_to_split(test_images, self.processed_dir / 'test', 'prueba')
        
        # Guardar estadísticas del dataset
        self._save_dataset_stats()
        
        # Generar reporte de división
        self._generate_split_report()
        
        self.logger.info("División del dataset completada exitosamente")
        return True
    
    def _copy_images_to_split(self, images, target_dir, split_name):
        """Copiar imágenes a un directorio de división específico"""
        self.logger.info(f"Copiando {len(images)} imágenes a {split_name}...")
        
        for image_path in tqdm(images, desc=f"Copiando a {split_name}"):
            try:
                target_path = target_dir / image_path.name
                shutil.copy2(image_path, target_path)
            except Exception as e:
                self.logger.error(f"Error copiando {image_path} a {split_name}: {e}")
    
    def _save_dataset_stats(self):
        """Guardar estadísticas del dataset"""
        stats_file = self.processed_dir / 'dataset_stats.json'
        
        stats_data = {
            'target_size': self.target_size,
            'statistics': self.dataset_stats,
            'split_ratios': {
                'train': 0.7,
                'validation': 0.15,
                'test': 0.15
            },
            'augmentation': {
                'augmentations_per_image': 3,
                'total_augmented': self.dataset_stats['augmented_count']
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        self.logger.info(f"Estadísticas del dataset guardadas en: {stats_file}")
    
    def _generate_split_report(self):
        """Generar reporte visual de la división del dataset"""
        try:
            # Crear gráfico de división
            labels = ['Entrenamiento', 'Validación', 'Prueba']
            sizes = [
                self.dataset_stats['train_count'],
                self.dataset_stats['val_count'],
                self.dataset_stats['test_count']
            ]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            plt.figure(figsize=(10, 8))
            
            # Gráfico de pastel
            plt.subplot(1, 2, 1)
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('División del Dataset')
            
            # Gráfico de barras
            plt.subplot(1, 2, 2)
            bars = plt.bar(labels, sizes, color=colors)
            plt.title('Cantidad de Imágenes por Subconjunto')
            plt.ylabel('Número de Imágenes')
            
            # Agregar valores en las barras
            for bar, size in zip(bars, sizes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sizes),
                        str(size), ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Guardar gráfico
            report_path = self.processed_dir / 'dataset_split_report.png'
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Reporte visual guardado en: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generando reporte visual: {e}")
    
    def run_complete_preparation(self):
        """Ejecutar preparación completa del dataset"""
        self.logger.info("=== INICIANDO PREPARACIÓN COMPLETA DEL DATASET ===")
        
        try:
            # Paso 1: Procesar todas las imágenes
            self.logger.info("PASO 1: Normalización y Data Augmentation")
            processed, augmented = self.process_all_images()
            
            if processed == 0:
                self.logger.error("No se pudieron procesar imágenes. Abortando.")
                return False
            
            # Paso 2: Dividir dataset
            self.logger.info("PASO 2: División del Dataset")
            if not self.split_dataset():
                self.logger.error("Error en la división del dataset")
                return False
            
            # Resumen final
            self.logger.info("=== PREPARACIÓN DEL DATASET COMPLETADA ===")
            self.logger.info(f" ESTADÍSTICAS FINALES:")
            self.logger.info(f"  • Imágenes originales: {self.dataset_stats['original_count']}")
            self.logger.info(f"  • Imágenes normalizadas: {self.dataset_stats['processed_count']}")
            self.logger.info(f"  • Imágenes aumentadas: {self.dataset_stats['augmented_count']}")
            self.logger.info(f"  • Total para entrenamiento: {self.dataset_stats['train_count']}")
            self.logger.info(f"  • Total para validación: {self.dataset_stats['val_count']}")
            self.logger.info(f"  • Total para prueba: {self.dataset_stats['test_count']}")
            self.logger.info(f" FORMATO: {self.target_size[0]}x{self.target_size[1]}px (MobileNet V2)")
            self.logger.info(f" DATASET PROCESADO EN: {self.processed_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en la preparación completa del dataset: {e}")
            return False
    
    def get_dataset_info(self):
        """Obtener información del dataset preparado"""
        return {
            'target_size': self.target_size,
            'statistics': self.dataset_stats,
            'directories': {
                'processed': str(self.processed_dir),
                'augmented': str(self.augmented_dir),
                'train': str(self.processed_dir / 'train'),
                'validation': str(self.processed_dir / 'validation'),
                'test': str(self.processed_dir / 'test')
            }
        }
