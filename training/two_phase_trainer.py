#!/usr/bin/env python3
"""
Entrenador de MobileNet V2 en Dos Fases para Dataset de Moda Guatemalteca
Optimizado para sistemas con memoria limitada
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import gc

# Agregar el directorio raíz al path para imports
current_dir = Path(__file__).parent 
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from storage.cloud_storage import GoogleCloudStorage
from config.settings import DIRECTORIES

class TwoPhaseMobileNetV2Trainer:
    """Entrenador de MobileNet V2 optimizado para memoria limitada"""
    
    def __init__(self, 
                 num_classes=11,
                 input_shape=(224, 224, 3),
                 learning_rate=0.001,
                 batch_size=8,  # Batch size muy pequeño para ahorrar memoria
                 epochs_phase1=20,  # Primera fase: entrenar solo las capas nuevas
                 epochs_phase2=30):  # Segunda fase: fine-tuning completo
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs_phase1 = epochs_phase1
        self.epochs_phase2 = epochs_phase2
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar GCS
        self.gcs = GoogleCloudStorage()
        
        # Directorios de trabajo
        self.work_dir = Path(DIRECTORIES['logs']) / 'training'
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelo y entrenamiento
        self.model = None
        self.history_phase1 = None
        self.history_phase2 = None
        
        # Configurar TensorFlow para usar menos memoria
        self._configure_tensorflow()
    
    def _configure_tensorflow(self):
        """Configura TensorFlow para usar menos memoria"""
        # Configurar para usar menos memoria GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info("GPU configurada para crecimiento de memoria")
            except RuntimeError as e:
                self.logger.warning(f"No se pudo configurar GPU: {e}")
        else:
            self.logger.info("ℹsando CPU para entrenamiento")
        
        # Configurar TensorFlow para usar menos memoria
        try:
            # Limitar el uso de memoria
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices('CPU')[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
        except Exception as e:
            self.logger.warning(f" No se pudo limitar memoria CPU: {e}")
    
    def setup_model(self):
        """Configura el modelo MobileNet V2"""
        self.logger.info(" Configurando modelo MobileNet V2...")
        
        try:
            # Cargar MobileNet V2 pre-entrenado
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Congelar las capas base inicialmente
            base_model.trainable = False
            
            # Agregar capas de clasificación personalizadas
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)  # Reducido de 1024 a 512
            x = Dropout(0.3)(x)  # Reducido de 0.5 a 0.3
            x = Dense(256, activation='relu')(x)  # Reducido de 512 a 256
            x = Dropout(0.2)(x)  # Reducido de 0.3 a 0.2
            predictions = Dense(self.num_classes, activation='softmax')(x)
            
            # Crear modelo final
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compilar modelo
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info(f"Modelo configurado: {self.model.count_params():,} parámetros")
            self.logger.info(f"Arquitectura: {len(self.model.layers)} capas")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error configurando modelo: {e}")
            return False
    
    def prepare_data_generators(self):
        """Prepara generadores de datos optimizados para memoria"""
        self.logger.info("Preparando generadores de datos optimizados...")
        
        try:
            from config.settings import DIRECTORIES
            
            train_dir = Path(DIRECTORIES['processed']) / 'train'
            validation_dir = Path(DIRECTORIES['processed']) / 'validation'
            test_dir = Path(DIRECTORIES['processed']) / 'test'
            
            # Configuración de data augmentation más conservadora
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,  # Reducido de 20 a 10
                width_shift_range=0.1,  # Reducido de 0.2 a 0.1
                height_shift_range=0.1,  # Reducido de 0.2 a 0.1
                shear_range=0.1,  # Reducido de 0.2 a 0.1
                zoom_range=0.1,  # Reducido de 0.2 a 0.1
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            validation_datagen = ImageDataGenerator(rescale=1./255)
            test_datagen = ImageDataGenerator(rescale=1./255)
            
            # Generadores con batch size pequeño
            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True
            )
            
            self.validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            self.test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            # Ajustar número de clases si es necesario
            detected_classes = len(self.train_generator.class_indices)
            if detected_classes != self.num_classes:
                self.logger.info(f"Ajustando modelo de {self.num_classes} a {detected_classes} clases...")
                self.num_classes = detected_classes
                self._adjust_model_classes()
            
            self.logger.info(" Generadores configurados:")
            self.logger.info(f"   • Entrenamiento: {self.train_generator.samples} muestras")
            self.logger.info(f"   • Validación: {self.validation_generator.samples} muestras")
            self.logger.info(f"   • Prueba: {self.test_generator.samples} muestras")
            self.logger.info(f"   • Clases: {self.num_classes}")
            
            return True
            
        except Exception as e:
            self.logger.error(f" Error preparando generadores: {e}")
            return False
    
    def _adjust_model_classes(self):
        """Ajusta el modelo para el número correcto de clases"""
        # Reemplazar la última capa
        base_model = self.model.layers[0]
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def setup_callbacks(self, phase_name):
        """Configura callbacks para el entrenamiento"""
        self.logger.info(f"Configurando callbacks para {phase_name}...")
        
        checkpoint_path = self.work_dir / f"mobilenet_v2_{phase_name}_best.h5"
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        self.logger.info(f" Callbacks configurados para {phase_name}")
        return callbacks
    
    def train_phase1(self):
        """Fase 1: Entrenar solo las capas nuevas (base_model congelado)"""
        self.logger.info("INICIANDO FASE 1: Entrenamiento de capas nuevas")
        self.logger.info("=" * 60)
        
        # Asegurar que las capas base estén congeladas
        for layer in self.model.layers[:-6]:  # Congelar todo excepto las últimas 6 capas
            layer.trainable = False
        
        # Recompilar con learning rate más alto para las capas nuevas
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = self.setup_callbacks("phase1")
        
        self.logger.info("CONFIGURACIÓN FASE 1:")
        self.logger.info(f"   • Épocas: {self.epochs_phase1}")
        self.logger.info(f"   • Batch size: {self.batch_size}")
        self.logger.info(f"   • Learning rate: {self.learning_rate}")
        self.logger.info(f"   • Capas entrenables: {sum([layer.trainable for layer in self.model.layers])}")
        
        # Entrenamiento
        self.history_phase1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs_phase1,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Limpiar memoria
        gc.collect()
        tf.keras.backend.clear_session()
        
        self.logger.info(" FASE 1 COMPLETADA")
        return True
    
    def train_phase2(self):
        """Fase 2: Fine-tuning completo (todas las capas entrenables)"""
        self.logger.info(" INICIANDO FASE 2: Fine-tuning completo")
        self.logger.info("=" * 60)
        
        # Descongelar todas las capas para fine-tuning
        for layer in self.model.layers:
            layer.trainable = True
        
        # Recompilar con learning rate más bajo para fine-tuning
        fine_tune_lr = self.learning_rate * 0.1
        self.model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = self.setup_callbacks("phase2")
        
        self.logger.info(" CONFIGURACIÓN FASE 2:")
        self.logger.info(f"   • Épocas: {self.epochs_phase2}")
        self.logger.info(f"   • Batch size: {self.batch_size}")
        self.logger.info(f"   • Learning rate: {fine_tune_lr}")
        self.logger.info(f"   • Capas entrenables: {sum([layer.trainable for layer in self.model.layers])}")
        
        # Entrenamiento
        self.history_phase2 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs_phase2,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Limpiar memoria
        gc.collect()
        tf.keras.backend.clear_session()
        
        self.logger.info("FASE 2 COMPLETADA")
        return True
    
    def evaluate_model(self):
        """Evalúa el modelo final"""
        self.logger.info("Evaluando modelo final...")
        
        try:
            # Evaluar en conjunto de prueba
            test_loss, test_accuracy = self.model.evaluate(
                self.test_generator,
                steps=self.test_generator.samples // self.batch_size,
                verbose=1
            )
            
            self.logger.info(f"EVALUACIÓN FINAL:")
            self.logger.info(f"   • Test Loss: {test_loss:.4f}")
            self.logger.info(f"   • Test Accuracy: {test_accuracy:.4f}")
            
            # Guardar métricas
            metrics = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy),
                'num_classes': self.num_classes,
                'total_params': self.model.count_params()
            }
            
            metrics_path = self.work_dir / 'final_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"Métricas guardadas en: {metrics_path}")
            
            return test_loss, test_accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluando modelo: {e}")
            return None, None
    
    def save_model(self):
        """Guarda el modelo final"""
        try:
            model_path = self.work_dir / 'mobilenet_v2_final.h5'
            self.model.save(str(model_path))
            self.logger.info(f"Modelo guardado en: {model_path}")
            
            # Guardar configuración
            config_path = self.work_dir / 'training_config.json'
            config = {
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'batch_size': self.batch_size,
                'epochs_phase1': self.epochs_phase1,
                'epochs_phase2': self.epochs_phase2,
                'learning_rate': self.learning_rate
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Configuración guardada en: {config_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {e}")
            return False
    
    def run_complete_training(self):
        """Ejecuta el entrenamiento completo en dos fases"""
        self.logger.info("ENTRENADOR MOBILENET V2 EN DOS FASES")
        self.logger.info("=" * 60)
        
        try:
            # Configurar modelo
            if not self.setup_model():
                return False
            
            # Preparar datos
            if not self.prepare_data_generators():
                return False
            
            # Fase 1: Entrenar capas nuevas
            if not self.train_phase1():
                return False
            
            # Fase 2: Fine-tuning completo
            if not self.train_phase2():
                return False
            
            # Evaluar modelo
            self.evaluate_model()
            
            # Guardar modelo
            self.save_model()
            
            self.logger.info("NTRENAMIENTO COMPLETADO EXITOSAMENTE")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {e}")
            return False

def main():
    """Función principal"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(" INICIANDO ENTRENAMIENTO EN DOS FASES")
    logger.info("=" * 60)
    
    # Crear entrenador optimizado para memoria
    trainer = TwoPhaseMobileNetV2Trainer(
        num_classes=11,
        batch_size=8,  # Batch size muy pequeño
        epochs_phase1=15,  # Menos épocas en fase 1
        epochs_phase2=20   # Menos épocas en fase 2
    )
    
    # Ejecutar entrenamiento
    success = trainer.run_complete_training()
    
    if success:
        logger.info("Entrenamiento completado exitosamente")
    else:
        logger.error("Error en el entrenamiento")
    
    return success

if __name__ == "__main__":
    main()
