#!/usr/bin/env python3
"""
Script para corregir la estructura del dataset y crear un dataset balanceado
"""

import os
import shutil
from collections import Counter
import random
from PIL import Image
import json

def analyze_current_structure():
    """Analizar la estructura actual del dataset"""
    print("🔍 ANALIZANDO ESTRUCTURA ACTUAL")
    print("="*50)
    
    data_path = "data"
    categories = {}
    
    # Buscar en subdirectorios principales
    main_dirs = ['train', 'test', 'validation']
    
    for main_dir in main_dirs:
        main_path = os.path.join(data_path, main_dir)
        if os.path.exists(main_path):
            print(f"\n📁 Analizando {main_dir}:")
            
            for root, dirs, files in os.walk(main_path):
                # Buscar imágenes
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                
                if image_files:
                    rel_path = os.path.relpath(root, main_path)
                    if rel_path != '.':
                        category = rel_path.split('/')[-1]
                        
                        if category not in categories:
                            categories[category] = {'train': 0, 'test': 0, 'validation': 0}
                        
                        categories[category][main_dir] = len(image_files)
                        print(f"   {category}: {len(image_files)} imágenes")
    
    return categories

def create_balanced_dataset(categories):
    """Crear un dataset balanceado"""
    print(f"\n🔄 CREANDO DATASET BALANCEADO")
    print("="*50)
    
    # Crear directorios de salida
    output_dirs = {
        'train': 'data/balanced/train',
        'test': 'data/balanced/test', 
        'validation': 'data/balanced/validation'
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Definir categorías objetivo (simplificadas)
    target_categories = {
        'tops': ['tops', 'shirts', 'camisetas', 'blusas'],
        'dress': ['dress', 'vestidos', 'vestido'],
        'shoes': ['shoes', 'zapatos', 'sneakers'],
        'bags': ['bags', 'bolsos', 'mochilas'],
        'scarves': ['scarves', 'bufandas', 'pañuelos'],
        'jewelry': ['jewelry', 'joyas', 'accesorios'],
        'pants': ['pants', 'pantalones', 'jeans'],
        'general': ['general', 'fashion', 'moda', 'ropa']
    }
    
    # Contar imágenes disponibles por categoría
    available_images = {}
    for category, counts in categories.items():
        total_images = sum(counts.values())
        if total_images > 0:
            available_images[category] = total_images
    
    print(f"📊 Imágenes disponibles por categoría:")
    for cat, count in available_images.items():
        print(f"   {cat}: {count} imágenes")
    
    # Determinar tamaño objetivo por categoría (balanceado)
    min_images = min(available_images.values()) if available_images else 0
    target_per_category = min(100, min_images)  # Máximo 100 por categoría para balance
    
    print(f"\n🎯 Objetivo: {target_per_category} imágenes por categoría")
    
    # Crear dataset balanceado
    balanced_stats = {}
    
    for target_cat, synonyms in target_categories.items():
        print(f"\n📂 Procesando categoría: {target_cat}")
        
        # Buscar imágenes que coincidan con sinónimos
        matching_images = []
        for category, counts in categories.items():
            if any(synonym in category.lower() for synonym in synonyms):
                # Encontrar imágenes reales
                for main_dir in ['train', 'test', 'validation']:
                    if counts.get(main_dir, 0) > 0:
                        source_dir = os.path.join('data', main_dir, category)
                        if os.path.exists(source_dir):
                            for file in os.listdir(source_dir):
                                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                                    matching_images.append({
                                        'source': os.path.join(source_dir, file),
                                        'category': target_cat,
                                        'original_category': category
                                    })
        
        # Seleccionar imágenes aleatoriamente
        selected_images = random.sample(matching_images, min(target_per_category, len(matching_images)))
        
        print(f"   Encontradas: {len(matching_images)} imágenes")
        print(f"   Seleccionadas: {len(selected_images)} imágenes")
        
        # Copiar imágenes a directorios balanceados
        balanced_stats[target_cat] = {'train': 0, 'test': 0, 'validation': 0}
        
        for i, img_info in enumerate(selected_images):
            # Determinar split (70% train, 15% test, 15% validation)
            if i < len(selected_images) * 0.7:
                split = 'train'
            elif i < len(selected_images) * 0.85:
                split = 'test'
            else:
                split = 'validation'
            
            # Crear directorio de destino
            dest_dir = os.path.join(output_dirs[split], target_cat)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copiar imagen
            dest_file = os.path.join(dest_dir, f"{target_cat}_{i:04d}.jpg")
            try:
                shutil.copy2(img_info['source'], dest_file)
                balanced_stats[target_cat][split] += 1
            except Exception as e:
                print(f"   ⚠️ Error copiando {img_info['source']}: {e}")
    
    return balanced_stats

def create_data_augmentation():
    """Crear data augmentation para categorías minoritarias"""
    print(f"\n🎨 CREANDO DATA AUGMENTATION")
    print("="*50)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    
    # Configurar data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Aplicar augmentation a categorías con pocas imágenes
    balanced_path = "data/balanced/train"
    
    for category in os.listdir(balanced_path):
        category_path = os.path.join(balanced_path, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) < 50:  # Si tiene menos de 50 imágenes
                print(f"   🔄 Aplicando augmentation a {category} ({len(images)} imágenes)")
                
                # Cargar imágenes
                for img_file in images[:10]:  # Solo las primeras 10 para augmentation
                    img_path = os.path.join(category_path, img_file)
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Generar imágenes aumentadas
                        aug_iter = datagen.flow(img_array, batch_size=1)
                        for i in range(5):  # Generar 5 variaciones
                            aug_img = next(aug_iter)[0].astype(np.uint8)
                            aug_img_pil = Image.fromarray(aug_img)
                            
                            # Guardar imagen aumentada
                            aug_filename = f"{img_file.split('.')[0]}_aug_{i}.jpg"
                            aug_path = os.path.join(category_path, aug_filename)
                            aug_img_pil.save(aug_path, 'JPEG')
                            
                    except Exception as e:
                        print(f"   ⚠️ Error en augmentation de {img_file}: {e}")

def generate_training_script():
    """Generar script de entrenamiento para el dataset balanceado"""
    print(f"\n📝 GENERANDO SCRIPT DE ENTRENAMIENTO")
    print("="*50)
    
    training_script = '''#!/usr/bin/env python3
"""
Script de entrenamiento para dataset balanceado
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

def create_model(num_classes=8):
    """Crear modelo MobileNetV2 mejorado"""
    # Base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model():
    """Entrenar el modelo"""
    # Configuración
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = (224, 224)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    # Train generator
    train_generator = train_datagen.flow_from_directory(
        'data/balanced/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        'data/balanced/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create model
    model = create_model(num_classes=len(train_generator.class_indices))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint('data/logs/training/balanced_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento con dataset balanceado...")
    model, history = train_model()
    print("✅ Entrenamiento completado!")
'''
    
    with open('train_balanced_model.py', 'w') as f:
        f.write(training_script)
    
    print("✅ Script de entrenamiento generado: train_balanced_model.py")

def main():
    """Función principal"""
    print("🔧 CORRIGIENDO ESTRUCTURA DEL DATASET")
    print("="*60)
    
    # Analizar estructura actual
    categories = analyze_current_structure()
    
    # Crear dataset balanceado
    balanced_stats = create_balanced_dataset(categories)
    
    # Mostrar estadísticas finales
    print(f"\n📊 ESTADÍSTICAS FINALES")
    print("="*50)
    
    total_images = 0
    for category, splits in balanced_stats.items():
        category_total = sum(splits.values())
        total_images += category_total
        print(f"{category}: {category_total} imágenes")
        for split, count in splits.items():
            print(f"  {split}: {count}")
    
    print(f"\nTotal: {total_images} imágenes")
    
    # Crear data augmentation
    create_data_augmentation()
    
    # Generar script de entrenamiento
    generate_training_script()
    
    # Guardar configuración
    config = {
        'balanced_stats': balanced_stats,
        'total_images': total_images,
        'categories': list(balanced_stats.keys())
    }
    
    with open('dataset_analysis/balanced_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Dataset balanceado creado exitosamente!")
    print(f"📁 Ubicación: data/balanced/")
    print(f"🚀 Próximo paso: Ejecutar train_balanced_model.py")

if __name__ == "__main__":
    main()
