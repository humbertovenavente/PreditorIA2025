#!/usr/bin/env python3
"""
Script para analizar la distribución del dataset y identificar problemas
"""

import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset_structure():
    """Analizar la estructura del dataset"""
    print("🔍 ANÁLISIS DE ESTRUCTURA DEL DATASET")
    print("="*50)
    
    # Buscar directorios de datos
    data_dirs = ['data', 'dataset', 'images', 'fashion_data']
    dataset_path = None
    
    for dir_name in data_dirs:
        if os.path.exists(dir_name):
            dataset_path = dir_name
            break
    
    if not dataset_path:
        print("❌ No se encontró directorio de dataset")
        return None
    
    print(f"📁 Dataset encontrado en: {dataset_path}")
    
    # Analizar estructura
    categories = []
    total_images = 0
    
    for root, dirs, files in os.walk(dataset_path):
        # Buscar imágenes en subdirectorios
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        if image_files:
            # Extraer categoría del path
            rel_path = os.path.relpath(root, dataset_path)
            if rel_path != '.':
                category = rel_path.split('/')[-1]  # Último directorio
                categories.append(category)
                total_images += len(image_files)
                print(f"   📂 {category}: {len(image_files)} imágenes")
    
    print(f"\n📊 RESUMEN:")
    print(f"   Total de categorías: {len(set(categories))}")
    print(f"   Total de imágenes: {total_images}")
    
    return categories, total_images

def analyze_category_distribution(categories):
    """Analizar distribución de categorías"""
    print(f"\n📊 ANÁLISIS DE DISTRIBUCIÓN")
    print("="*50)
    
    category_counts = Counter(categories)
    
    # Mostrar distribución
    print("Distribución por categoría:")
    for category, count in category_counts.most_common():
        percentage = (count / len(categories)) * 100
        print(f"   {category}: {count} imágenes ({percentage:.1f}%)")
    
    # Identificar problemas
    print(f"\n⚠️ PROBLEMAS IDENTIFICADOS:")
    
    # Categorías sobre-representadas
    max_count = max(category_counts.values())
    over_represented = [cat for cat, count in category_counts.items() if count > max_count * 0.4]
    if over_represented:
        print(f"   🔴 Categorías sobre-representadas: {over_represented}")
    
    # Categorías sub-representadas
    min_count = min(category_counts.values())
    under_represented = [cat for cat, count in category_counts.items() if count < min_count * 2]
    if under_represented:
        print(f"   🔵 Categorías sub-representadas: {under_represented}")
    
    # Balance general
    counts = list(category_counts.values())
    balance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"   📊 Ratio de balance: {balance_ratio:.1f}:1")
    
    if balance_ratio > 5:
        print(f"   ⚠️ Dataset muy desbalanceado (ratio > 5:1)")
    elif balance_ratio > 2:
        print(f"   ⚠️ Dataset moderadamente desbalanceado (ratio > 2:1)")
    else:
        print(f"   ✅ Dataset bien balanceado")
    
    return category_counts

def create_visualizations(category_counts):
    """Crear visualizaciones del dataset"""
    print(f"\n📊 CREANDO VISUALIZACIONES")
    print("="*50)
    
    # Crear directorio para visualizaciones
    os.makedirs('dataset_analysis', exist_ok=True)
    
    # 1. Gráfico de barras
    plt.figure(figsize=(12, 6))
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    plt.bar(categories, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Distribución de Categorías en el Dataset')
    plt.xlabel('Categorías')
    plt.ylabel('Número de Imágenes')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis/category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de pastel
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(categories)))
    wedges, texts, autotexts = plt.pie(counts, labels=categories, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    
    plt.title('Distribución Porcentual de Categorías')
    plt.axis('equal')
    
    # Mejorar legibilidad
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis/category_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizaciones guardadas en 'dataset_analysis/'")

def generate_recommendations(category_counts):
    """Generar recomendaciones de mejora"""
    print(f"\n💡 RECOMENDACIONES DE MEJORA")
    print("="*50)
    
    counts = list(category_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    balance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("1. 📊 BALANCEO DEL DATASET:")
    if balance_ratio > 5:
        print("   🔴 CRÍTICO: Dataset extremadamente desbalanceado")
        print("   - Eliminar exceso de categorías sobre-representadas")
        print("   - Duplicar imágenes de categorías sub-representadas")
        print("   - Aplicar data augmentation agresivo")
    elif balance_ratio > 2:
        print("   🟡 MODERADO: Dataset desbalanceado")
        print("   - Aplicar data augmentation a categorías minoritarias")
        print("   - Considerar técnicas de balanceo")
    else:
        print("   ✅ BUENO: Dataset relativamente balanceado")
    
    print("\n2. 🎯 CATEGORÍAS ESPECÍFICAS:")
    
    # Identificar categorías problemáticas
    for category, count in category_counts.items():
        percentage = (count / sum(counts)) * 100
        if percentage > 40:
            print(f"   🔴 {category}: {count} imágenes ({percentage:.1f}%) - SOBRE-REPRESENTADA")
        elif percentage < 5:
            print(f"   🔵 {category}: {count} imágenes ({percentage:.1f}%) - SUB-REPRESENTADA")
        else:
            print(f"   ✅ {category}: {count} imágenes ({percentage:.1f}%) - BALANCEADA")
    
    print("\n3. 🔄 ACCIONES RECOMENDADAS:")
    print("   - Revisar etiquetas del dataset")
    print("   - Implementar data augmentation")
    print("   - Balancear clases antes del entrenamiento")
    print("   - Usar técnicas de balanceo (SMOTE, etc.)")
    print("   - Considerar transfer learning")

def main():
    """Función principal"""
    print("🔍 ANÁLISIS COMPLETO DEL DATASET")
    print("="*60)
    
    # Analizar estructura
    result = analyze_dataset_structure()
    if result is None:
        return
    
    categories, total_images = result
    
    # Analizar distribución
    category_counts = analyze_category_distribution(categories)
    
    # Crear visualizaciones
    create_visualizations(category_counts)
    
    # Generar recomendaciones
    generate_recommendations(category_counts)
    
    # Guardar análisis
    analysis_data = {
        'total_images': total_images,
        'total_categories': len(set(categories)),
        'category_distribution': dict(category_counts),
        'balance_ratio': max(category_counts.values()) / min(category_counts.values()) if min(category_counts.values()) > 0 else float('inf')
    }
    
    with open('dataset_analysis/dataset_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\n💾 Análisis guardado en 'dataset_analysis/dataset_analysis.json'")
    print(f"🎯 Próximo paso: Implementar mejoras según recomendaciones")

if __name__ == "__main__":
    main()
