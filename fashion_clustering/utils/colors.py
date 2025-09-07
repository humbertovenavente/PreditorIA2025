"""
Módulo para análisis de colores dominantes
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Paleta de colores básica en español
COLOR_NAMES = {
    'red': 'rojo',
    'orange': 'naranja', 
    'yellow': 'amarillo',
    'green': 'verde',
    'blue': 'azul',
    'purple': 'morado',
    'pink': 'rosa',
    'brown': 'marrón',
    'gray': 'gris',
    'black': 'negro',
    'white': 'blanco',
    'beige': 'beige',
    'navy': 'azul marino',
    'teal': 'verde azulado',
    'coral': 'coral',
    'gold': 'dorado',
    'silver': 'plateado',
    'burgundy': 'burdeos',
    'cream': 'crema',
    'khaki': 'caqui'
}

def extract_dominant_colors(image_path: str, 
                          n_colors: int = 5,
                          sample_size: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extrae colores dominantes de una imagen, enfocándose en el centro (donde está la ropa)
    
    Args:
        image_path: Ruta a la imagen
        n_colors: Número de colores dominantes a extraer
        sample_size: Tamaño de muestra para reducir píxeles (5x5 = 25 píxeles por muestra)
    
    Returns:
        Lista de tuplas RGB de colores dominantes
    """
    try:
        # Cargar imagen
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # Redimensionar para acelerar el procesamiento
        img = img.resize((224, 224))
        
        # Enfocarse en el centro de la imagen (donde está la ropa)
        # Tomar una región central más pequeña para evitar el fondo
        center_x, center_y = img.width // 2, img.height // 2
        crop_size = min(img.width, img.height) // 2  # Usar la mitad del tamaño
        
        # Calcular región de recorte centrada
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(img.width, center_x + crop_size // 2)
        bottom = min(img.height, center_y + crop_size // 2)
        
        # Recortar imagen al centro
        cropped_img = img.crop((left, top, right, bottom))
        
        # Muestrear píxeles del área recortada (cada sample_size píxeles)
        pixels = []
        for y in range(0, cropped_img.height, sample_size):
            for x in range(0, cropped_img.width, sample_size):
                pixel = cropped_img.getpixel((x, y))
                pixels.append(pixel)
        
        pixels = np.array(pixels)
        
        # Filtrar píxeles muy oscuros (fondo) y muy claros (reflejos)
        # Priorizar colores saturados (ropa) sobre colores neutros (fondo)
        filtered_pixels = []
        for pixel in pixels:
            r, g, b = pixel
            brightness = (r + g + b) / 3
            
            # Calcular saturación (diferencia entre el canal más alto y más bajo)
            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            saturation = max_channel - min_channel if max_channel > 0 else 0
            
            # Priorizar píxeles con buena saturación y brillo medio
            # Esto ayuda a detectar colores de ropa en lugar del fondo
            if 40 < brightness < 200 and saturation > 30:
                filtered_pixels.append(pixel)
            elif 60 < brightness < 180:  # Fallback para píxeles menos saturados pero con buen brillo
                filtered_pixels.append(pixel)
        
        if len(filtered_pixels) < n_colors:
            # Si no hay suficientes píxeles filtrados, usar todos
            filtered_pixels = pixels.tolist()
        
        filtered_pixels = np.array(filtered_pixels)
        
        # Clustering de colores con K-Means
        kmeans = KMeans(n_clusters=min(n_colors, len(filtered_pixels)), random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Obtener colores dominantes
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Ordenar por frecuencia
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        sorted_indices = np.argsort(color_counts)[::-1]
        dominant_colors = dominant_colors[sorted_indices]
        
        return [tuple(color) for color in dominant_colors]
        
    except Exception as e:
        logger.error(f"Error extrayendo colores de {image_path}: {e}")
        return [(0, 0, 0)] * n_colors

def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Convierte RGB a nombre de color aproximado con mayor precisión
    
    Args:
        rgb: Tupla RGB (r, g, b)
    
    Returns:
        Nombre del color en español
    """
    r, g, b = rgb
    
    # Calcular valores HSV para mejor clasificación
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Calcular saturación y valor
    saturation = diff / max_val if max_val > 0 else 0
    value = max_val / 255.0
    
    # Calcular matiz (hue)
    if diff == 0:
        hue = 0  # Gris
    elif max_val == r:
        hue = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        hue = 60 * ((b - r) / diff) + 120
    else:
        hue = 60 * ((r - g) / diff) + 240
    
    # Clasificación completa de colores basada en HSV
    if value < 0.15:
        return 'negro'
    elif value > 0.85 and saturation < 0.1:
        return 'blanco'
    elif saturation < 0.15:
        if value < 0.3:
            return 'negro'
        elif value < 0.5:
            return 'gris muy oscuro'
        elif value < 0.7:
            return 'gris oscuro'
        elif value < 0.85:
            return 'gris'
        else:
            return 'gris claro'
    else:
        # Colores saturados - Clasificación detallada
        if 0 <= hue < 15 or 345 <= hue <= 360:
            # ROJOS
            if value > 0.85:
                return 'rojo muy brillante'
            elif value > 0.7:
                return 'rojo brillante'
            elif value > 0.5:
                return 'rojo'
            elif value > 0.3:
                return 'rojo oscuro'
            else:
                return 'rojo muy oscuro'
        elif 15 <= hue < 30:
            # ROJO-NARANJA
            if value > 0.85:
                return 'rojo-naranja brillante'
            elif value > 0.7:
                return 'rojo-naranja'
            elif value > 0.5:
                return 'rojo-naranja oscuro'
            else:
                return 'rojo-naranja muy oscuro'
        elif 30 <= hue < 45:
            # NARANJAS
            if value > 0.85:
                return 'naranja muy brillante'
            elif value > 0.7:
                return 'naranja brillante'
            elif value > 0.5:
                return 'naranja'
            elif value > 0.3:
                return 'naranja oscuro'
            else:
                return 'naranja muy oscuro'
        elif 45 <= hue < 60:
            # NARANJA-AMARILLO
            if value > 0.85:
                return 'naranja-amarillo brillante'
            elif value > 0.7:
                return 'naranja-amarillo'
            elif value > 0.5:
                return 'naranja-amarillo oscuro'
            else:
                return 'naranja-amarillo muy oscuro'
        elif 60 <= hue < 75:
            # AMARILLOS
            if value > 0.85:
                return 'amarillo muy brillante'
            elif value > 0.7:
                return 'amarillo brillante'
            elif value > 0.5:
                return 'amarillo'
            elif value > 0.3:
                return 'amarillo oscuro'
            else:
                return 'amarillo muy oscuro'
        elif 75 <= hue < 90:
            # AMARILLO-VERDE
            if value > 0.85:
                return 'amarillo-verde brillante'
            elif value > 0.7:
                return 'amarillo-verde'
            elif value > 0.5:
                return 'amarillo-verde oscuro'
            else:
                return 'amarillo-verde muy oscuro'
        elif 90 <= hue < 105:
            # VERDE-AMARILLO
            if value > 0.85:
                return 'verde-amarillo brillante'
            elif value > 0.7:
                return 'verde-amarillo'
            elif value > 0.5:
                return 'verde-amarillo oscuro'
            else:
                return 'verde-amarillo muy oscuro'
        elif 105 <= hue < 120:
            # VERDES LIMA
            if value > 0.85:
                return 'verde lima brillante'
            elif value > 0.7:
                return 'verde lima'
            elif value > 0.5:
                return 'verde lima oscuro'
            else:
                return 'verde lima muy oscuro'
        elif 120 <= hue < 135:
            # VERDES
            if value > 0.85:
                return 'verde muy brillante'
            elif value > 0.7:
                return 'verde brillante'
            elif value > 0.5:
                return 'verde'
            elif value > 0.3:
                return 'verde oscuro'
            else:
                return 'verde muy oscuro'
        elif 135 <= hue < 150:
            # VERDE-AZUL
            if value > 0.85:
                return 'verde-azul brillante'
            elif value > 0.7:
                return 'verde-azul'
            elif value > 0.5:
                return 'verde-azul oscuro'
            else:
                return 'verde-azul muy oscuro'
        elif 150 <= hue < 165:
            # AZUL-VERDE
            if value > 0.85:
                return 'azul-verde brillante'
            elif value > 0.7:
                return 'azul-verde'
            elif value > 0.5:
                return 'azul-verde oscuro'
            else:
                return 'azul-verde muy oscuro'
        elif 165 <= hue < 180:
            # CIAN
            if value > 0.85:
                return 'cian muy brillante'
            elif value > 0.7:
                return 'cian brillante'
            elif value > 0.5:
                return 'cian'
            elif value > 0.3:
                return 'cian oscuro'
            else:
                return 'cian muy oscuro'
        elif 180 <= hue < 195:
            # AZUL-CIAN
            if value > 0.85:
                return 'azul-cian brillante'
            elif value > 0.7:
                return 'azul-cian'
            elif value > 0.5:
                return 'azul-cian oscuro'
            else:
                return 'azul-cian muy oscuro'
        elif 195 <= hue < 210:
            # AZULES
            if value > 0.85:
                return 'azul muy brillante'
            elif value > 0.7:
                return 'azul brillante'
            elif value > 0.5:
                return 'azul'
            elif value > 0.3:
                return 'azul oscuro'
            else:
                return 'azul muy oscuro'
        elif 210 <= hue < 225:
            # AZUL-MORADO
            if value > 0.85:
                return 'azul-morado brillante'
            elif value > 0.7:
                return 'azul-morado'
            elif value > 0.5:
                return 'azul-morado oscuro'
            else:
                return 'azul-morado muy oscuro'
        elif 225 <= hue < 240:
            # AZUL MARINO
            if value > 0.85:
                return 'azul marino brillante'
            elif value > 0.7:
                return 'azul marino'
            elif value > 0.5:
                return 'azul marino oscuro'
            else:
                return 'azul marino muy oscuro'
        elif 240 <= hue < 255:
            # MORADO-AZUL
            if value > 0.85:
                return 'morado-azul brillante'
            elif value > 0.7:
                return 'morado-azul'
            elif value > 0.5:
                return 'morado-azul oscuro'
            else:
                return 'morado-azul muy oscuro'
        elif 255 <= hue < 270:
            # MORADOS
            if value > 0.85:
                return 'morado muy brillante'
            elif value > 0.7:
                return 'morado brillante'
            elif value > 0.5:
                return 'morado'
            elif value > 0.3:
                return 'morado oscuro'
            else:
                return 'morado muy oscuro'
        elif 270 <= hue < 285:
            # MORADO-ROSA
            if value > 0.85:
                return 'morado-rosa brillante'
            elif value > 0.7:
                return 'morado-rosa'
            elif value > 0.5:
                return 'morado-rosa oscuro'
            else:
                return 'morado-rosa muy oscuro'
        elif 285 <= hue < 300:
            # VIOLETA
            if value > 0.85:
                return 'violeta brillante'
            elif value > 0.7:
                return 'violeta'
            elif value > 0.5:
                return 'violeta oscuro'
            else:
                return 'violeta muy oscuro'
        elif 300 <= hue < 315:
            # ROSA-MORADO
            if value > 0.85:
                return 'rosa-morado brillante'
            elif value > 0.7:
                return 'rosa-morado'
            elif value > 0.5:
                return 'rosa-morado oscuro'
            else:
                return 'rosa-morado muy oscuro'
        elif 315 <= hue < 330:
            # ROSAS
            if value > 0.85:
                return 'rosa muy brillante'
            elif value > 0.7:
                return 'rosa brillante'
            elif value > 0.5:
                return 'rosa'
            elif value > 0.3:
                return 'rosa oscuro'
            else:
                return 'rosa muy oscuro'
        elif 330 <= hue < 345:
            # ROSA-ROJO
            if value > 0.85:
                return 'rosa-rojo brillante'
            elif value > 0.7:
                return 'rosa-rojo'
            elif value > 0.5:
                return 'rosa-rojo oscuro'
            else:
                return 'rosa-rojo muy oscuro'
        else:  # 345 <= hue < 360
            # ROJO-ROSA
            if value > 0.85:
                return 'rojo-rosa brillante'
            elif value > 0.7:
                return 'rojo-rosa'
            elif value > 0.5:
                return 'rojo-rosa oscuro'
            else:
                return 'rojo-rosa muy oscuro'

def get_color_names(dominant_colors: List[Tuple[int, int, int]]) -> List[str]:
    """
    Convierte lista de colores RGB a nombres
    
    Args:
        dominant_colors: Lista de tuplas RGB
    
    Returns:
        Lista de nombres de colores
    """
    return [rgb_to_color_name(rgb) for rgb in dominant_colors]

def analyze_cluster_colors(cluster_images: List[str],
                          n_colors: int = 5,
                          sample_size: int = 5) -> Dict[str, Any]:
    """
    Analiza colores dominantes de un cluster
    
    Args:
        cluster_images: Lista de rutas de imágenes del cluster
        n_colors: Número de colores por imagen
        sample_size: Tamaño de muestra para análisis
    
    Returns:
        Diccionario con análisis de colores
    """
    try:
        all_colors = []
        color_frequencies = {}
        
        # Extraer colores de todas las imágenes
        for img_path in cluster_images:
            try:
                colors = extract_dominant_colors(img_path, n_colors, sample_size)
                all_colors.extend(colors)
                
                # Contar colores por nombre
                color_names = get_color_names(colors)
                for color_name in color_names:
                    color_frequencies[color_name] = color_frequencies.get(color_name, 0) + 1
                    
            except Exception as e:
                logger.warning(f"Error analizando colores de {img_path}: {e}")
                continue
        
        if not all_colors:
            return {
                'dominant_colors': [],
                'color_names': [],
                'color_frequencies': {},
                'palette_description': 'Sin colores detectados'
            }
        
        # Encontrar colores más frecuentes
        sorted_colors = sorted(color_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_colors = [color for color, freq in sorted_colors[:n_colors]]
        
        # Crear descripción de paleta
        if top_colors:
            palette_description = ', '.join(top_colors)
        else:
            palette_description = 'Colores variados'
        
        return {
            'dominant_colors': all_colors[:n_colors],
            'color_names': top_colors,
            'color_frequencies': color_frequencies,
            'palette_description': palette_description
        }
        
    except Exception as e:
        logger.error(f"Error analizando colores del cluster: {e}")
        return {
            'dominant_colors': [],
            'color_names': [],
            'color_frequencies': {},
            'palette_description': 'Error en análisis'
        }

def create_color_palette_visualization(colors: List[Tuple[int, int, int]],
                                     color_names: List[str],
                                     save_path: str = "reports/color_palette.png",
                                     figsize: Tuple[int, int] = (12, 3)) -> None:
    """
    Crea visualización de paleta de colores
    
    Args:
        colors: Lista de colores RGB
        color_names: Lista de nombres de colores
        save_path: Ruta para guardar la imagen
        figsize: Tamaño de la figura
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear barras de colores
        bar_width = 1.0 / len(colors)
        for i, (color, name) in enumerate(zip(colors, color_names)):
            x = i * bar_width
            rect = Rectangle((x, 0), bar_width, 1, 
                           facecolor=[c/255.0 for c in color], 
                           edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Añadir etiqueta
            ax.text(x + bar_width/2, 0.5, name, 
                   ha='center', va='center', fontweight='bold',
                   color='white' if sum(color) < 400 else 'black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Paleta de Colores Dominantes', fontsize=14, fontweight='bold')
        
        # Guardar
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Paleta de colores guardada en {save_path}")
        
    except Exception as e:
        logger.error(f"Error creando visualización de paleta: {e}")

def get_color_temperature(rgb: Tuple[int, int, int]) -> str:
    """
    Determina la temperatura de color (cálido/frío)
    
    Args:
        rgb: Tupla RGB
    
    Returns:
        'cálido', 'frío' o 'neutral'
    """
    r, g, b = rgb
    
    # Calcular temperatura basada en proporciones
    total = r + g + b
    if total == 0:
        return 'neutral'
    
    r_ratio = r / total
    b_ratio = b / total
    
    if r_ratio > 0.4:
        return 'cálido'
    elif b_ratio > 0.4:
        return 'frío'
    else:
        return 'neutral'


