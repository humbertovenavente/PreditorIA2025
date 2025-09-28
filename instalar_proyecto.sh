#!/bin/bash

# Script de instalación automática para PreditorIA2025
# Instala todas las dependencias y configura el proyecto

echo "🚀 INSTALACIÓN AUTOMÁTICA - PreditorIA2025"
echo "=========================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar progreso
show_progress() {
    echo -e "${BLUE}📦 $1${NC}"
}

# Función para mostrar éxito
show_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Función para mostrar advertencia
show_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Función para mostrar error
show_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar si estamos en el directorio correcto
if [ ! -f "requirements.txt" ]; then
    show_error "No se encontró requirements.txt. Ejecuta este script desde el directorio raíz del proyecto."
    exit 1
fi

# Paso 1: Verificar Python
show_progress "Verificando Python..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    show_success "Python encontrado: $python_version"
else
    show_error "Python 3 no encontrado. Instala Python 3.11 o superior."
    exit 1
fi

# Paso 2: Crear entorno virtual
show_progress "Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    show_success "Entorno virtual creado"
else
    show_warning "Entorno virtual ya existe"
fi

# Paso 3: Activar entorno virtual
show_progress "Activando entorno virtual..."
source venv/bin/activate
if [[ "$VIRTUAL_ENV" != "" ]]; then
    show_success "Entorno virtual activado: $VIRTUAL_ENV"
else
    show_error "No se pudo activar el entorno virtual"
    exit 1
fi

# Paso 4: Actualizar pip
show_progress "Actualizando pip..."
pip install --upgrade pip
show_success "pip actualizado"

# Paso 5: Instalar dependencias principales
show_progress "Instalando dependencias principales..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    show_success "Dependencias principales instaladas"
else
    show_error "Error instalando dependencias principales"
    exit 1
fi

# Paso 6: Instalar dependencias de la app web
show_progress "Instalando dependencias de la aplicación web..."
if [ -f "fashion_trend_app/requirements.txt" ]; then
    pip install -r fashion_trend_app/requirements.txt
    show_success "Dependencias de la app web instaladas"
else
    show_warning "No se encontró fashion_trend_app/requirements.txt"
fi

# Paso 7: Crear directorios necesarios
show_progress "Creando directorios necesarios..."
directories=("data/images" "data/processed/train" "data/processed/validation" "data/processed/test" "data/logs" "data/embeddings" "data/reduced" "data/clustering_results" "data/analysis" "clustering_results" "credentials")
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done
show_success "Directorios creados"

# Paso 8: Crear archivo de configuración
show_progress "Creando archivo de configuración..."
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        show_success "Archivo .env creado desde env.example"
        show_warning "Recuerda editar .env con tus configuraciones"
    else
        show_warning "No se encontró env.example"
    fi
else
    show_warning "Archivo .env ya existe"
fi

# Paso 9: Descargar modelo pre-entrenado
show_progress "Descargando modelo pre-entrenado..."
mkdir -p data/logs/training
python -c "
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import os

print('Descargando MobileNetV2 pre-entrenado...')
model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Guardar modelo
model_path = 'data/logs/training/mobilenet_v2_pretrained.h5'
model.save(model_path)
print(f'Modelo guardado en: {model_path}')
print(f'Tamaño del archivo: {os.path.getsize(model_path) / (1024*1024):.1f} MB')
" 2>/dev/null

if [ $? -eq 0 ]; then
    show_success "Modelo pre-entrenado descargado"
else
    show_warning "Error descargando modelo pre-entrenado (se puede descargar manualmente después)"
fi

# Paso 10: Verificar instalación
show_progress "Verificando instalación..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
except ImportError as e:
    print(f'Error TensorFlow: {e}')

try:
    import sklearn
    print(f'scikit-learn: {sklearn.__version__}')
except ImportError as e:
    print(f'Error scikit-learn: {e}')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'Error OpenCV: {e}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'Error NumPy: {e}')

try:
    import pandas as pd
    print(f'Pandas: {pd.__version__}')
except ImportError as e:
    print(f'Error Pandas: {e}')

try:
    import matplotlib
    print(f'Matplotlib: {matplotlib.__version__}')
except ImportError as e:
    print(f'Error Matplotlib: {e}')

try:
    import flask
    print(f'Flask: {flask.__version__}')
except ImportError as e:
    print(f'Error Flask: {e}')
"

# Paso 11: Mostrar resumen
echo ""
echo "🎉 INSTALACIÓN COMPLETADA"
echo "========================="
echo ""
echo "📁 Estructura del proyecto:"
echo "   ├── venv/                    # Entorno virtual"
echo "   ├── data/                    # Datos del proyecto"
echo "   ├── fashion_clustering/      # Módulo de clustering"
echo "   ├── fashion_trend_app/       # Aplicación web"
echo "   ├── .env                     # Configuración"
echo "   └── requirements.txt         # Dependencias"
echo ""
echo "🚀 PRÓXIMOS PASOS:"
echo "1. Activar entorno virtual:"
echo "   source venv/bin/activate"
echo ""
echo "2. Verificar instalación:"
echo "   ./verificar_instalacion.sh"
echo ""
echo "3. Configurar variables (opcional):"
echo "   nano .env"
echo ""
echo "4. Ejecutar clustering:"
echo "   cd fashion_clustering"
echo "   python run_clustering.py"
echo ""
echo "5. Iniciar aplicación web:"
echo "   cd fashion_trend_app"
echo "   python app_with_progress.py"
echo ""
echo "6. Ver documentación completa:"
echo "   cat ANEXO_A_REPLICACION_PROYECTO.md"
echo ""
echo "📚 Para más información, consulta: ANEXO_A_REPLICACION_PROYECTO.md"
echo ""
show_success "¡Instalación completada exitosamente!"
