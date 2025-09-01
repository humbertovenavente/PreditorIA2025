#!/bin/bash

echo " INSTALANDO DEPENDENCIAS PARA PREPARACIÓN DEL DATASET"
echo "============================================================"
echo "Librerías necesarias:"
echo "  • scikit-learn (división del dataset)"
echo "  • opencv-python (procesamiento de imágenes)"
echo "  • pillow (manipulación de imágenes)"
echo "  • numpy (operaciones numéricas)"
echo "  • matplotlib (reportes visuales)"
echo "  • tqdm (barras de progreso)"
echo "============================================================"

# Activar entorno virtual si existe
if [ -"activate_venv.sh" ]; then
    echo "🔧 Activando entorno virtual..."
    source activate_venv.sh
else
    echo "  No se encontró activate_venv.sh, usando Python del sistema"
fi

# Verificar Python
echo " Verificando versión de Python..."
python3 --version

# Instalar dependencias una por una
echo ""
echo " Instalando scikit-learn..."
pip install scikit-learn

echo ""
echo "Instalando opencv-python..."
pip install opencv-python

echo ""
echo "Instalando pillow..."
pip install pillow

echo ""
echo " Instalando numpy..."
pip install numpy

echo ""
echo " Instalando matplotlib..."
pip install matplotlib

echo ""
echo " Instalando tqdm..."
pip install tqdm

# Verificar instalación
echo ""
echo "Verificando instalación..."
python3 -c "
try:
    import sklearn
    print(' scikit-learn instalado correctamente')
except ImportError:
    print(' scikit-learn NO instalado')

try:
    import cv2
    print(' opencv-python instalado correctamente')
except ImportError:
    print(' opencv-python NO instalado')

try:
    import PIL
    print(' pillow instalado correctamente')
except ImportError:
    print(' pillow NO instalado')

try:
    import numpy
    print(' numpy instalado correctamente')
except ImportError:
    print(' numpy NO instalado')

try:
    import matplotlib
    print(' matplotlib instalado correctamente')
except ImportError:
    print(' matplotlib NO instalado')

try:
    import tqdm
    print(' tqdm instalado correctamente')
except ImportError:
    print(' tqdm NO instalado')
"

echo ""
echo "INSTALACIÓN DE DEPENDENCIAS COMPLETADA"
echo " Ahora puedes ejecutar: python3 prepare_dataset.py"

