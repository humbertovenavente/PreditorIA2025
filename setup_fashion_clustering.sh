#!/bin/bash
"""
Script de instalación y verificación para Fashion Clustering
"""

echo "🚀 Configurando Fashion Clustering..."

# Verificar Python
echo "📋 Verificando Python..."
python3 --version || { echo "❌ Python 3 no encontrado"; exit 1; }

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Verificar instalación
echo "✅ Verificando instalación..."
python3 -c "
import tensorflow as tf
import sklearn
import matplotlib
import seaborn
import umap
import yaml
print('✅ Todas las dependencias instaladas correctamente')
"

# Ejecutar tests
echo "🧪 Ejecutando tests..."
python3 -m pytest tests/test_fashion_clustering.py -v

# Crear directorio de reportes
echo "📁 Creando directorio de reportes..."
mkdir -p reports

# Verificar estructura del proyecto
echo "📋 Verificando estructura del proyecto..."
if [ -d "fashion_clustering" ] && [ -f "fashion_clustering/config.yaml" ]; then
    echo "✅ Estructura del proyecto correcta"
else
    echo "❌ Estructura del proyecto incorrecta"
    exit 1
fi

echo "🎉 ¡Fashion Clustering configurado exitosamente!"
echo ""
echo "📚 PRÓXIMOS PASOS:"
echo "1. Activa el entorno virtual: source venv/bin/activate"
echo "2. Ejecuta el ejemplo: python3 example_fashion_clustering.py"
echo "3. O usa los scripts individuales:"
echo "   - python3 -m fashion_clustering.extract_embeddings --help"
echo "   - python3 -m fashion_clustering.run_clustering --help"
echo "   - python3 -m fashion_clustering.visualize --help"
echo ""
echo "📖 Para más información, consulta README_FASHION_CLUSTERING.md"


