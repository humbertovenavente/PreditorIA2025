#!/bin/bash

# Fashion Trend Analysis App - Setup Script
# Configuración automática del entorno

set -e  # Exit on any error

echo " Configurando Fashion Trend Analysis App..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar Python
print_status "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 no está instalado"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION encontrado"

# Verificar pip
print_status "Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 no está instalado"
    exit 1
fi
print_success "pip3 encontrado"

# Crear directorio de logs
print_status "Creando directorio de logs..."
mkdir -p logs
print_success "Directorio de logs creado"

# Crear directorio de uploads
print_status "Creando directorio de uploads..."
mkdir -p static/images/uploads
print_success "Directorio de uploads creado"

# Verificar dependencias del clustering
print_status "Verificando módulo de clustering..."
CLUSTERING_DIR="/home/jose/PreditorIA2025/reports"
if [ ! -d "$CLUSTERING_DIR" ]; then
    print_error "Directorio de clustering no encontrado: $CLUSTERING_DIR"
    print_warning "Ejecuta primero el módulo de clustering:"
    print_warning "cd /home/jose/PreditorIA2025"
    print_warning "python -m fashion_clustering.run_clustering --algo auto"
    exit 1
fi

# Verificar archivos necesarios
REQUIRED_FILES=(
    "embeddings.npy"
    "metadata.csv"
    "cluster_assignments.csv"
    "cluster_stats.json"
)

print_status "Verificando archivos de clustering..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CLUSTERING_DIR/$file" ]; then
        print_error "Archivo requerido no encontrado: $file"
        exit 1
    fi
done
print_success "Todos los archivos de clustering encontrados"

# Instalar dependencias
print_status "Instalando dependencias..."
pip3 install -r requirements.txt
print_success "Dependencias instaladas"

# Hacer ejecutables los scripts
print_status "Configurando permisos..."
chmod +x run.py
chmod +x app.py
print_success "Permisos configurados"

# Verificar instalación
print_status "Verificando instalación..."
python3 -c "
import sys
sys.path.append('/home/jose/PreditorIA2025')
try:
    import fashion_clustering
    print(' Módulo fashion_clustering importado correctamente')
except ImportError as e:
    print(f' Error importando fashion_clustering: {e}')
    sys.exit(1)

try:
    import tensorflow as tf
    print(' TensorFlow importado correctamente')
except ImportError as e:
    print(f' Error importando TensorFlow: {e}')
    sys.exit(1)

try:
    import flask
    print(' Flask importado correctamente')
except ImportError as e:
    print(f' Error importando Flask: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Verificación completada exitosamente"
else
    print_error "Error en la verificación"
    exit 1
fi

# Mostrar información de uso
echo ""
echo " ¡Configuración completada exitosamente!"
echo ""
echo " Para ejecutar la aplicación:"
echo "   python3 run.py"
echo ""
echo " La aplicación estará disponible en:"
echo "   http://localhost:5000"
echo ""
echo " Estructura del proyecto:"
echo "   $(pwd)/"
echo "   ├── app.py              # Aplicación principal"
echo "   ├── run.py              # Script de inicio"
echo "   ├── templates/          # Plantillas HTML"
echo "   ├── static/            # Archivos estáticos"
echo "   ├── utils/             # Utilidades"
echo "   └── logs/              # Archivos de log"
echo ""
echo " Para desarrollo:"
echo "   export FLASK_ENV=development"
echo "   python3 app.py"
echo ""
echo " Para más información, consulta README.md"
echo ""
print_success "¡Setup completado! 🚀"


