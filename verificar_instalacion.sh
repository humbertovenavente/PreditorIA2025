#!/bin/bash

# Script de verificación rápida para PreditorIA2025
# Verifica que todas las dependencias estén instaladas correctamente

echo "VERIFICACIÓN DE INSTALACIÓN - PreditorIA2025"
echo "=============================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para verificar comando
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN} $1: $(command -v $1)${NC}"
        return 0
    else
        echo -e "${RED} $1: No encontrado${NC}"
        return 1
    fi
}

# Función para verificar módulo Python
check_python_module() {
    if python -c "import $1" &> /dev/null; then
        version=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "instalado")
        echo -e "${GREEN}$1: $version${NC}"
        return 0
    else
        echo -e "${RED} $1: No instalado${NC}"
        return 1
    fi
}

# Verificar comandos del sistema
echo "VERIFICANDO COMANDOS DEL SISTEMA"
echo "-----------------------------------"
check_command python3
check_command pip3
check_command git
check_command docker
echo ""

# Verificar Python
echo "ERIFICANDO PYTHON"
echo "---------------------"
python_version=$(python3 --version 2>/dev/null || echo "No encontrado")
echo "Versión de Python: $python_version"

if [[ $python_version == *"3.11"* ]] || [[ $python_version == *"3.12"* ]]; then
    echo -e "${GREEN} Versión de Python compatible${NC}"
else
    echo -e "${YELLOW}  Se recomienda Python 3.11 o superior${NC}"
fi
echo ""

# Verificar entorno virtual
echo "VERIFICANDO ENTORNO VIRTUAL"
echo "-------------------------------"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN} Entorno virtual activo: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}  No hay entorno virtual activo${NC}"
    echo "   Para activar: source venv/bin/activate"
fi
echo ""

# Verificar dependencias Python críticas
echo " VERIFICANDO DEPENDENCIAS PYTHON"
echo "----------------------------------"
check_python_module tensorflow
check_python_module sklearn
check_python_module cv2
check_python_module numpy
check_python_module pandas
check_python_module matplotlib
check_python_module seaborn
check_python_module umap
check_python_module flask
check_python_module PIL
echo ""

# Verificar estructura del proyecto
echo " VERIFICANDO ESTRUCTURA DEL PROYECTO"
echo "--------------------------------------"
directories=("fashion_clustering" "fashion_trend_app" "data" "analysis" "scrapers" "storage")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN} $dir/${NC}"
    else
        echo -e "${RED} $dir/ - No encontrado${NC}"
    fi
done
echo ""

# Verificar archivos de configuración
echo "  VERIFICANDO ARCHIVOS DE CONFIGURACIÓN"
echo "----------------------------------------"
config_files=("requirements.txt" "fashion_clustering/config.yaml" "docker-compose.yml" "Dockerfile")
for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN} $file${NC}"
    else
        echo -e "${RED} $file - No encontrado${NC}"
    fi
done
echo ""

# Verificar directorios de datos
echo "VERIFICANDO DIRECTORIOS DE DATOS"
echo "----------------------------------"
data_dirs=("data/images" "data/processed" "data/logs" "clustering_results")
for dir in "${data_dirs[@]}"; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f | wc -l)
        echo -e "${GREEN}$dir/ ($file_count archivos)${NC}"
    else
        echo -e "${YELLOW}$dir/ - No encontrado (se creará automáticamente)${NC}"
    fi
done
echo ""

# Verificar modelos entrenados
echo "VERIFICANDO MODELOS ENTRENADOS"
echo "--------------------------------"
model_files=("data/logs/training/mobilenet_v2_final.h5" "fashion_trend_app/models/kmeans_model.pkl" "fashion_trend_app/models/pca_model.pkl")
for file in "${model_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}$file ($size)${NC}"
    else
        echo -e "${YELLOW} $file - No encontrado (se descargará/entrenará automáticamente)${NC}"
    fi
done
echo ""

# Verificar Docker
echo " VERIFICANDO DOCKER"
echo "---------------------"
if command -v docker &> /dev/null; then
    docker_version=$(docker --version)
    echo -e "${GREEN}Docker: $docker_version${NC}"
    
    if command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose --version)
        echo -e "${GREEN}Docker Compose: $compose_version${NC}"
    else
        echo -e "${YELLOW}  Docker Compose no encontrado${NC}"
    fi
else
    echo -e "${YELLOW} Docker no encontrado (opcional)${NC}"
fi
echo ""

# Resumen final
echo "📊 RESUMEN DE VERIFICACIÓN"
echo "=========================="

# Contar errores
errors=0
if ! command -v python3 &> /dev/null; then ((errors++)); fi
if ! python -c "import tensorflow" &> /dev/null; then ((errors++)); fi
if ! python -c "import sklearn" &> /dev/null; then ((errors++)); fi
if ! python -c "import cv2" &> /dev/null; then ((errors++)); fi

if [ $errors -eq 0 ]; then
    echo -e "${GREEN}¡Verificación exitosa! El sistema está listo para usar.${NC}"
    echo ""
    echo " PRÓXIMOS PASOS:"
    echo "1. Activar entorno virtual: source venv/bin/activate"
    echo "2. Ejecutar clustering: cd fashion_clustering && python run_clustering.py"
    echo "3. Iniciar app web: cd fashion_trend_app && python app_with_progress.py"
    echo "4. Ver documentación: cat ANEXO_A_REPLICACION_PROYECTO.md"
else
    echo -e "${RED} Se encontraron $errors errores críticos.${NC}"
    echo ""
    echo "SOLUCIONES:"
    echo "1. Instalar dependencias: pip install -r requirements.txt"
    echo "2. Activar entorno virtual: source venv/bin/activate"
    echo "3. Reinstalar dependencias: pip install --upgrade -r requirements.txt"
    echo "4. Ver documentación completa: cat ANEXO_A_REPLICACION_PROYECTO.md"
fi


