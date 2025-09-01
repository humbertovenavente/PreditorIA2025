#!/bin/bash

# Script para activar el entorno virtual de Python
# Este script debe ser ejecutado antes de usar el sistema de scraping

echo "Activando entorno virtual de Python..."
echo " Entorno: venv/"

# Verificar que el entorno virtual existe
if [ ! -d "venv" ]; then
    echo " Error: El entorno virtual 'venv' no existe"
    echo " Ejecuta: python3 -m venv venv"
    exit 1
fi

# Activar el entorno virtual
source venv/bin/activate

# Verificar que se activó correctamente
if [ -n "$VIRTUAL_ENV" ]; then
    echo " Entorno virtual activado: $VIRTUAL_ENV"
    echo " Python: $(which python)"
    echo " Pip: $(which pip)"
    echo ""
    echo " Ahora puedes ejecutar:"
    echo "   • python test_scraping_system.py"
    echo "   • python massive_guatemala_scraping.py"
    echo "   • python monitor_scraping.py"
    echo ""
    echo " Para desactivar: deactivate"
    echo ""
    
    # Cambiar el prompt para indicar que está activo
    export PS1="(venv) $PS1"
    
    # Ejecutar bash con el entorno activado
    bash
else
    echo " Error: No se pudo activar el entorno virtual"
    exit 1
fi

