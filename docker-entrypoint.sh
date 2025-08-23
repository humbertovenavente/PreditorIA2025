#!/bin/bash

# Script de entrada para el contenedor Docker
set -e

# Iniciar Xvfb para el display virtual (necesario para Chrome headless)
Xvfb :99 -screen 0 1920x1080x24 &

# Esperar a que Xvfb esté listo
sleep 2

# Verificar variables de entorno necesarias
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Advertencia: GOOGLE_CLOUD_PROJECT no está configurado"
fi

if [ -z "$GCS_BUCKET_NAME" ]; then
    echo "Advertencia: GCS_BUCKET_NAME no está configurado"
fi

# Mostrar información del contenedor
echo "=== Fashion Image Scraper Container ==="
echo "Python version: $(python --version)"
echo "Chrome version: $(google-chrome --version)"
echo "ChromeDriver version: $(chromedriver --version)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "======================================"

# Ejecutar el comando principal
exec "$@"
