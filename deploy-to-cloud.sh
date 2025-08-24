#!/bin/bash

# Script para deployar el scraper a Google Cloud Run
# Esto evitará sobrecargar tu computadora local

set -e

echo "🚀 Deployando PreditorIA2025 a Google Cloud Run..."

# Variables
PROJECT_ID="preditoria2025-1755994382"
SERVICE_NAME="fashion-scraper"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

# Verificar que gcloud esté configurado
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: No hay una cuenta de Google Cloud autenticada"
    echo "Ejecuta: gcloud auth login"
    exit 1
fi

# Configurar proyecto
gcloud config set project $PROJECT_ID

# Habilitar APIs necesarias
echo "📋 Habilitando APIs de Google Cloud..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Construir y subir imagen Docker
echo "🔨 Construyendo imagen Docker..."
docker build -t $IMAGE_NAME .

echo "📤 Subiendo imagen a Container Registry..."
docker push $IMAGE_NAME

# Deployar a Cloud Run
echo "☁️ Deployando a Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --region $REGION \
    --platform managed \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 1 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
    --set-env-vars GCS_BUCKET_NAME=fashion-images-preditoria2025-1755994382-1755994702 \
    --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcp-key.json \
    --set-env-vars TARGET_IMAGES=1000 \
    --set-env-vars HEADLESS_BROWSER=true \
    --allow-unauthenticated

echo "✅ Deployment completado!"
echo "🌐 El scraper ahora se ejecuta en la nube y no sobrecargará tu computadora"

# Obtener URL del servicio
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "🔗 URL del servicio: $SERVICE_URL"

echo ""
echo "📊 Para monitorear el progreso:"
echo "   - Logs: gcloud run logs tail $SERVICE_NAME --region=$REGION"
echo "   - Consola: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME"
echo ""
echo "🎯 El scraper recolectará automáticamente 1000 imágenes procesadas a 224x224 px"
