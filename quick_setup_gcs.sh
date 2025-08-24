#!/bin/bash

# Script simplificado para configurar Google Cloud Storage
# PreditorIA2025 - Fashion Image Scraper

echo "🚀 Configuración rápida de Google Cloud Storage"
echo "=============================================="

# Verificar autenticación
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ No estás autenticado. Ejecuta: gcloud auth login"
    exit 1
fi

echo "✅ Usuario autenticado"

# Crear proyecto (opcional, puedes usar uno existente)
echo "📋 Proyectos disponibles:"
gcloud projects list --format="table(projectId,name,projectNumber)"

echo ""
echo -n "Ingresa el Project ID a usar (o presiona Enter para crear uno nuevo): "
read PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    # Crear nuevo proyecto
    TIMESTAMP=$(date +%s)
    PROJECT_ID="preditoria2025-${TIMESTAMP}"
    echo "🆕 Creando nuevo proyecto: $PROJECT_ID"
    gcloud projects create $PROJECT_ID --name="PreditorIA2025 Fashion Scraper"
fi

# Configurar proyecto por defecto
gcloud config set project $PROJECT_ID
echo "✅ Proyecto configurado: $PROJECT_ID"

# Habilitar facturación (necesario para Cloud Storage)
echo "💳 Habilitando facturación..."
echo "⚠️  IMPORTANTE: Necesitas habilitar facturación en Google Cloud Console"
echo "   Ve a: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
echo ""
echo -n "¿Has habilitado la facturación? (y/N): "
read BILLING_ENABLED

if [[ ! "$BILLING_ENABLED" =~ ^[Yy]$ ]]; then
    echo "❌ La facturación es necesaria para usar Cloud Storage"
    echo "   Habilítala y ejecuta este script nuevamente"
    exit 1
fi

# Habilitar APIs
echo "🔧 Habilitando APIs necesarias..."
gcloud services enable storage-component.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com

# Crear bucket único
TIMESTAMP=$(date +%s)
BUCKET_NAME="fashion-images-${PROJECT_ID}-${TIMESTAMP}"

echo "📦 Creando bucket: $BUCKET_NAME"
gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://$BUCKET_NAME

# Crear Service Account
SERVICE_ACCOUNT_NAME="fashion-scraper"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🔑 Creando Service Account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --description="Service account para fashion image scraper" \
    --display-name="Fashion Scraper"

# Asignar permisos
echo "🛡️  Asignando permisos..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin" \
    --quiet

# Crear credenciales
echo "📄 Creando archivo de credenciales..."
mkdir -p credentials
gcloud iam service-accounts keys create credentials/gcp-key.json \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

# Actualizar .env
echo "⚙️  Actualizando configuración..."
cat > .env << EOF
# Configuración de Google Cloud Platform
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GCS_BUCKET_NAME=$BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS=credentials/gcp-key.json

# Configuración del scraper
TARGET_IMAGES=5000
HEADLESS_BROWSER=true

# Configuración de delays (en segundos)
DELAY_BETWEEN_REQUESTS=2
DELAY_BETWEEN_PAGES=5
DELAY_ON_ERROR=10

# Configuración de filtros de calidad
MIN_IMAGE_WIDTH=300
MIN_IMAGE_HEIGHT=300
MAX_FILE_SIZE_MB=10
MIN_QUALITY_SCORE=0.7
EOF

echo ""
echo "🎉 ¡Configuración completada!"
echo "=========================="
echo "Project ID: $PROJECT_ID"
echo "Bucket: gs://$BUCKET_NAME"
echo "Credenciales: credentials/gcp-key.json"
echo ""
echo "💰 Costo estimado: ~$0.25/mes para 5,000 imágenes"
echo ""
echo "🧪 Probar configuración:"
echo "   python main.py --demo --images 5"
echo ""
echo "🚀 Colección inicial:"
echo "   python main.py --demo --images 50"
echo ""
echo "📈 Producción completa:"
echo "   python main.py --images 5000"
