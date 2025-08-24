#!/bin/bash

# Script de configuración automática para Google Cloud Storage
# PreditorIA2025 - Fashion Image Scraper

set -e

echo "🚀 Configurando Google Cloud Storage para PreditorIA2025"
echo "=================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar si gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    print_error "Google Cloud CLI no está instalado"
    echo "Instala gcloud desde: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_status "Google Cloud CLI encontrado"

# Verificar autenticación
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "No hay cuentas autenticadas en gcloud"
    echo "Ejecuta: gcloud auth login"
    exit 1
fi

print_status "Usuario autenticado en gcloud"

# Obtener o solicitar Project ID
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo -n "Ingresa tu Google Cloud Project ID: "
    read PROJECT_ID
else
    PROJECT_ID=$GOOGLE_CLOUD_PROJECT
fi

# Configurar proyecto por defecto
gcloud config set project $PROJECT_ID
print_status "Proyecto configurado: $PROJECT_ID"

# Generar nombre único para el bucket
TIMESTAMP=$(date +%s)
BUCKET_NAME="fashion-images-dataset-${PROJECT_ID}-${TIMESTAMP}"

echo "📦 Bucket name: $BUCKET_NAME"

# Habilitar APIs necesarias
print_status "Habilitando APIs necesarias..."
gcloud services enable storage-component.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com

# Crear Service Account
SERVICE_ACCOUNT_NAME="fashion-scraper"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

print_status "Creando Service Account..."
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL &> /dev/null; then
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --description="Service account para fashion image scraper" \
        --display-name="Fashion Scraper"
    print_status "Service Account creado: $SERVICE_ACCOUNT_EMAIL"
else
    print_warning "Service Account ya existe: $SERVICE_ACCOUNT_EMAIL"
fi

# Asignar roles
print_status "Asignando roles..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin" \
    --quiet

# Crear clave del service account
print_status "Creando credenciales..."
mkdir -p credentials
gcloud iam service-accounts keys create credentials/gcp-key.json \
    --iam-account=$SERVICE_ACCOUNT_EMAIL

# Crear bucket
print_status "Creando bucket de Cloud Storage..."
if ! gsutil ls gs://$BUCKET_NAME &> /dev/null; then
    gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 gs://$BUCKET_NAME
    print_status "Bucket creado: gs://$BUCKET_NAME"
else
    print_warning "Bucket ya existe: gs://$BUCKET_NAME"
fi

# Configurar permisos del bucket
print_status "Configurando permisos del bucket..."
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT_EMAIL:objectAdmin gs://$BUCKET_NAME

# Crear estructura de directorios en el bucket
print_status "Creando estructura de directorios..."
echo "# Fashion Images Dataset" | gsutil cp - gs://$BUCKET_NAME/README.md
gsutil cp /dev/null gs://$BUCKET_NAME/images/.keep
gsutil cp /dev/null gs://$BUCKET_NAME/metadata/.keep

# Actualizar archivo .env
print_status "Actualizando archivo .env..."
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

print_status "Archivo .env actualizado"

# Mostrar resumen
echo ""
echo "🎉 ¡Configuración completada!"
echo "=================================================="
echo "Project ID: $PROJECT_ID"
echo "Bucket Name: $BUCKET_NAME"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "Credenciales: credentials/gcp-key.json"
echo ""
echo "💰 Costo estimado mensual: ~$0.25 USD para 5,000 imágenes"
echo ""
echo "🔧 Próximos pasos:"
echo "1. Probar conexión: python main.py --stats"
echo "2. Colección inicial: python main.py --demo --images 10"
echo "3. Producción: python main.py --images 5000"
echo ""
print_status "¡Google Cloud Storage configurado exitosamente!"
