# Configuración de Google Cloud Platform

## 1. Crear Proyecto en GCP

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Anota el **Project ID**

## 2. Habilitar APIs Necesarias

```bash
# Habilitar Cloud Storage API
gcloud services enable storage-component.googleapis.com

# Habilitar Cloud Resource Manager API
gcloud services enable cloudresourcemanager.googleapis.com
```

## 3. Crear Service Account

```bash
# Crear service account
gcloud iam service-accounts create fashion-scraper \
    --description="Service account para fashion image scraper" \
    --display-name="Fashion Scraper"

# Asignar roles necesarios
gcloud projects add-iam-policy-binding TU-PROJECT-ID \
    --member="serviceAccount:fashion-scraper@TU-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Crear y descargar clave
gcloud iam service-accounts keys create credentials/gcp-key.json \
    --iam-account=fashion-scraper@TU-PROJECT-ID.iam.gserviceaccount.com
```

## 4. Crear Bucket de Cloud Storage

```bash
# Crear bucket (el nombre debe ser único globalmente)
gsutil mb -p TU-PROJECT-ID -c STANDARD -l us-central1 gs://fashion-images-dataset-TU-NOMBRE

# Configurar permisos
gsutil iam ch serviceAccount:fashion-scraper@TU-PROJECT-ID.iam.gserviceaccount.com:objectAdmin gs://fashion-images-dataset-TU-NOMBRE
```

## 5. Configurar Variables de Entorno

Copia `.env.example` a `.env` y configura:

```bash
cp .env.example .env
```

Edita `.env`:
```
GOOGLE_CLOUD_PROJECT=tu-project-id
GCS_BUCKET_NAME=fashion-images-dataset-tu-nombre
```

## 6. Estructura de Directorios

Crea la carpeta de credenciales:
```bash
mkdir -p credentials
```

Coloca tu archivo `gcp-key.json` en `credentials/gcp-key.json`

## Estimación de Costos

Para 5,000 imágenes (~10GB):
- **Cloud Storage**: ~$0.20/mes
- **Operaciones**: ~$0.05/mes
- **Total estimado**: ~$0.25/mes

## Comandos Útiles

```bash
# Ver estadísticas del bucket
gsutil du -sh gs://tu-bucket-name

# Listar archivos
gsutil ls gs://tu-bucket-name/images/

# Descargar todo el dataset
gsutil -m cp -r gs://tu-bucket-name/images/ ./local-backup/
```
