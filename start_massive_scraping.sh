#!/bin/bash

# Script para iniciar scraping masivo de 5000 imágenes en Cloud Run
# Usa todos los métodos: Instagram, Web, APIs públicas

SERVICE_URL="https://fashion-scraper-285994278779.us-central1.run.app"

echo "🚀 Iniciando scraping masivo para 5000 imágenes de moda..."

# Iniciar scraping con método fast (genera imágenes rápidamente)
echo "📊 Iniciando generador rápido para 2000 imágenes..."
curl -X POST "$SERVICE_URL/start" \
  -H "Content-Type: application/json" \
  -d '{"images": 2000, "method": "fast"}'

echo ""
echo "⏳ Esperando 30 segundos antes del siguiente método..."
sleep 30

# Iniciar scraping con APIs públicas
echo "🌐 Iniciando APIs públicas para 1500 imágenes..."
curl -X POST "$SERVICE_URL/start" \
  -H "Content-Type: application/json" \
  -d '{"images": 1500, "method": "api"}'

echo ""
echo "⏳ Esperando 30 segundos antes del siguiente método..."
sleep 30

# Iniciar scraping de Instagram
echo "📱 Iniciando Instagram scraping para 1000 imágenes..."
curl -X POST "$SERVICE_URL/start" \
  -H "Content-Type: application/json" \
  -d '{"images": 1000, "method": "instagram"}'

echo ""
echo "⏳ Esperando 30 segundos antes del siguiente método..."
sleep 30

# Iniciar scraping web
echo "🌍 Iniciando Web scraping para 500 imágenes..."
curl -X POST "$SERVICE_URL/start" \
  -H "Content-Type: application/json" \
  -d '{"images": 500, "method": "web"}'

echo ""
echo "✅ Todos los scrapers iniciados!"
echo "📊 Total objetivo: 5000 imágenes (2000 fast + 1500 API + 1000 Instagram + 500 Web)"
echo ""
echo "🔍 Para monitorear el progreso:"
echo "   - Estado: curl $SERVICE_URL/status"
echo "   - Estadísticas: curl $SERVICE_URL/stats"
echo "   - Logs: curl $SERVICE_URL/logs"
echo ""
echo "🌐 Consola web: https://console.cloud.google.com/run/detail/us-central1/fashion-scraper"
