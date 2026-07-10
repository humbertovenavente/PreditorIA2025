# Backend de analisis de tendencias (Flask + TensorFlow) para AWS
FROM python:3.11-slim

WORKDIR /app

# Dependencias de sistema para onnxruntime/rembg (segmentacion) y pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el codigo de la aplicacion
COPY . .

# Directorios de runtime
RUN mkdir -p static/uploads static/images/uploads models logs

# Puerto configurable (App Runner/ECS inyectan PORT)
ENV PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# Servidor de produccion. 1 worker (modelos pesados en RAM) + threads para I/O.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 180 app:app"]
