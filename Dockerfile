# Dockerfile para Fashion Image Scraper
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    xvfb \
    libglib2.0-0 \
    libnss3 \
    && rm -rf /var/lib/apt/lists/*

# Instalar Google Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/googlechrome-linux-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/googlechrome-linux-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd -m -u 1000 scraper

# Configurar directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .
RUN chown -R scraper:scraper /app

# Cambiar a usuario no-root
USER scraper

# Crear directorios necesarios
RUN mkdir -p data/images data/logs data/metadata data/temp

# Variables de entorno para Cloud Run
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV HEADLESS_BROWSER=true

# Comando optimizado para Cloud Run - Aplicación web con scraping masivo
CMD ["python", "web_app.py"]
