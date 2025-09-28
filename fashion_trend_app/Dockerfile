# Usar Python 3.9 con más dependencias preinstaladas
FROM python:3.9

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p static/uploads static/images/uploads models

# Exponer puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app_with_progress.py"]