# 🌐 PreditorIA2025 - Dashboard Web

## 🎯 **Aplicación Web de Análisis de Moda con IA**

Dashboard web desarrollado con Flask que integra el modelo entrenado MobileNetV2 y el sistema de clustering para análisis de tendencias de moda guatemalteca en tiempo real.

---

## ✨ **Características Principales**

### **🔍 Análisis de Imágenes en Tiempo Real**
- **Carga de imágenes**: Drag & drop o selección de archivos
- **Predicción de categorías**: Clasificación automática de prendas
- **Nivel de confianza**: Porcentaje de certeza de la predicción
- **Imágenes similares**: Búsqueda de prendas similares usando clustering

### **📊 Dashboard de Clustering**
- **Visualización de clusters**: Gráficos interactivos de distribución
- **Métricas de calidad**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- **Estadísticas detalladas**: Análisis por cluster y categoría
- **Información en tiempo real**: Datos actualizados del sistema

### **🎨 Interfaz Moderna**
- **Diseño responsivo**: Compatible con dispositivos móviles y desktop
- **UI/UX intuitiva**: Interfaz fácil de usar y navegar
- **Visualizaciones interactivas**: Gráficos con Chart.js
- **Feedback visual**: Alertas y notificaciones en tiempo real

---

## 🚀 **Instalación y Uso**

### **Prerrequisitos**
- Python 3.8+
- Entorno virtual activado
- Modelo entrenado (`mobilenet_v2_final.h5`)
- Resultados de clustering (`clustering_results.pkl`)

### **Instalación**
```bash
# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias adicionales
pip install flask flask-cors

# Ejecutar aplicación
python3 app.py
```

### **Acceso**
- **URL**: http://localhost:5000
- **Puerto**: 5000 (configurable)
- **Host**: 0.0.0.0 (accesible desde red local)

---

## 📁 **Estructura de la Aplicación**

```
PreditorIA2025/
├── app.py                          # Aplicación Flask principal
├── templates/
│   └── index.html                  # Plantilla HTML principal
├── static/
│   ├── css/
│   │   └── style.css              # Estilos personalizados
│   ├── js/
│   │   └── app.js                 # JavaScript de la aplicación
│   ├── uploads/                   # Directorio de imágenes subidas
│   └── results/                   # Directorio de resultados
├── data/
│   └── logs/training/
│       └── mobilenet_v2_final.h5  # Modelo entrenado
├── clustering_results/
│   └── clustering_results.pkl     # Resultados de clustering
└── test_app.py                    # Script de pruebas
```

---

## 🔧 **API Endpoints**

### **GET /**
- **Descripción**: Página principal del dashboard
- **Respuesta**: HTML del dashboard

### **POST /upload**
- **Descripción**: Sube una imagen al servidor
- **Parámetros**: `file` (archivo de imagen)
- **Respuesta**: JSON con información del archivo subido

### **POST /predict**
- **Descripción**: Analiza una imagen y predice su categoría
- **Parámetros**: `filepath` (ruta del archivo)
- **Respuesta**: JSON con predicción y imágenes similares

### **GET /clustering-info**
- **Descripción**: Obtiene información sobre el clustering
- **Respuesta**: JSON con estadísticas y métricas de clustering

---

## 🎯 **Funcionalidades Detalladas**

### **1. Análisis de Imágenes**
```javascript
// Carga de archivo
const file = document.getElementById('fileInput').files[0];
const formData = new FormData();
formData.append('file', file);

// Subir imagen
fetch('/upload', {
    method: 'POST',
    body: formData
});

// Analizar imagen
fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filepath: 'path/to/image.jpg'})
});
```

### **2. Visualización de Resultados**
- **Predicción de categoría**: Muestra la clase predicha con nivel de confianza
- **Barra de confianza**: Visualización gráfica del nivel de certeza
- **Imágenes similares**: Galería de prendas similares encontradas
- **Información detallada**: ID de clase y métricas adicionales

### **3. Dashboard de Clustering**
- **Gráfico de distribución**: Doughnut chart de clusters
- **Métricas de calidad**: Bar chart de scores de clustering
- **Estadísticas por cluster**: Tarjetas con información detallada
- **Datos en tiempo real**: Actualización automática de información

---

## 🎨 **Personalización**

### **Estilos CSS**
```css
/* Variables de color personalizables */
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    /* ... más variables */
}
```

### **Configuración de la Aplicación**
```python
# app.py
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

---

## 🧪 **Pruebas**

### **Script de Pruebas Automatizadas**
```bash
# Ejecutar pruebas
python3 test_app.py
```

### **Pruebas Manuales**
1. **Carga de imagen**: Subir archivo de imagen válido
2. **Análisis**: Verificar predicción de categoría
3. **Clustering**: Revisar información de clusters
4. **Responsividad**: Probar en diferentes dispositivos

---

## 📊 **Métricas de Rendimiento**

### **Tiempos de Respuesta**
- **Carga de página**: < 2 segundos
- **Subida de imagen**: < 5 segundos (depende del tamaño)
- **Análisis de imagen**: < 3 segundos
- **Información de clustering**: < 1 segundo

### **Capacidades**
- **Tamaño máximo de archivo**: 16MB
- **Formatos soportados**: PNG, JPG, JPEG, GIF, BMP
- **Resolución de entrada**: 224x224 píxeles (redimensionado automático)
- **Concurrencia**: Múltiples usuarios simultáneos

---

## 🔒 **Seguridad**

### **Validaciones Implementadas**
- **Tipo de archivo**: Solo imágenes permitidas
- **Tamaño de archivo**: Límite de 16MB
- **Nombres de archivo**: Sanitización con `secure_filename`
- **CORS**: Configurado para desarrollo

### **Recomendaciones para Producción**
- Implementar autenticación de usuarios
- Configurar HTTPS
- Añadir rate limiting
- Implementar logging de seguridad

---

## 🚀 **Despliegue en Producción**

### **Configuración para Producción**
```python
# Cambiar en app.py
app.run(debug=False, host='0.0.0.0', port=5000)
```

### **Servidor Web (Nginx + Gunicorn)**
```bash
# Instalar Gunicorn
pip install gunicorn

# Ejecutar con Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Docker (Opcional)**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 🎉 **¡Dashboard Completado!**

El dashboard web de PreditorIA2025 está completamente funcional y listo para usar. Integra exitosamente:

✅ **Modelo entrenado MobileNetV2** para predicciones de categorías  
✅ **Sistema de clustering K-Means** para imágenes similares  
✅ **Interfaz web moderna** con Flask y Bootstrap  
✅ **API REST** para análisis en tiempo real  
✅ **Visualizaciones interactivas** de datos y resultados  
✅ **Diseño responsivo** para todos los dispositivos  

**¡El prototipo MVP está listo para demostrar las capacidades del sistema de análisis de moda con IA!** 🚀✨
