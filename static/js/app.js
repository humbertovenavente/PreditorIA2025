// JavaScript para PreditorIA2025 Dashboard

class FashionAnalyzer {
    constructor() {
        this.currentFile = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const analyzeBtn = document.getElementById('analyzeBtn');

        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Analyze button
        analyzeBtn.addEventListener('click', () => {
            this.analyzeImage();
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validar que sea un archivo de imagen (más flexible)
        const validImageTypes = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
            'image/bmp', 'image/tiff', 'image/webp', 'image/svg+xml'
        ];
        
        // Verificar por tipo MIME o extensión
        const isValidImage = validImageTypes.includes(file.type) || 
                           /\.(jpg|jpeg|png|gif|bmp|tiff|webp|svg)$/i.test(file.name);
        
        if (!isValidImage) {
            this.showAlert('Por favor selecciona un archivo de imagen válido (JPG, PNG, GIF, BMP, TIFF, WEBP, SVG).', 'danger');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showAlert('El archivo es demasiado grande. Máximo 16MB.', 'danger');
            return;
        }

        this.currentFile = file;
        this.previewImage(file);
        this.uploadFile(file);
    }

    previewImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('imagePreview');
            const img = document.getElementById('previewImg');
            img.src = e.target.result;
            preview.style.display = 'block';
            preview.scrollIntoView({ behavior: 'smooth' });
        };
        reader.readAsDataURL(file);
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading('Subiendo archivo...');
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentFile.path = result.filepath;
                document.getElementById('analyzeBtn').disabled = false;
                this.showAlert('Archivo subido exitosamente. Puedes analizarlo ahora.', 'success');
            } else {
                this.showAlert(result.error || 'Error subiendo archivo', 'danger');
            }
        } catch (error) {
            this.showAlert('Error de conexión: ' + error.message, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async analyzeImage() {
        if (!this.currentFile || !this.currentFile.path) {
            this.showAlert('No hay archivo para analizar', 'warning');
            return;
        }

        try {
            this.showLoading('Analizando imagen...');
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filepath: this.currentFile.path
                })
            });

            const result = await response.json();

            if (result.prediction) {
                this.displayResults(result);
                this.showAlert('Análisis completado exitosamente', 'success');
            } else {
                this.showAlert(result.error || 'Error analizando imagen', 'danger');
            }
        } catch (error) {
            this.showAlert('Error de conexión: ' + error.message, 'danger');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Show results section
        const resultsSection = document.getElementById('results');
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Display prediction
        this.displayPrediction(result.prediction);
    }

    displayPrediction(prediction) {
        const container = document.getElementById('predictionResult');
        
        const isTrendy = prediction.is_trendy;
        const trendConfidence = Math.round(prediction.trend_confidence * 100);
        const category = this.capitalizeFirst(prediction.category);
        const categoryConfidence = Math.round(prediction.category_confidence * 100);
        
        // Determinar colores y iconos
        const trendColor = isTrendy ? 'success' : 'danger';
        const trendIcon = isTrendy ? 'fas fa-check-circle' : 'fas fa-times-circle';
        const trendText = isTrendy ? '¡SÍ ESTÁ DE MODA!' : 'NO ESTÁ DE MODA';
        
        container.innerHTML = `
            <div class="trend-result">
                <div class="trend-icon mb-4">
                    <i class="${trendIcon} fa-5x text-${trendColor}"></i>
                </div>
                <h2 class="trend-text text-${trendColor} mb-4">${trendText}</h2>
                <div class="trend-confidence mb-4">
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar bg-${trendColor}" 
                             style="width: ${trendConfidence}%" 
                             role="progressbar">
                            ${trendConfidence}% de confianza
                        </div>
                    </div>
                </div>
                <div class="trend-details">
                    <p class="lead mb-3">${prediction.trend_reason}</p>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="detail-item">
                                <h6><i class="fas fa-tag me-2"></i>Categoría</h6>
                                <p class="mb-0">${category} (${categoryConfidence}%)</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="detail-item">
                                <h6><i class="fas fa-chart-line me-2"></i>Tendencia</h6>
                                <p class="mb-0">${trendConfidence}% de confianza</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="mt-4">
                    <button class="btn btn-outline-primary" onclick="location.reload()">
                        <i class="fas fa-redo me-2"></i>
                        Analizar Otra Imagen
                    </button>
                </div>
            </div>
        `;
    }


    // Utility functions
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    getFileName(path) {
        return path.split('/').pop();
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }

    showLoading(message = 'Cargando...') {
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.innerHTML = `
            <span class="loading-spinner me-2"></span>
            ${message}
        `;
        analyzeBtn.disabled = true;
    }

    hideLoading() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.innerHTML = `
            <i class="fas fa-magic me-2"></i>
            ¿Está de Moda?
        `;
        analyzeBtn.disabled = false;
    }
}

// Smooth scrolling function
function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({
        behavior: 'smooth'
    });
}

// Global variable for debugging
let fashionAnalyzer;

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    fashionAnalyzer = new FashionAnalyzer();
    console.log('FashionAnalyzer initialized');
});

// Global function for debugging (temporary)
window.analyzeImage = function() {
    if (fashionAnalyzer) {
        fashionAnalyzer.analyzeImage();
    } else {
        console.error('FashionAnalyzer not initialized');
    }
};
    }
};
