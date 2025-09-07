// Fashion Trend Analysis - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Image preview functionality
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            previewImage(e.target);
        });
    }

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            // Skip if href is just '#' or empty
            if (href && href !== '#' && href.length > 1) {
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // Add loading states to buttons
    const submitButtons = document.querySelectorAll('button[type="submit"]');
    submitButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.form && this.form.checkValidity()) {
                this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Procesando...';
                this.disabled = true;
            }
        });
    });
});

// Image preview function
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewContainer = document.getElementById('imagePreview');
            const previewImg = document.getElementById('previewImg');
            
            if (previewContainer && previewImg) {
                previewImg.src = e.target.result;
                previewContainer.style.display = 'block';
                
                // Add fade-in animation
                previewContainer.classList.add('fade-in-up');
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// Show statistics modal
function showStats() {
    fetch('/api/stats')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showAlert('Error cargando estadísticas: ' + data.error, 'danger');
                return;
            }
            
            // Update modal content
            updateStatsModal(data);
            
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('statsModal'));
            modal.show();
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Error cargando estadísticas: ' + error.message, 'danger');
        });
}

// Update statistics modal content
function updateStatsModal(data) {
    const elements = {
        'totalClusters': data.total_clusters,
        'totalImages': data.total_images,
        'algorithm': data.algorithm,
        'silhouetteScore': data.silhouette_score ? data.silhouette_score.toFixed(3) : 'N/A',
        'dbiScore': data.dbi_score ? data.dbi_score.toFixed(3) : 'N/A'
    };

    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

// Show alert function
function showAlert(message, type = 'info') {
    const alertContainer = document.querySelector('.container.mt-3') || document.querySelector('main .container');
    if (!alertContainer) return;

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    alertContainer.insertBefore(alertDiv, alertContainer.firstChild);

    // Auto-hide after 5 seconds
    setTimeout(() => {
        const bsAlert = new bootstrap.Alert(alertDiv);
        bsAlert.close();
    }, 5000);
}

// API call function
async function callAPI(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        }
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Drag and drop functionality for file upload
function initDragAndDrop() {
    const dropZone = document.querySelector('.card-body');
    const fileInput = document.getElementById('file');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('border-primary', 'border-3');
    }

    function unhighlight(e) {
        dropZone.classList.remove('border-primary', 'border-3');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            previewImage(fileInput);
        }
    }
}

// Initialize drag and drop when DOM is loaded
document.addEventListener('DOMContentLoaded', initDragAndDrop);

// Utility functions
const utils = {
    // Format number with commas
    formatNumber: (num) => {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    // Format percentage
    formatPercentage: (num, decimals = 1) => {
        return (num * 100).toFixed(decimals) + '%';
    },

    // Get color class based on value
    getColorClass: (value, thresholds = { low: 30, medium: 70 }) => {
        if (value < thresholds.low) return 'text-danger';
        if (value < thresholds.medium) return 'text-warning';
        return 'text-success';
    },

    // Debounce function
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Analysis progress monitoring
let analysisInterval = null;
let currentAnalysisFilename = null;

// Start analysis with progress monitoring
async function startAnalysis(file) {
    console.log('startAnalysis llamada con archivo:', file);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        console.log('Mostrando contenedor de progreso...');
        // Show progress container
        showProgressContainer();
        
        console.log('Enviando petición a /api/analyze...');
        // Start analysis
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        console.log('Respuesta recibida:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Datos de respuesta:', data);
        
        if (data.success) {
            currentAnalysisFilename = data.filename;
            console.log('Iniciando monitoreo de progreso...');
            startProgressMonitoring();
        } else {
            throw new Error(data.error || 'Error iniciando análisis');
        }
    } catch (error) {
        console.error('Error starting analysis:', error);
        showAlert('Error iniciando análisis: ' + error.message, 'danger');
        hideProgressContainer();
    }
}

// Start monitoring analysis progress
function startProgressMonitoring() {
    if (analysisInterval) {
        clearInterval(analysisInterval);
    }
    
    analysisInterval = setInterval(async () => {
        try {
            // Check progress
            const progressResponse = await fetch('/api/analysis_progress');
            const progressData = await progressResponse.json();
            
            updateProgressBar(progressData.progress, progressData.status);
            
            // If analysis is complete, get results
            if (!progressData.analysis_active && progressData.progress === 0) {
                clearInterval(analysisInterval);
                analysisInterval = null;
                
                // Wait a moment then get results
                setTimeout(async () => {
                    await getAnalysisResults();
                }, 500);
            }
        } catch (error) {
            console.error('Error monitoring progress:', error);
            clearInterval(analysisInterval);
            analysisInterval = null;
            showAlert('Error monitoreando progreso: ' + error.message, 'danger');
            hideProgressContainer();
        }
    }, 1000); // Check every second
}

// Update progress bar
function updateProgressBar(progress, status) {
    const progressBar = document.getElementById('analysisProgress');
    const progressText = document.getElementById('analysisStatus');
    
    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.setAttribute('aria-valuenow', progress);
    }
    
    if (progressText) {
        progressText.textContent = status;
    }
}

// Get analysis results
async function getAnalysisResults() {
    try {
        const response = await fetch('/api/analysis_result');
        const data = await response.json();
        
        if (data.success) {
            showAnalysisResults(data.result, data.filename);
        } else {
            throw new Error(data.message || 'Error obteniendo resultados');
        }
    } catch (error) {
        console.error('Error getting results:', error);
        showAlert('Error obteniendo resultados: ' + error.message, 'danger');
    } finally {
        hideProgressContainer();
    }
}

// Show analysis results
function showAnalysisResults(result, filename) {
    // Create results container
    const resultsContainer = document.getElementById('analysisResults');
    if (!resultsContainer) return;
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    // Create results HTML
    const resultsHTML = `
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">
                    <i class="fas fa-chart-line me-2"></i>
                    Resultados del Análisis
                </h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="fw-bold text-primary">Análisis de Tendencias</h6>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Trend Score:</span>
                                <span class="fw-bold ${result.trend_score >= 70 ? 'text-success' : 'text-warning'}">
                                    ${result.trend_score}/100
                                </span>
                            </div>
                            <div class="progress mt-1" style="height: 8px;">
                                <div class="progress-bar ${result.trend_score >= 70 ? 'bg-success' : 'bg-warning'}" 
                                     style="width: ${result.trend_score}%"></div>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>¿Está en tendencia?</span>
                                <span class="badge ${result.is_trending ? 'bg-success' : 'bg-secondary'}">
                                    ${result.is_trending ? 'SÍ' : 'NO'}
                                </span>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Cluster ID:</span>
                                <span class="fw-bold">${result.cluster_id}</span>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Similitud:</span>
                                <span class="fw-bold">${result.similarity_score ? result.similarity_score.toFixed(1) : 'N/A'}%</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="fw-bold text-primary">Análisis Visual</h6>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Confianza:</span>
                                <span class="fw-bold">${result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A'}%</span>
                            </div>
                        </div>
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span>Clase predicha:</span>
                                <span class="fw-bold">${result.predicted_class || 'N/A'}</span>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-bold">Colores dominantes:</label>
                            <div class="d-flex flex-wrap gap-1">
                                ${result.colors ? result.colors.map(color => 
                                    `<span class="badge bg-secondary">${color}</span>`
                                ).join('') : '<span class="text-muted">No detectados</span>'}
                            </div>
                        </div>
                        ${result.cluster_info ? `
                        <div class="mb-3">
                            <label class="form-label fw-bold">Información del Cluster:</label>
                            <p class="text-muted small mb-0">${result.cluster_info.description}</p>
                        </div>
                        ` : ''}
                    </div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-12">
                        <small class="text-muted">
                            <i class="fas fa-clock me-1"></i>
                            Análisis completado: ${result.timestamp}
                        </small>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
    resultsContainer.style.display = 'block';
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// Show progress container
function showProgressContainer() {
    const progressContainer = document.getElementById('progressContainer');
    if (progressContainer) {
        progressContainer.style.display = 'block';
    }
    
    // Reset progress
    updateProgressBar(0, 'Iniciando análisis...');
}

// Hide progress container
function hideProgressContainer() {
    const progressContainer = document.getElementById('progressContainer');
    if (progressContainer) {
        progressContainer.style.display = 'none';
    }
}

// Enhanced form submission for analysis
function initAnalysisForm() {
    const form = document.getElementById('analysisForm');
    if (!form) {
        console.error('No se encontró el formulario con id="analysisForm"');
        return;
    }
    
    console.log('Formulario encontrado:', form);
    
    form.addEventListener('submit', async function(e) {
        console.log('Formulario enviado!');
        e.preventDefault();
        
        const fileInput = document.getElementById('file');
        console.log('File input:', fileInput);
        console.log('Files:', fileInput.files);
        
        if (!fileInput.files || fileInput.files.length === 0) {
            console.log('No hay archivo seleccionado');
            showAlert('Por favor selecciona una imagen', 'warning');
            return;
        }
        
        const file = fileInput.files[0];
        console.log('Archivo seleccionado:', file);
        
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            console.log('Tipo de archivo no permitido:', file.type);
            showAlert('Tipo de archivo no permitido. Usa JPG, PNG, GIF, BMP o WebP', 'warning');
            return;
        }
        
        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            console.log('Archivo demasiado grande:', file.size);
            showAlert('El archivo es demasiado grande. Máximo 16MB', 'warning');
            return;
        }
        
        console.log('Iniciando análisis...');
        // Start analysis
        await startAnalysis(file);
    });
}

// Initialize analysis form when DOM is loaded
document.addEventListener('DOMContentLoaded', initAnalysisForm);

// Export functions for global use
window.previewImage = previewImage;
window.showStats = showStats;
window.showAlert = showAlert;
window.utils = utils;
window.startAnalysis = startAnalysis;
