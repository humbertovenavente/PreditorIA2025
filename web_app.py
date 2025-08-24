"""
Aplicación web para ejecutar el scraper en Cloud Run
Expone endpoints HTTP para controlar el scraping
"""

import os
import json
import logging
from flask import Flask, jsonify, request
from threading import Thread
import time
from scrapers.fast_image_generator import FastImageGenerator
from scrapers.public_api_scraper import PublicAPIImageScraper
from scrapers.instagram_scraper import InstagramScraper
from scrapers.web_scraper import WebScraper
from storage.database import ImageDatabase

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para el estado del scraping
scraping_status = {
    'running': False,
    'progress': 0,
    'total_target': 0,
    'images_collected': 0,
    'current_method': None,
    'start_time': None,
    'logs': []
}

def add_log(message):
    """Agregar mensaje al log"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    scraping_status['logs'].append(log_entry)
    logger.info(message)
    
    # Mantener solo los últimos 100 logs
    if len(scraping_status['logs']) > 100:
        scraping_status['logs'] = scraping_status['logs'][-100:]

def run_scraping_task(target_images, method='fast'):
    """Ejecutar tarea de scraping en background"""
    global scraping_status
    
    try:
        scraping_status['running'] = True
        scraping_status['total_target'] = target_images
        scraping_status['current_method'] = method
        scraping_status['start_time'] = time.time()
        
        add_log(f"Iniciando scraping con método '{method}' - Objetivo: {target_images} imágenes")
        
        if method == 'fast':
            scraper = FastImageGenerator()
            collected = scraper.run_fast_generation(target_images)
        elif method == 'api':
            scraper = PublicAPIImageScraper()
            collected = scraper.run_scraping_session(target_images)
        elif method == 'instagram':
            scraper = InstagramScraper()
            collected = scraper.run_scraping_session(target_images)
        elif method == 'web':
            scraper = WebScraper()
            collected = scraper.run_scraping_session(target_images)
        else:
            scraper = PublicAPIImageScraper()
            collected = scraper.run_scraping_session(target_images)
        
        scraping_status['images_collected'] = collected
        add_log(f"Scraping completado: {collected} imágenes recolectadas")
        
    except Exception as e:
        add_log(f"Error en scraping: {str(e)}")
        logger.error(f"Error en scraping: {e}")
    
    finally:
        scraping_status['running'] = False

@app.route('/')
def home():
    """Página principal"""
    return jsonify({
        'service': 'PreditorIA2025 Fashion Scraper',
        'status': 'running',
        'endpoints': {
            '/start': 'POST - Iniciar scraping',
            '/status': 'GET - Estado actual',
            '/stats': 'GET - Estadísticas de la base de datos',
            '/logs': 'GET - Logs recientes'
        }
    })

@app.route('/start', methods=['POST'])
def start_scraping():
    """Iniciar proceso de scraping"""
    if scraping_status['running']:
        return jsonify({
            'error': 'Scraping ya está en ejecución',
            'status': scraping_status
        }), 400
    
    data = request.get_json() or {}
    target_images = data.get('images', 1000)
    method = data.get('method', 'fast')
    
    # Iniciar scraping en thread separado
    thread = Thread(target=run_scraping_task, args=(target_images, method))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Scraping iniciado',
        'target_images': target_images,
        'method': method
    })

@app.route('/status')
def get_status():
    """Obtener estado actual del scraping"""
    # Actualizar progreso si está corriendo
    if scraping_status['running']:
        try:
            db = ImageDatabase()
            current_count = db.get_image_count()
            scraping_status['images_collected'] = current_count
            
            if scraping_status['total_target'] > 0:
                scraping_status['progress'] = (current_count / scraping_status['total_target']) * 100
        except Exception as e:
            add_log(f"Error actualizando estado: {str(e)}")
    
    return jsonify(scraping_status)

@app.route('/stats')
def get_stats():
    """Obtener estadísticas de la base de datos"""
    try:
        db = ImageDatabase()
        stats = db.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs')
def get_logs():
    """Obtener logs recientes"""
    return jsonify({
        'logs': scraping_status['logs'][-50:],  # Últimos 50 logs
        'total_logs': len(scraping_status['logs'])
    })

@app.route('/health')
def health_check():
    """Health check para Cloud Run"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/auto-start')
def auto_start():
    """Auto-iniciar scraping al deployar"""
    if not scraping_status['running']:
        # Iniciar automáticamente con 1000 imágenes
        thread = Thread(target=run_scraping_task, args=(1000, 'fast'))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Scraping auto-iniciado',
            'target_images': 1000,
            'method': 'fast'
        })
    else:
        return jsonify({
            'message': 'Scraping ya está en ejecución'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    # Auto-iniciar scraping masivo al arrancar
    add_log("Aplicación web iniciada")
    add_log("Iniciando scraping masivo para 5000 imágenes...")
    
    # Iniciar scraping automáticamente con objetivo de 5000 imágenes
    thread = Thread(target=run_scraping_task, args=(5000, 'fast'))
    thread.daemon = True
    thread.start()
    
    app.run(host='0.0.0.0', port=port, debug=False)
