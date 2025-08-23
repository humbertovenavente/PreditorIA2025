import sqlite3
import json
from datetime import datetime
from pathlib import Path
from config.settings import DATABASE_CONFIG, DIRECTORIES

class ImageDatabase:
    def __init__(self):
        self.db_path = Path(DIRECTORIES['images']) / DATABASE_CONFIG['name']
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos con las tablas necesarias"""
        Path(DIRECTORIES['images']).mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    source_url TEXT,
                    source_platform TEXT,
                    download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    quality_score REAL,
                    hashtags TEXT,
                    description TEXT,
                    location TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scraping_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    platform TEXT,
                    images_collected INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                )
            ''')
    
    def save_image_metadata(self, image_data):
        """Guarda los metadatos de una imagen en la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO images 
                (filename, source_url, source_platform, file_size, width, height, 
                 format, quality_score, hashtags, description, location, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_data['filename'],
                image_data['source_url'],
                image_data['source_platform'],
                image_data['file_size'],
                image_data['width'],
                image_data['height'],
                image_data['format'],
                image_data['quality_score'],
                json.dumps(image_data.get('hashtags', [])),
                image_data.get('description', ''),
                image_data.get('location', ''),
                json.dumps(image_data.get('metadata', {}))
            ))
    
    def get_image_count(self):
        """Retorna el número total de imágenes almacenadas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM images')
            return cursor.fetchone()[0]
    
    def image_exists(self, filename):
        """Verifica si una imagen ya existe en la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT 1 FROM images WHERE filename = ?', (filename,))
            return cursor.fetchone() is not None
    
    def start_session(self, platform):
        """Inicia una nueva sesión de scraping"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'INSERT INTO scraping_sessions (platform) VALUES (?)',
                (platform,)
            )
            return cursor.lastrowid
    
    def update_session(self, session_id, images_collected, status='running'):
        """Actualiza una sesión de scraping"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE scraping_sessions 
                SET images_collected = ?, status = ?, end_time = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (images_collected, status, session_id))
    
    def get_statistics(self):
        """Retorna estadísticas de la colección"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Total de imágenes
            cursor = conn.execute('SELECT COUNT(*) FROM images')
            stats['total_images'] = cursor.fetchone()[0]
            
            # Por plataforma
            cursor = conn.execute('''
                SELECT source_platform, COUNT(*) 
                FROM images 
                GROUP BY source_platform
            ''')
            stats['by_platform'] = dict(cursor.fetchall())
            
            # Calidad promedio
            cursor = conn.execute('SELECT AVG(quality_score) FROM images')
            avg_quality = cursor.fetchone()[0]
            stats['avg_quality'] = round(avg_quality, 2) if avg_quality else 0
            
            return stats
