#!/usr/bin/env python3
"""
PreditorIA2025 - Fashion Image Scraper
Scraper principal para recolectar 5,000 imágenes de moda de Guatemala
"""

import argparse
import logging
from pathlib import Path
from scrapers.instagram_scraper import InstagramScraper
from scrapers.web_scraper import WebScraper
from storage.database import ImageDatabase
from config.settings import TARGET_IMAGES, DIRECTORIES

def setup_logging():
    """Configura el sistema de logging"""
    Path(DIRECTORIES['logs']).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{DIRECTORIES['logs']}/main.log"),
            logging.StreamHandler()
        ]
    )

def show_statistics():
    """Muestra estadísticas actuales de la colección"""
    db = ImageDatabase()
    stats = db.get_statistics()
    
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE LA COLECCIÓN")
    print("="*50)
    print(f"Total de imágenes: {stats['total_images']}")
    print(f"Objetivo: {TARGET_IMAGES}")
    print(f"Progreso: {(stats['total_images']/TARGET_IMAGES)*100:.1f}%")
    print(f"Calidad promedio: {stats['avg_quality']}")
    
    if stats['by_platform']:
        print("\nPor plataforma:")
        for platform, count in stats['by_platform'].items():
            print(f"  {platform}: {count} imágenes")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Fashion Image Scraper para tesis')
    parser.add_argument('--platform', choices=['instagram', 'web', 'all'], 
                       default='all', help='Plataforma a scrapear')
    parser.add_argument('--images', type=int, default=TARGET_IMAGES,
                       help='Número de imágenes objetivo')
    parser.add_argument('--stats', action='store_true',
                       help='Mostrar estadísticas actuales')
    parser.add_argument('--instagram-only', action='store_true',
                       help='Solo scrapear Instagram')
    parser.add_argument('--web-only', action='store_true',
                       help='Solo scrapear sitios web')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Mostrar estadísticas si se solicita
    if args.stats:
        show_statistics()
        return
    
    logger.info("Iniciando PreditorIA2025 Fashion Image Scraper")
    logger.info(f"Objetivo: {args.images} imágenes")
    
    # Verificar progreso actual
    db = ImageDatabase()
    current_count = db.get_image_count()
    remaining = args.images - current_count
    
    if remaining <= 0:
        logger.info(f"¡Objetivo alcanzado! Ya tienes {current_count} imágenes.")
        show_statistics()
        return
    
    logger.info(f"Imágenes actuales: {current_count}")
    logger.info(f"Imágenes restantes: {remaining}")
    
    # Distribuir trabajo entre plataformas
    if args.instagram_only or args.platform == 'instagram':
        # Solo Instagram
        instagram_scraper = InstagramScraper()
        instagram_scraper.run_scraping_session(remaining)
    
    elif args.web_only or args.platform == 'web':
        # Solo sitios web
        web_scraper = WebScraper()
        web_scraper.run_scraping_session(remaining)
    
    else:
        # Ambas plataformas (por defecto)
        instagram_target = int(remaining * 0.7)  # 70% Instagram
        web_target = remaining - instagram_target  # 30% sitios web
        
        logger.info(f"Distribución: {instagram_target} Instagram, {web_target} sitios web")
        
        # Scrapear Instagram
        if instagram_target > 0:
            logger.info("Iniciando scraping de Instagram...")
            instagram_scraper = InstagramScraper()
            instagram_scraper.run_scraping_session(instagram_target)
        
        # Scrapear sitios web
        if web_target > 0:
            logger.info("Iniciando scraping de sitios web...")
            web_scraper = WebScraper()
            web_scraper.run_scraping_session(web_target)
    
    # Mostrar estadísticas finales
    logger.info("Scraping completado!")
    show_statistics()

if __name__ == "__main__":
    main()
