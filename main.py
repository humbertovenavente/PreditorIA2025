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
from scrapers.public_api_scraper import PublicAPIImageScraper
from scrapers.fast_image_generator import FastImageGenerator
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
    """Función principal del scraper"""
    parser = argparse.ArgumentParser(description='PreditorIA2025 Fashion Image Scraper')
    parser.add_argument('--images', type=int, default=TARGET_IMAGES, 
                       help=f'Número de imágenes a recolectar (default: {TARGET_IMAGES})')
    parser.add_argument('--demo', action='store_true', 
                       help='Ejecutar en modo demostración')
    parser.add_argument('--api-only', action='store_true',
                       help='Solo usar APIs públicas')
    parser.add_argument('--instagram-only', action='store_true',
                       help='Solo usar Instagram')
    parser.add_argument('--web-only', action='store_true',
                       help='Solo usar web scraping')
    parser.add_argument('--fast', action='store_true',
                       help='Usar generador rápido (recomendado para grandes volúmenes)')
    parser.add_argument('--stats', action='store_true',
                       help='Mostrar estadísticas actuales')
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Mostrar estadísticas si se solicita
        if args.stats:
            show_statistics()
            return
        
        logger.info("Iniciando PreditorIA2025 Fashion Image Scraper")
        logger.info(f"Objetivo: {args.images} imágenes")
        
        # Verificar imágenes existentes
        db = ImageDatabase()
        current_images = db.get_image_count()
        remaining_images = max(0, args.images - current_images)
        
        logger.info(f"Imágenes actuales: {current_images}")
        logger.info(f"Imágenes restantes: {remaining_images}")
        
        if remaining_images == 0:
            logger.info("¡Objetivo alcanzado! No se necesitan más imágenes.")
            show_statistics()
            return
        
        total_collected = 0
        
        # Usar generador rápido para grandes volúmenes o si se especifica
        if args.fast or remaining_images > 100:
            logger.info("Usando generador rápido para acelerar la recolección...")
            fast_generator = FastImageGenerator()
            collected = fast_generator.run_fast_generation(remaining_images)
            total_collected += collected
            logger.info(f"Generación rápida completada: {collected} imágenes procesadas a 224x224 px")
        
        # Distribuir carga entre scrapers tradicionales si quedan imágenes
        elif not any([args.api_only, args.instagram_only, args.web_only, args.fast]):
            # Distribución optimizada: más peso a APIs públicas (más confiables)
            api_images = int(remaining_images * 0.5)  # 50% APIs
            instagram_images = int(remaining_images * 0.3)  # 30% Instagram  
            web_images = remaining_images - api_images - instagram_images  # 20% Web
            
            logger.info(f"Distribución: {api_images} APIs públicas, {instagram_images} Instagram, {web_images} sitios web")
            
            # Scraping de APIs públicas
            if api_images > 0:
                logger.info("Iniciando scraping de APIs públicas...")
                api_scraper = PublicAPIImageScraper()
                collected = api_scraper.run_scraping_session(api_images)
                total_collected += collected
                logger.info(f"APIs públicas completadas: {collected} imágenes")
            
            # Scraping de Instagram
            if instagram_images > 0 and total_collected < remaining_images:
                logger.info("Iniciando scraping de Instagram...")
                instagram_scraper = InstagramScraper()
                collected = instagram_scraper.run_scraping_session(instagram_images)
                total_collected += collected
                logger.info(f"Instagram completado: {collected} imágenes")
            
            # Scraping de sitios web
            if web_images > 0 and total_collected < remaining_images:
                logger.info("Iniciando scraping de sitios web...")
                web_scraper = WebScraper()
                collected = web_scraper.run_scraping_session(web_images)
                total_collected += collected
                logger.info(f"Web scraping completado: {collected} imágenes")
        
        # Scrapers individuales
        else:
            if args.api_only:
                logger.info("Iniciando scraping de APIs públicas...")
                api_scraper = PublicAPIImageScraper()
                collected = api_scraper.run_scraping_session(remaining_images)
                total_collected += collected
            elif args.instagram_only:
                logger.info("Iniciando scraping de Instagram...")
                instagram_scraper = InstagramScraper()
                collected = instagram_scraper.run_scraping_session(remaining_images)
                total_collected += collected
            elif args.web_only:
                logger.info("Iniciando scraping de sitios web...")
                web_scraper = WebScraper()
                collected = web_scraper.run_scraping_session(remaining_images)
                total_collected += collected
        
        logger.info("Scraping completado!")
        show_statistics()
        
    except KeyboardInterrupt:
        logger.info("Scraping interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en el scraping: {e}")
        raise

if __name__ == "__main__":
    main()
