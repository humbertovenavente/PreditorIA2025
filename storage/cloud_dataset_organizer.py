#!/usr/bin/env python3
"""
Organizador de Dataset en la Nube para Entrenamiento MobileNet V2
Organiza las imágenes en train/validation/test en Google Cloud Storage
"""

import os
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys

# Agregar el directorio raíz al path para imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from storage.cloud_storage import GoogleCloudStorage
from config.settings import DIRECTORIES

class CloudDatasetOrganizer:
    """Organiza el dataset dividido en la nube para entrenamiento"""
    
    def __init__(self):
        self.gcs = GoogleCloudStorage()
        self.logger = logging.getLogger(__name__)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Estructura de directorios en la nube
        self.cloud_structure = {
            'train': 'processed/train',
            'validation': 'processed/validation', 
            'test': 'processed/test',
            'augmented_train': 'augmented/train',
            'augmented_validation': 'augmented/validation',
            'augmented_test': 'augmented/test',
            'metadata': 'metadata'
        }
    
    def organize_dataset_in_cloud(self):
        """Organiza el dataset completo en la nube con división train/val/test"""
        self.logger.info(" Iniciando organización del dataset en la nube...")
        
        if not self.gcs.is_enabled():
            self.logger.error(" Google Cloud Storage no está habilitado")
            return False
        
        try:
            # Paso 1: Verificar división local
            self.logger.info(" Verificando división local del dataset...")
            local_splits = self._verify_local_splits()
            
            if not local_splits:
                self.logger.error(" No se encontró división local del dataset")
                return False
            
            # Paso 2: Crear estructura en la nube
            self.logger.info("Creando estructura de directorios en la nube...")
            self._create_cloud_structure()
            
            # Paso 3: Subir imágenes divididas
            self.logger.info(" Subiendo imágenes divididas a la nube...")
            upload_results = self._upload_split_images(local_splits)
            
            # Paso 4: Subir metadatos organizados
            self.logger.info(" Subiendo metadatos organizados...")
            self._upload_organized_metadata(local_splits)
            
            # Paso 5: Generar reporte de organización
            self.logger.info(" Generando reporte de organización...")
            self._generate_organization_report(upload_results)
            
            self.logger.info(" Dataset organizado exitosamente en la nube!")
            return True
            
        except Exception as e:
            self.logger.error(f" Error organizando dataset en la nube: {e}")
            return False
    
    def _verify_local_splits(self):
        """Verifica que existan las divisiones locales del dataset"""
        processed_dir = Path(DIRECTORIES['processed'])
        
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            split_dir = processed_dir / split_name
            if split_dir.exists():
                images = list(split_dir.glob('*.jpg'))
                splits[split_name] = {
                    'local_path': split_dir,
                    'image_count': len(images),
                    'images': images
                }
                self.logger.info(f"{split_name}: {len(images)} imágenes encontradas")
            else:
                self.logger.warning(f"{split_name}: directorio no encontrado")
                splits[split_name] = None
        
        return splits
    
    def _create_cloud_structure(self):
        """Crea la estructura de directorios en la nube"""
        for split_name, cloud_path in self.cloud_structure.items():
            try:
                # Crear directorio virtual en GCS (GCS no tiene directorios reales, pero organizamos por prefijos)
                self.logger.info(f" Estructura preparada: {cloud_path}")
            except Exception as e:
                self.logger.warning(f" No se pudo crear estructura {cloud_path}: {e}")
    
    def _upload_split_images(self, local_splits):
        """Sube las imágenes divididas a la nube"""
        upload_results = {}
        
        for split_name, split_info in local_splits.items():
            if not split_info:
                continue
                
            self.logger.info(f" Subiendo {split_name} ({split_info['image_count']} imágenes)...")
            
            # Subir imágenes en lotes
            cloud_prefix = self.cloud_structure[split_name]
            uploaded_count = self._upload_images_batch(
                split_info['images'], 
                cloud_prefix,
                split_name
            )
            
            upload_results[split_name] = {
                'total': split_info['image_count'],
                'uploaded': uploaded_count,
                'success_rate': uploaded_count / split_info['image_count']
            }
            
            self.logger.info(f" {split_name}: {uploaded_count}/{split_info['image_count']} subidas")
        
        return upload_results
    
    def _upload_images_batch(self, image_paths, cloud_prefix, split_name):
        """Sube un lote de imágenes a la nube"""
        uploaded_count = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Preparar trabajos de subida
            future_to_path = {}
            
            for image_path in image_paths:
                cloud_path = f"{cloud_prefix}/{image_path.name}"
                future = executor.submit(self._upload_single_image, image_path, cloud_path)
                future_to_path[future] = image_path
            
            # Procesar resultados
            for future in as_completed(future_to_path):
                try:
                    success = future.result()
                    if success:
                        uploaded_count += 1
                        if uploaded_count % 50 == 0:
                            self.logger.info(f"{split_name}: {uploaded_count} subidas...")
                except Exception as e:
                    self.logger.warning(f"Error subiendo imagen: {e}")
        
        return uploaded_count
    
    def _upload_single_image(self, local_path, cloud_path):
        """Sube una imagen individual a la nube"""
        try:
            result = self.gcs.upload_image(str(local_path), cloud_path)
            return result and result != str(local_path)
        except Exception as e:
            self.logger.debug(f"Error subiendo {local_path}: {e}")
            return False
    
    def _upload_organized_metadata(self, local_splits):
        """Sube metadatos organizados por subconjunto"""
        for split_name, split_info in local_splits.items():
            if not split_info:
                continue
            
            try:
                # Crear metadatos del subconjunto
                split_metadata = {
                    'split_name': split_name,
                    'image_count': split_info['image_count'],
                    'upload_timestamp': time.time(),
                    'cloud_structure': self.cloud_structure[split_name],
                    'images': [
                        {
                            'filename': img.name,
                            'cloud_path': f"{self.cloud_structure[split_name]}/{img.name}",
                            'local_path': str(img),
                            'size_bytes': img.stat().st_size
                        }
                        for img in split_info['images']
                    ]
                }
                
                # Subir metadatos del subconjunto
                metadata_path = f"metadata/{split_name}_metadata.json"
                self.gcs.upload_metadata(split_metadata, metadata_path)
                
                self.logger.info(f" Metadatos de {split_name} subidos: {metadata_path}")
                
            except Exception as e:
                self.logger.error(f"Error subiendo metadatos de {split_name}: {e}")
    
    def _generate_organization_report(self, upload_results):
        """Genera reporte de la organización en la nube"""
        try:
            # Estadísticas generales
            total_images = sum(result['total'] for result in upload_results.values())
            total_uploaded = sum(result['uploaded'] for result in upload_results.values())
            overall_success_rate = total_uploaded / total_images if total_images > 0 else 0
            
            report = {
                'organization_timestamp': time.time(),
                'overall_statistics': {
                    'total_images': total_images,
                    'total_uploaded': total_uploaded,
                    'overall_success_rate': overall_success_rate
                },
                'split_details': upload_results,
                'cloud_structure': self.cloud_structure,
                'bucket_name': self.gcs.bucket_name,
                'project_id': self.gcs.project_id
            }
            
            # Subir reporte a la nube
            report_path = "metadata/dataset_organization_report.json"
            self.gcs.upload_metadata(report, report_path)
            
            self.logger.info(f" Reporte de organización subido: {report_path}")
            
            # Mostrar resumen
            self.logger.info("RESUMEN DE ORGANIZACIÓN:")
            self.logger.info(f"   • Total imágenes: {total_images}")
            self.logger.info(f"   • Total subidas: {total_uploaded}")
            self.logger.info(f"   • Tasa de éxito: {overall_success_rate:.2%}")
            
            for split_name, result in upload_results.items():
                self.logger.info(f"   • {split_name}: {result['uploaded']}/{result['total']} ({result['success_rate']:.2%})")
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
    
    def get_cloud_dataset_info(self):
        """Obtiene información del dataset organizado en la nube"""
        if not self.gcs.is_enabled():
            return None
        
        try:
            info = {
                'bucket_name': self.gcs.bucket_name,
                'project_id': self.gcs.project_id,
                'cloud_structure': self.cloud_structure,
                'split_urls': {}
            }
            
            # Generar URLs para cada subconjunto
            for split_name, cloud_path in self.cloud_structure.items():
                info['split_urls'][split_name] = f"gs://{self.gcs.bucket_name}/{cloud_path}"
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error obteniendo info del dataset en la nube: {e}")
            return None
    
    def verify_cloud_organization(self):
        """Verifica que la organización en la nube esté correcta"""
        if not self.gcs.is_enabled():
            return False
        
        try:
            self.logger.info("🔍 Verificando organización en la nube...")
            
            verification_results = {}
            for split_name, cloud_path in self.cloud_structure.items():
                try:
                    # Contar archivos en cada subconjunto
                    files = self.gcs.list_images(cloud_path)
                    count = len(files)
                    verification_results[split_name] = {
                        'cloud_path': cloud_path,
                        'file_count': count,
                        'status': 'OK' if count > 0 else 'EMPTY'
                    }
                except Exception as e:
                    verification_results[split_name] = {
                        'cloud_path': cloud_path,
                        'file_count': 0,
                        'status': f'ERROR: {e}'
                    }
            
            # Mostrar resultados
            self.logger.info(" VERIFICACIÓN DE ORGANIZACIÓN EN LA NUBE:")
            for split_name, result in verification_results.items():
                status_icon = "check" if result['status'] == 'OK' else "⚠️" if result['status'] == 'EMPTY' else "❌"
                self.logger.info(f"   {status_icon} {split_name}: {result['file_count']} archivos - {result['status']}")
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Error verificando organización: {e}")
            return None

def main():
    """Función principal para organizar dataset en la nube"""
    organizer = CloudDatasetOrganizer()
    
    print("🚀 ORGANIZADOR DE DATASET EN LA NUBE PARA MOBILENET V2")
    print("=" * 60)
    
    # Organizar dataset
    success = organizer.organize_dataset_in_cloud()
    
    if success:
        print("\nDATASET ORGANIZADO EXITOSAMENTE EN LA NUBE!")
        print("\n ESTRUCTURA CREADA:")
        print("   • processed/train/     → Imágenes de entrenamiento")
        print("   • processed/validation/ → Imágenes de validación")
        print("   • processed/test/       → Imágenes de prueba")
        print("   • augmented/train/      → Imágenes aumentadas para entrenamiento")
        print("   • augmented/validation/ → Imágenes aumentadas para validación")
        print("   • augmented/test/       → Imágenes aumentadas para prueba")
        print("   • metadata/             → Metadatos organizados")
        
        # Verificar organización
        print("\n🔍 Verificando organización...")
        verification = organizer.verify_cloud_organization()
        
        # Mostrar información para entrenamiento
        cloud_info = organizer.get_cloud_dataset_info()
        if cloud_info:
            print(f"\n LISTO PARA ENTRENAMIENTO MOBILENET V2:")
            print(f"   • Bucket: {cloud_info['bucket_name']}")
            print(f"   • Proyecto: {cloud_info['project_id']}")
            print(f"   • URLs de subconjuntos disponibles en cloud_info['split_urls']")
        
    else:
        print("\n ERROR ORGANIZANDO DATASET EN LA NUBE")
        print("   Verifica que Google Cloud Storage esté configurado correctamente")

if __name__ == "__main__":
    main()
