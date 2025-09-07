"""
Image processing utilities for Fashion Trend Analysis App
"""

import os
import logging
from PIL import Image
from typing import List, Optional
import sys

# Add fashion_clustering to path
sys.path.append('/home/jose/PreditorIA2025')
from fashion_clustering.utils.colors import extract_dominant_colors, get_color_names

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self, allowed_extensions: set = None):
        self.allowed_extensions = allowed_extensions or {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def validate_image(self, image_path: str) -> bool:
        """Validate that the file is a valid image"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"❌ Invalid image file {image_path}: {e}")
            return False
    
    def get_image_info(self, image_path: str) -> Optional[dict]:
        """Get basic image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height
                }
        except Exception as e:
            logger.error(f"❌ Error getting image info: {e}")
            return None
    
    def analyze_colors(self, image_path: str, n_colors: int = 5) -> List[str]:
        """Analyze dominant colors in the image"""
        try:
            image = Image.open(image_path)
            colors = extract_dominant_colors(image, n_colors=n_colors)
            color_names = get_color_names(colors)
            return color_names
        except Exception as e:
            logger.error(f"❌ Error analyzing colors: {e}")
            return []
    
    def resize_image(self, image_path: str, target_size: tuple = (224, 224), 
                    output_path: Optional[str] = None) -> Optional[str]:
        """Resize image to target size"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save resized image
                if output_path is None:
                    output_path = image_path.replace('.', '_resized.')
                
                resized_img.save(output_path, 'JPEG', quality=95)
                return output_path
                
        except Exception as e:
            logger.error(f"❌ Error resizing image: {e}")
            return None
    
    def create_thumbnail(self, image_path: str, size: tuple = (150, 150), 
                        output_path: Optional[str] = None) -> Optional[str]:
        """Create a thumbnail of the image"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                if output_path is None:
                    name, ext = os.path.splitext(image_path)
                    output_path = f"{name}_thumb{ext}"
                
                img.save(output_path, 'JPEG', quality=90)
                return output_path
                
        except Exception as e:
            logger.error(f"❌ Error creating thumbnail: {e}")
            return None
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"❌ Error getting file size: {e}")
            return 0
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"✅ Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.error(f"❌ Error cleaning up {file_path}: {e}")


