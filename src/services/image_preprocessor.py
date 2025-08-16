import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Union, Optional
from pathlib import Path

from src.utils.logger import Logger

class ImagePreprocessor:
    """
    Image preprocessing service for CP-VTON model.
    Handles resizing, normalization, and format conversion for optimal performance.
    """
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.target_size = (192, 256)  # (width, height) for CP-VTON
        self.logger.info(f"Initialized ImagePreprocessor with target size: {self.target_size}")
    
    def preprocess_image(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image],
        image_type: str = "unknown"
    ) -> np.ndarray:
        """
        Main preprocessing function that handles different input types and resizes to target resolution.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            image_type: Type of image for logging purposes (e.g., "person", "clothing")
            
        Returns:
            Preprocessed image as numpy array with shape (256, 192, 3)
        """
        try:
            # Load image if it's a path
            if isinstance(image, (str, Path)):
                img_array = self._load_image(image)
            elif isinstance(image, Image.Image):
                img_array = np.array(image)
            elif isinstance(image, np.ndarray):
                img_array = image.copy()
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Validate image dimensions
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Image must be RGB with 3 channels, got shape: {img_array.shape}")
            
            # Resize to target resolution
            resized_img = self._resize_image(img_array)
            
            # Normalize pixel values
            normalized_img = self._normalize_image(resized_img)
            
            self.logger.info(f"Successfully preprocessed {image_type} image to {self.target_size}")
            return normalized_img
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {image_type} image: {str(e)}")
            raise
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path using OpenCV."""
        try:
            # Load with OpenCV (BGR format)
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image from path: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
            
        except Exception as e:
            self.logger.error(f"Error loading image from {image_path}: {str(e)}")
            raise
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target resolution while maintaining aspect ratio.
        Uses padding to avoid distortion.
        """
        try:
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            # Calculate scaling factor to fit image within target dimensions
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image with target dimensions
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Calculate padding to center the image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image in center of padded image
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            self.logger.debug(f"Resized image from {w}x{h} to {target_w}x{target_h} with padding")
            return padded
            
        except Exception as e:
            self.logger.error(f"Error resizing image: {str(e)}")
            raise
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1] range for optimal model performance.
        """
        try:
            # Convert to float and normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            
            # Ensure values are in valid range
            normalized = np.clip(normalized, 0.0, 1.0)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing image: {str(e)}")
            raise
    
    def preprocess_person_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess person image specifically for CP-VTON."""
        return self.preprocess_image(image, "person")
    
    def preprocess_clothing_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess clothing image specifically for CP-VTON."""
        return self.preprocess_image(image, "clothing")
    
    def batch_preprocess(
        self, 
        images: list, 
        image_types: Optional[list] = None
    ) -> list:
        """
        Preprocess multiple images in batch for efficiency.
        
        Args:
            images: List of input images
            image_types: Optional list of image types for logging
            
        Returns:
            List of preprocessed images
        """
        if image_types is None:
            image_types = [f"image_{i}" for i in range(len(images))]
        
        if len(images) != len(image_types):
            raise ValueError("Number of images must match number of image types")
        
        preprocessed = []
        for img, img_type in zip(images, image_types):
            try:
                processed = self.preprocess_image(img, img_type)
                preprocessed.append(processed)
            except Exception as e:
                self.logger.error(f"Failed to preprocess {img_type}: {str(e)}")
                raise
        
        self.logger.info(f"Successfully preprocessed {len(preprocessed)} images")
        return preprocessed
    
    def get_target_size(self) -> Tuple[int, int]:
        """Get the target image size (width, height)."""
        return self.target_size
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate that an image meets the requirements for CP-VTON.
        
        Args:
            image: Image to validate
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check shape
            if len(image.shape) != 3:
                return False
            
            # Check channels
            if image.shape[2] != 3:
                return False
            
            # Check dimensions
            if image.shape[:2] != self.target_size[::-1]:  # (height, width) vs (width, height)
                return False
            
            # Check data type and range
            if image.dtype != np.float32:
                return False
            
            if np.min(image) < 0 or np.max(image) > 1:
                return False
            
            return True
            
        except Exception:
            return False
