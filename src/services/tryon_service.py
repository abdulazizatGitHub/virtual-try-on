import os
import uuid
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2

from .image_preprocessor import ImagePreprocessor
from ..utils.logger import Logger
from ..models.cpvton_model import CPVTONModel
import config

class TryOnService:
    """
    Service for handling virtual try-on operations using CP-VTON model.
    Manages the complete workflow from image preprocessing to result generation.
    """
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.preprocessor = ImagePreprocessor()
        self.model = CPVTONModel()
        self.logger.info("TryOnService initialized successfully")
    
    def process_tryon(
        self, 
        person_image_path: str, 
        clothing_image_path: str,
        output_filename: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Process virtual try-on request.
        
        Args:
            person_image_path: Path to the uploaded person image
            clothing_image_path: Path to the uploaded clothing image
            output_filename: Optional custom filename for output
            
        Returns:
            Tuple of (output_image_path, output_url)
        """
        try:
            self.logger.info(f"Starting try-on process for person: {person_image_path}, clothing: {clothing_image_path}")
            
            # Validate input files
            self._validate_input_files(person_image_path, clothing_image_path)
            
            # Preprocess images
            person_processed = self.preprocessor.preprocess_person_image(person_image_path)
            clothing_processed = self.preprocessor.preprocess_clothing_image(clothing_image_path)
            
            self.logger.info("Images preprocessed successfully")
            
            # Run CP-VTON model
            result_image = self.model.generate_tryon(person_processed, clothing_processed)
            
            self.logger.info("CP-VTON model completed successfully")
            
            # Save result
            output_path, output_url = self._save_result(result_image, output_filename)
            
            self.logger.info(f"Try-on completed successfully. Output saved to: {output_path}")
            
            return output_path, output_url
            
        except Exception as e:
            self.logger.error(f"Error in try-on process: {str(e)}")
            raise
    
    def _validate_input_files(self, person_path: str, clothing_path: str) -> None:
        """Validate that input files exist and are valid images."""
        try:
            # Check if files exist
            if not os.path.exists(person_path):
                raise ValueError(f"Person image not found: {person_path}")
            
            if not os.path.exists(clothing_path):
                raise ValueError(f"Clothing image not found: {clothing_path}")
            
            # Validate file types
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            person_ext = Path(person_path).suffix.lower()
            clothing_ext = Path(clothing_path).suffix.lower()
            
            if person_ext not in valid_extensions:
                raise ValueError(f"Invalid person image format: {person_ext}. Supported: {valid_extensions}")
            
            if clothing_ext not in valid_extensions:
                raise ValueError(f"Invalid clothing image format: {clothing_ext}. Supported: {valid_extensions}")
            
            # Validate image dimensions (basic check)
            person_img = Image.open(person_path)
            clothing_img = Image.open(clothing_path)
            
            if person_img.size[0] < 100 or person_img.size[1] < 100:
                raise ValueError(f"Person image too small: {person_img.size}. Minimum: 100x100")
            
            if clothing_img.size[0] < 50 or clothing_img.size[1] < 50:
                raise ValueError(f"Clothing image too small: {clothing_img.size}. Minimum: 50x50")
            
            self.logger.info("Input file validation passed")
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            raise
    
    def _save_result(self, result_image: np.ndarray, custom_filename: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the result image and return file path and URL.
        
        Args:
            result_image: Processed image as numpy array
            custom_filename: Optional custom filename
            
        Returns:
            Tuple of (file_path, file_url)
        """
        try:
            # Generate filename
            if custom_filename:
                filename = f"{custom_filename}.png"
            else:
                filename = f"tryon_result_{uuid.uuid4().hex[:8]}.png"
            
            # Ensure results directory exists
            config.path_config.RESULTS.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Results directory: {config.path_config.RESULTS} (exists: {config.path_config.RESULTS.exists()})")
            
            # Save image
            output_path = config.path_config.RESULTS / filename
            self.logger.info(f"Attempting to save to: {output_path}")
            
            # Convert from normalized [0,1] to [0,255] for saving
            if result_image.dtype == np.float32 and result_image.max() <= 1.0:
                save_image = (result_image * 255).astype(np.uint8)
            else:
                save_image = result_image.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(save_image.shape) == 3 and save_image.shape[2] == 3:
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            
            # Save the image
            success = cv2.imwrite(str(output_path), save_image)
            if not success:
                raise RuntimeError(f"Failed to save image to {output_path}")
            
            # Verify file was created
            if output_path.exists():
                self.logger.info(f"File successfully saved: {output_path} (size: {output_path.stat().st_size} bytes)")
            else:
                raise RuntimeError(f"File was not created at {output_path}")
            
            # Generate URL (relative to API base)
            output_url = f"/results/{filename}"
            
            self.logger.info(f"Result saved to: {output_path}")
            
            return str(output_path), output_url
            
        except Exception as e:
            self.logger.error(f"Error saving result: {str(e)}")
            raise
    
    def cleanup_temp_files(self, *file_paths: str) -> None:
        """Clean up temporary uploaded files."""
        try:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up files: {str(e)}")
    
    def get_processing_status(self) -> dict:
        """Get current processing status and model information."""
        return {
            "status": "ready",
            "model": "CP-VTON+",
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"],
            "target_resolution": self.preprocessor.get_target_size(),
            "max_file_size": "10MB"
        }
