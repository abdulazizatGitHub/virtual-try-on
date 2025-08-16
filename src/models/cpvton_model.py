import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import cv2

from ..utils.logger import Logger
import config

# Import the real CP-VTON network architectures
from .networks import GMM, UnetGenerator, load_checkpoint
from ..preprocessing.preprocessing import AgnosticBuilder, ClothMaskBuilder

class CPVTONModel:
    """
    Real CP-VTON+ model implementation for virtual try-on generation.
    Uses the actual network architectures from the CP-VTON repository.
    """
    
    def __init__(self):
        self.logger = Logger(__name__)
        self.device = config.device
        self.logger.info(f"Initializing CP-VTON+ model on device: {self.device}")
        
        # Model paths
        self.checkpoint_dir = Path("CP-VTON/checkpoints")
        self.gmm_checkpoint = self.checkpoint_dir / "GMM" / "gmm_final.pth"
        self.tom_checkpoint = self.checkpoint_dir / "TOM" / "tom_final.pth"
        
        # Model components
        self.gmm_model = None
        self.tom_model = None
        self.is_loaded = False
        
        # CP-VTON configuration
        self.fine_width = 192
        self.fine_height = 256
        self.grid_size = 5
        
        # Create options object for GMM
        self.gmm_options = type('Options', (), {
            'fine_height': self.fine_height,
            'fine_width': self.fine_width,
            'grid_size': self.grid_size
        })()
        
        # Builders for preprocessing
        self.agnostic_builder = AgnosticBuilder(device=self.device, image_size=(self.fine_height, self.fine_width))
        self.cloth_mask_builder = ClothMaskBuilder(image_size=(self.fine_height, self.fine_width))

        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load GMM and TOM models from checkpoints."""
        try:
            self.logger.info("Loading CP-VTON+ models from checkpoints...")
            
            # Check if checkpoints exist
            if not self.gmm_checkpoint.exists():
                raise FileNotFoundError(f"GMM checkpoint not found: {self.gmm_checkpoint}")
            
            if not self.tom_checkpoint.exists():
                raise FileNotFoundError(f"TOM checkpoint not found: {self.tom_checkpoint}")
            
            self.logger.info(f"GMM checkpoint: {self.gmm_checkpoint}")
            self.logger.info(f"TOM checkpoint: {self.tom_checkpoint}")
            
            # Load GMM model (Geometric Matching Module)
            self.gmm_model = self._load_gmm_model()
            self.logger.info("GMM model loaded successfully")
            
            # Load TOM model (Try-On Module)
            self.tom_model = self._load_tom_model()
            self.logger.info("TOM model loaded successfully")
            
            self.is_loaded = True
            self.logger.info("CP-VTON+ models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.is_loaded = False
            raise
    
    def _load_gmm_model(self):
        """Load the Geometric Matching Module."""
        try:
            self.logger.info("Creating GMM model...")
            # Create GMM model with correct architecture
            # GMM expects an options object with fine_height, fine_width, and grid_size
            gmm_model = GMM(self.gmm_options)
            self.logger.info(f"GMM model created with options: {self.gmm_options.__dict__}")
            
            self.logger.info(f"Loading GMM checkpoint from: {self.gmm_checkpoint}")
            # Load checkpoint using the repository's load_checkpoint function
            load_checkpoint(gmm_model, str(self.gmm_checkpoint))
            self.logger.info("GMM checkpoint loaded successfully")
            
            # Move to device and set to eval mode
            gmm_model.to(self.device)
            gmm_model.eval()
            self.logger.info(f"GMM model moved to device: {self.device}")
            
            return gmm_model
            
        except Exception as e:
            self.logger.error(f"Error loading GMM model: {str(e)}")
            raise
    
    def _load_tom_model(self):
        """Load the Try-On Module."""
        try:
            self.logger.info("Creating TOM model...")
            # Create TOM model with correct architecture (CP-VTON+ uses 26 input channels)
            # Use InstanceNorm2d to match the original CP-VTON+ checkpoint
            tom_model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
            self.logger.info("TOM model created successfully")
            
            self.logger.info(f"Loading TOM checkpoint from: {self.tom_checkpoint}")
            # Load checkpoint using the repository's load_checkpoint function
            load_checkpoint(tom_model, str(self.tom_checkpoint))
            self.logger.info("TOM checkpoint loaded successfully")
            
            # Move to device and set to eval mode
            tom_model.to(self.device)
            tom_model.eval()
            self.logger.info(f"TOM model moved to device: {self.device}")
            
            return tom_model
            
        except Exception as e:
            self.logger.error(f"Error loading TOM model: {str(e)}")
            raise
    
    def generate_tryon(self, person_image: np.ndarray, clothing_image: np.ndarray) -> np.ndarray:
        """
        Generate virtual try-on result using real CP-VTON+ models.
        
        Args:
            person_image: Preprocessed person image (256, 192, 3) normalized [0,1]
            clothing_image: Preprocessed clothing image (256, 192, 3) normalized [0,1]
            
        Returns:
            Generated try-on result image (256, 192, 3) normalized [0,1]
        """
        try:
            if not self.is_loaded:
                raise RuntimeError("Models not loaded. Cannot generate try-on.")
            
            self.logger.info("Generating try-on result using CP-VTON+ models")
            
            # Validate input shapes
            if person_image.shape != (256, 192, 3):
                raise ValueError(f"Person image must be (256, 192, 3), got {person_image.shape}")
            
            if clothing_image.shape != (256, 192, 3):
                raise ValueError(f"Clothing image must be (256, 192, 3), got {clothing_image.shape}")
            
            # Validate input ranges
            if person_image.min() < 0 or person_image.max() > 1:
                self.logger.warning(f"Person image range [{person_image.min():.3f}, {person_image.max():.3f}] - normalizing to [0,1]")
                person_image = np.clip(person_image, 0, 1)
            
            if clothing_image.min() < 0 or clothing_image.max() > 1:
                self.logger.warning(f"Clothing image range [{clothing_image.min():.3f}, {clothing_image.max():.3f}] - normalizing to [0,1]")
                clothing_image = np.clip(clothing_image, 0, 1)
            
            # Convert to PyTorch tensors
            person_tensor = self._numpy_to_tensor(person_image)
            clothing_tensor = self._numpy_to_tensor(clothing_image)
            
            # Generate try-on result
            with torch.no_grad():
                result = self._forward_pass(person_tensor, clothing_tensor)
            
            # Convert back to numpy and map from [-1,1] to [0,1]
            result_numpy = self._tensor_to_numpy(result)
            result_numpy = np.clip(result_numpy, 0, 1)
            
            self.logger.info("CP-VTON+ try-on generation completed successfully")
            return result_numpy
            
        except Exception as e:
            self.logger.error(f"Error in try-on generation: {str(e)}")
            raise
    
    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        # Convert from (H, W, C) to (C, H, W) and add batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        # CP-VTON normalizes images to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        # Remove batch dimension and convert from (C, H, W) to (H, W, C)
        tensor = tensor.squeeze(0).cpu()
        numpy_array = tensor.permute(1, 2, 0).numpy()
        return numpy_array
    
    def _forward_pass(self, person: torch.Tensor, clothing: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GMM and TOM models.
        
        Args:
            person: Person image tensor (1, 3, 256, 192)
            clothing: Clothing image tensor (1, 3, 256, 192)
            
        Returns:
            Try-on result tensor (1, 3, 256, 192)
        """
        try:
            self.logger.info(f"Starting forward pass with shapes: person {person.shape}, clothing {clothing.shape}")
            
            # Step 1: GMM - Geometric Matching Module
            # Create agnostic representation (22 channels) using proper builder
            agnostic = self._create_agnostic_representation(person)
            self.logger.info(f"Agnostic representation shape: {agnostic.shape}")
            
            # Create cloth mask from clothing (1 channel)
            cloth_mask = self._create_cloth_mask(clothing)
            self.logger.info(f"Cloth mask shape: {cloth_mask.shape}")
            
            # GMM expects: inputA (agnostic) and inputB (cloth mask)
            # Get geometric transformation from GMM
            grid, theta = self.gmm_model(agnostic, cloth_mask)
            self.logger.info(f"GMM output - grid shape: {grid.shape}, theta shape: {theta.shape}")
            
            # Ensure grid is in the correct format for grid_sample: (B, H, W, 2)
            if grid.dim() == 4 and grid.shape[1] == 2:
                # Grid is in (B, 2, H, W) format, need to transpose to (B, H, W, 2)
                grid = grid.permute(0, 2, 3, 1)
                self.logger.info(f"Grid transposed to shape: {grid.shape}")
            
            # Apply geometric transformation to clothing
            warped_clothing = F.grid_sample(clothing, grid, padding_mode='border', align_corners=True)
            warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros', align_corners=True)
            self.logger.info(f"Warped clothing shape: {warped_clothing.shape}, warped mask shape: {warped_mask.shape}")
            
            # Step 2: TOM - Try-On Module
            # Concatenate agnostic, warped clothing, and warped mask for TOM input
            # TOM expects: 22 (agnostic) + 3 (warped clothing) + 1 (warped mask) = 26 channels
            tom_input = torch.cat([agnostic, warped_clothing, warped_mask], dim=1)  # (1, 26, 256, 192)
            self.logger.info(f"TOM input shape: {tom_input.shape}")
            
            # Validate TOM input channels
            if tom_input.shape[1] != 26:
                raise ValueError(f"TOM input must have 26 channels, got {tom_input.shape[1]}")
            
            # Generate final try-on result
            outputs = self.tom_model(tom_input)
            self.logger.info(f"TOM output shape: {outputs.shape}")
            
            # Split outputs: p_rendered and m_composite
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            self.logger.info(f"Split outputs - p_rendered: {p_rendered.shape}, m_composite: {m_composite.shape}")
            
            # Debug: Check value ranges
            self.logger.info(f"p_rendered range: [{p_rendered.min():.3f}, {p_rendered.max():.3f}]")
            self.logger.info(f"m_composite range: [{m_composite.min():.3f}, {m_composite.max():.3f}]")
            self.logger.info(f"warped_clothing range: [{warped_clothing.min():.3f}, {warped_clothing.max():.3f}]")
            
            # Apply activation functions as per CP-VTON+ paper
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            
            # Debug: Check activation ranges
            self.logger.info(f"After activation - p_rendered range: [{p_rendered.min():.3f}, {p_rendered.max():.3f}]")
            self.logger.info(f"After activation - m_composite range: [{m_composite.min():.3f}, {m_composite.max():.3f}]")
            
            # Final try-on result with better composition
            # Ensure m_composite is properly normalized
            m_composite = torch.clamp(m_composite, 0.01, 0.99)  # Avoid extreme values
            
            # Blend the warped clothing with the rendered result
            p_tryon = warped_clothing * m_composite + p_rendered * (1 - m_composite)
            
            # Ensure final result is in valid range [-1,1] then map to [0,1] at the very end
            p_tryon = torch.clamp(p_tryon, -1, 1)
            
            self.logger.info(f"Final try-on result shape: {p_tryon.shape}")
            self.logger.info(f"Final result range: [{p_tryon.min():.3f}, {p_tryon.max():.3f}]")
            return p_tryon
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def _create_agnostic_representation(self, person: torch.Tensor) -> torch.Tensor:
        """Create agnostic representation (1,22,H,W) using pose + segmentation builder."""
        try:
            # Convert torch tensor [0,1] (1,3,H,W) to numpy HWC
            np_img = person.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            agnostic = self.agnostic_builder.build(np_img)
            # Safety: clamp and ensure shape
            agnostic = torch.clamp(agnostic, 0.0, 1.0)
            return agnostic
        except Exception as e:
            self.logger.error(f"Error creating agnostic representation: {str(e)}")
            # Fallback to repeating grayscale
            gray = 0.299 * person[:, 0:1, :, :] + 0.587 * person[:, 1:2, :, :] + 0.114 * person[:, 2:3, :, :]
            return gray.repeat(1, 22, 1, 1)
    
    def _create_cloth_mask(self, clothing: torch.Tensor) -> torch.Tensor:
        """Create cloth mask using GrabCut-based builder."""
        try:
            np_img = clothing.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            mask_np = self.cloth_mask_builder.build(np_img)  # (H,W) float32 [0,1]
            mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
            return mask
        except Exception as e:
            self.logger.error(f"Error creating cloth mask: {str(e)}")
            return torch.ones_like(clothing[:, :1, :, :])
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.is_loaded
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_type": "CP-VTON+",
            "status": "loaded" if self.is_loaded else "fallback",
            "loaded": self.is_loaded,
            "checkpoint_path": str(self.checkpoint_dir),
            "gmm_checkpoint": str(self.gmm_checkpoint),
            "tom_checkpoint": str(self.tom_checkpoint),
            "device": str(self.device),
            "fine_width": self.fine_width,
            "fine_height": self.fine_height,
            "grid_size": self.grid_size,
            "note": "Real CP-VTON+ implementation with actual network architectures"
        }
