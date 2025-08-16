import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import urllib.request
import os
from pathlib import Path

from ..utils.logger import Logger


class SCHPParser:
    """Self-Correction Human Parsing for generating semantic segmentation maps.
    
    This generates the exact parsing format that CP-VTON expects:
    - Background, hair, face, upper-clothes, dress, coat, socks, pants, etc.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 192)) -> None:
        self.logger = Logger(__name__)
        self.out_h, self.out_w = image_size
        self.model = None
        self.device = torch.device('cpu')
        
        # SCHP label mapping (20 classes)
        self.labels = {
            0: 'background',
            1: 'hat',
            2: 'hair', 
            3: 'sunglasses',
            4: 'upper-clothes',
            5: 'skirt',
            6: 'pants',
            7: 'dress',
            8: 'belt',
            9: 'left-shoe',
            10: 'right-shoe',
            11: 'face',
            12: 'left-leg',
            13: 'right-leg',
            14: 'left-arm',
            15: 'right-arm',
            16: 'bag',
            17: 'scarf',
            18: 'coat',
            19: 'sock'
        }
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load SCHP parsing model or fallback."""
        try:
            # For now, we'll use a simplified segmentation approach
            # In production, you'd load actual SCHP weights
            self.logger.info("SCHP parser initialized (lightweight version)")
            
        except Exception as e:
            self.logger.warning(f"Could not load SCHP model: {e}")
            self.model = None
    
    def parse_human(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Generate human parsing segmentation map.
        
        Args:
            rgb_image: RGB image (H, W, 3) in [0, 1] range
            
        Returns:
            parsing_map: (H, W) segmentation map with class indices
        """
        try:
            return self._simple_parsing(rgb_image)
            
        except Exception as e:
            self.logger.warning(f"Human parsing failed: {e}")
            return self._fallback_parsing(rgb_image)
    
    def _simple_parsing(self, rgb_image: np.ndarray) -> np.ndarray:
        """Enhanced parsing using MediaPipe and morphological operations."""
        h, w = rgb_image.shape[:2]
        parsing_map = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Use MediaPipe for better person segmentation
            import mediapipe as mp
            
            # Get person mask using MediaPipe selfie segmentation
            mp_selfie = mp.solutions.selfie_segmentation
            with mp_selfie.SelfieSegmentation(model_selection=1) as selfie_seg:
                img_uint8 = (rgb_image * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                results = selfie_seg.process(img_bgr)
                
                if results.segmentation_mask is not None:
                    # Get clean person mask
                    person_mask = (results.segmentation_mask > 0.7).astype(np.uint8)
                    parsing_map = self._segment_body_parts(img_uint8, person_mask)
                else:
                    parsing_map = self._fallback_parsing_simple(rgb_image)
                    
        except ImportError:
            parsing_map = self._fallback_parsing_simple(rgb_image)
        except Exception as e:
            self.logger.warning(f"MediaPipe parsing failed: {e}")
            parsing_map = self._fallback_parsing_simple(rgb_image)
        
        return parsing_map
    
    def _segment_body_parts(self, img_uint8: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Segment body parts from person mask using enhanced heuristics."""
        h, w = img_uint8.shape[:2]
        parsing_map = np.zeros((h, w), dtype=np.uint8)
        
        if person_mask.sum() == 0:
            return parsing_map
        
        # Find person bounding box
        y_coords, x_coords = np.where(person_mask > 0)
        if len(y_coords) == 0:
            return parsing_map
            
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        center_x = (x_min + x_max) // 2
        
        # Enhanced body part segmentation with anatomical proportions
        
        # 1. Hair region (top 8-15% of person)
        hair_y_end = y_min + int(0.15 * bbox_h)
        hair_mask = person_mask.copy()
        hair_mask[hair_y_end:] = 0
        # Refine hair using color-based segmentation
        hair_mask = self._refine_hair_mask(img_uint8, hair_mask)
        parsing_map[hair_mask > 0] = 2  # hair
        
        # 2. Face region (8-20% of person, central area)
        face_y_start = y_min + int(0.08 * bbox_h)
        face_y_end = y_min + int(0.20 * bbox_h)
        face_x_start = center_x - int(0.15 * bbox_w)
        face_x_end = center_x + int(0.15 * bbox_w)
        face_mask = person_mask.copy()
        face_mask[:face_y_start] = 0
        face_mask[face_y_end:] = 0
        face_mask[:, :face_x_start] = 0
        face_mask[:, face_x_end:] = 0
        # Exclude hair from face
        face_mask[hair_mask > 0] = 0
        parsing_map[face_mask > 0] = 11  # face
        
        # 3. Upper body (torso) - central region 15-60%
        torso_y_start = y_min + int(0.15 * bbox_h)
        torso_y_end = y_min + int(0.60 * bbox_h)
        torso_x_start = center_x - int(0.20 * bbox_w)
        torso_x_end = center_x + int(0.20 * bbox_w)
        torso_mask = person_mask.copy()
        torso_mask[:torso_y_start] = 0
        torso_mask[torso_y_end:] = 0
        torso_mask[:, :torso_x_start] = 0
        torso_mask[:, torso_x_end:] = 0
        parsing_map[torso_mask > 0] = 4  # upper-clothes
        
        # 4. Arms (side regions)
        arm_y_start = y_min + int(0.18 * bbox_h)
        arm_y_end = y_min + int(0.65 * bbox_h)
        
        # Left arm (person's left, image right)
        left_arm_x_start = center_x + int(0.12 * bbox_w)
        left_arm_mask = person_mask.copy()
        left_arm_mask[:arm_y_start] = 0
        left_arm_mask[arm_y_end:] = 0
        left_arm_mask[:, :left_arm_x_start] = 0
        parsing_map[left_arm_mask > 0] = 14  # left-arm
        
        # Right arm (person's right, image left)
        right_arm_x_end = center_x - int(0.12 * bbox_w)
        right_arm_mask = person_mask.copy()
        right_arm_mask[:arm_y_start] = 0
        right_arm_mask[arm_y_end:] = 0
        right_arm_mask[:, right_arm_x_end:] = 0
        parsing_map[right_arm_mask > 0] = 15  # right-arm
        
        # 5. Legs (bottom 35% of person)
        leg_y_start = y_min + int(0.65 * bbox_h)
        leg_mask = person_mask.copy()
        leg_mask[:leg_y_start] = 0
        
        # Split legs at center
        left_leg_mask = leg_mask.copy()
        left_leg_mask[:, :center_x] = 0
        parsing_map[left_leg_mask > 0] = 12  # left-leg
        
        right_leg_mask = leg_mask.copy()
        right_leg_mask[:, center_x:] = 0
        parsing_map[right_leg_mask > 0] = 13  # right-leg
        
        # Apply morphological operations to clean up segmentation
        parsing_map = self._clean_parsing_map(parsing_map)
        
        return parsing_map
    
    def _refine_hair_mask(self, img_uint8: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Refine hair mask using color analysis."""
        if initial_mask.sum() == 0:
            return initial_mask
        
        # Simple hair color detection (darker regions in the hair area)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        hair_region = gray[initial_mask > 0]
        
        if len(hair_region) > 0:
            # Hair is typically darker than skin
            hair_threshold = np.percentile(hair_region, 60)  # Keep darker 40%
            refined_mask = initial_mask.copy()
            refined_mask[(initial_mask > 0) & (gray > hair_threshold)] = 0
            return refined_mask
        
        return initial_mask
    
    def _clean_parsing_map(self, parsing_map: np.ndarray) -> np.ndarray:
        """Clean up parsing map using morphological operations."""
        cleaned = parsing_map.copy()
        
        # Small morphological closing for each label
        for label in np.unique(parsing_map):
            if label == 0:  # Skip background
                continue
            
            mask = (parsing_map == label).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            cleaned[mask > 0] = label
        
        return cleaned
    
    def _fallback_parsing_simple(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback parsing using basic thresholding."""
        h, w = rgb_image.shape[:2]
        parsing_map = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to uint8 and create basic person mask
        img_uint8 = (rgb_image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        _, person_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Fill with upper-clothes label as default
        parsing_map[person_mask > 0] = 4
        
        return parsing_map
    
    def _fallback_parsing(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback parsing when segmentation fails."""
        h, w = rgb_image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)  # All background
    
    def get_agnostic_mask(self, parsing_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate masks for CP-VTON agnostic representation.
        
        Args:
            parsing_map: (H, W) segmentation map
            
        Returns:
            shape_mask: (H, W) person silhouette mask [0, 1]
            keep_mask: (H, W) mask for regions to keep (head, arms) [0, 1]
        """
        # Person silhouette (all non-background)
        shape_mask = (parsing_map > 0).astype(np.float32)
        
        # Keep regions: face, hair, arms, legs (not upper-clothes)
        keep_classes = [2, 11, 14, 15, 12, 13]  # hair, face, left-arm, right-arm, left-leg, right-leg
        keep_mask = np.zeros_like(parsing_map, dtype=np.float32)
        
        for class_id in keep_classes:
            keep_mask[parsing_map == class_id] = 1.0
        
        # Smooth masks
        shape_mask = cv2.GaussianBlur(shape_mask, (5, 5), 0)
        keep_mask = cv2.GaussianBlur(keep_mask, (5, 5), 0)
        
        # Ensure values in [0, 1]
        shape_mask = np.clip(shape_mask, 0, 1)
        keep_mask = np.clip(keep_mask, 0, 1)
        
        return shape_mask, keep_mask
