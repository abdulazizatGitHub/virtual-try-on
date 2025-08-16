import numpy as np
import cv2
import torch
from typing import Tuple

from ..utils.logger import Logger
try:
    from rembg import remove  # type: ignore
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False


class ClothMaskBuilder:
    """Generates a robust binary mask for the clothing image.

    Strategy:
    - Edge-based initialization + GrabCut
    - Post-refinement with morphology and smoothing
    - Output (1, H, W) tensor-like numpy float32 mask in [0,1]
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 192)) -> None:
        self.logger = Logger(__name__)
        self.out_h, self.out_w = image_size

    def build(self, cloth_rgb01: np.ndarray) -> np.ndarray:
        assert cloth_rgb01.shape == (self.out_h, self.out_w, 3)

        img_u8 = (cloth_rgb01 * 255).astype(np.uint8)

        # First try rembg for high-quality product cutout
        if _HAS_REMBG:
            try:
                rgba = remove(img_u8)  # returns RGBA if background removed
                if rgba.shape[-1] == 4:
                    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
                    if alpha.sum() > 500:
                        return np.clip(alpha, 0.0, 1.0)
            except Exception:
                pass

        # Initial mask via edge magnitude
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Rectangle for GrabCut: focus center area
        rect = (int(self.out_w * 0.05), int(self.out_h * 0.05), int(self.out_w * 0.90), int(self.out_h * 0.90))
        mask = np.zeros((self.out_h, self.out_w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_u8, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            # treat edges as sure foreground
            mask[edges > 0] = cv2.GC_FGD
            cv2.grabCut(img_u8, mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
            mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.float32)
        except Exception:
            # fallback simple threshold
            _, mask_bin = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
            mask_bin = mask_bin.astype(np.float32)

        # Morphology + smoothing
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_bin = cv2.GaussianBlur(mask_bin, (7, 7), 0)
        mask_bin = np.clip(mask_bin, 0.0, 1.0)

        # Ensure non-empty mask
        if mask_bin.sum() < 200:
            self.logger.warning("Cloth mask extremely small; using centered rectangle fallback.")
            mask_bin[:] = 0
            mask_bin[int(self.out_h*0.2):int(self.out_h*0.8), int(self.out_w*0.2):int(self.out_w*0.8)] = 1.0

        return mask_bin


