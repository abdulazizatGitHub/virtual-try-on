import numpy as np
import cv2
from typing import Tuple

from ..utils.logger import Logger

try:
    # Simple, lightweight parser using selfie segmentation as fallback.
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False


class HumanParsing:
    """Generates a coarse parsing needed by CP-VTON: head/arms retained, upper-cloth removed.

    For exact CP-VTON replication, SCHP or Graphonomy should be used. This module provides a
    drop-in abstraction with a MediaPipe fallback while we optionally integrate SCHP weights.
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 192)) -> None:
        self.logger = Logger(__name__)
        self.out_h, self.out_w = image_size
        if _HAS_MEDIAPIPE:
            self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)  # type: ignore
        else:
            self.selfie = None

    def parse(self, rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (shape_mask, keep_mask) in [0,1].

        - shape_mask: full person silhouette
        - keep_mask: head+arms approximate mask (to preserve these regions)
        """
        h, w, _ = rgb01.shape
        if self.selfie is not None:
            res = self.selfie.process((rgb01 * 255).astype(np.uint8))  # type: ignore
            shape = res.segmentation_mask.astype(np.float32)
        else:
            gray = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, shape = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
            shape = shape.astype(np.float32)

        # crude head+arms from shape’s upper region bands
        keep = np.zeros_like(shape)
        keep[: int(0.35 * h), :] = shape[: int(0.35 * h), :]
        # add side bands where arms are likely
        band_w = int(0.25 * w)
        keep[:, :band_w] = np.maximum(keep[:, :band_w], shape[:, :band_w])
        keep[:, -band_w:] = np.maximum(keep[:, -band_w:], shape[:, -band_w:])

        # smooth
        shape = cv2.GaussianBlur(shape, (11, 11), 0)
        keep = cv2.GaussianBlur(keep, (11, 11), 0)
        shape = np.clip(shape, 0.0, 1.0)
        keep = np.clip(keep, 0.0, 1.0)
        return shape, keep


