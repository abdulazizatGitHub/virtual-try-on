import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple

from ..utils.logger import Logger
from .openpose_estimator import OpenPoseEstimator
from .schp_parser import SCHPParser

try:
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:  # pragma: no cover
    _HAS_MEDIAPIPE = False


class AgnosticBuilder:
    """Builds the 22-channel person agnostic representation expected by CP-VTON.

    Channels layout (H, W = 256, 192):
    - 0..2: RGB person image with torso/upper-clothes region removed (head/arms kept)
    - 3..20: 18 pose heatmaps (gaussian blobs on COCO-like keypoints)
    - 21: shape mask (coarse person silhouette)
    """

    def __init__(self, device: torch.device, image_size: Tuple[int, int] = (256, 192)) -> None:
        self.logger = Logger(__name__)
        self.device = device
        self.out_h, self.out_w = image_size
        # CP-VTON exact preprocessing tools
        self.pose_estimator = OpenPoseEstimator(image_size=image_size)
        self.human_parser = SCHPParser(image_size=image_size)
        if _HAS_MEDIAPIPE:
            self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)  # type: ignore
        else:
            self.mp_selfie = None

    # ---- Public API ----
    def build(self, person_rgb01: np.ndarray) -> torch.Tensor:
        """Create a (1, 22, H, W) tensor on the target device from an RGB [0,1] image.

        Note: The CP-VTON pipeline feeds tensors normalized to [-1,1] into networks.
        This builder returns channels in [0,1]; the caller converts to [-1,1] where needed.
        """
        # Ensure expected shape
        assert person_rgb01.shape == (self.out_h, self.out_w, 3), f"Expected {(self.out_h, self.out_w, 3)}, got {person_rgb01.shape}"

        # CP-VTON exact preprocessing: SCHP parsing + OpenPose heatmaps
        parsing_map = self.human_parser.parse_human(person_rgb01)  # (H,W) class indices
        shape_mask, keep_mask = self.human_parser.get_agnostic_mask(parsing_map)  # (H,W) [0,1]
        heatmaps = self.pose_estimator.generate_heatmaps(person_rgb01)  # (18,H,W)
        torso_removed = person_rgb01.copy()
        torso_mask = np.clip(shape_mask - keep_mask, 0, 1)
        torso_removed[torso_mask > 0.5] = 0.0

        # Stack channels
        rgb_tensor = torch.from_numpy(torso_removed).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        heat_tensor = torch.from_numpy(heatmaps).unsqueeze(0).to(self.device, dtype=torch.float32)
        shape_tensor = torch.from_numpy(shape_mask).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

        agnostic = torch.cat([rgb_tensor, heat_tensor, shape_tensor], dim=1)
        return agnostic

    # ---- Helpers ----
    # old helper methods replaced by dedicated PoseEstimator and HumanParsing


