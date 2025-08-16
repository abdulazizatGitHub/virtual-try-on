# preprocessing/preprocessing.py
import os
from typing import Tuple
import numpy as np
import cv2
import torch

from .human_parser import HumanParser
from .pose_estimator import PoseEstimator
from .image_preprocessor import ImagePreprocessor


class PreprocessingPipeline:
    def __init__(self, device="cuda"):
        self.parser = HumanParser(device=device)
        self.pose_estimator = PoseEstimator(device=device)
        self.img_preprocessor = ImagePreprocessor()

    def run(self, person_img, cloth_img, output_dir="uploads/processed", pair_id="0001"):
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Human Parsing
        mask_path = os.path.join(output_dir, f"{pair_id}_label.png")
        self.parser.parse(person_img, mask_path)

        # Step 2: Pose Estimation
        pose_path = os.path.join(output_dir, f"{pair_id}_pose.json")
        self.pose_estimator.estimate(person_img, pose_path)

        # Step 3: Resize images
        resized_person = os.path.join(output_dir, f"{pair_id}_person.jpg")
        resized_cloth = os.path.join(output_dir, f"{pair_id}_cloth.jpg")

        self.img_preprocessor.resize_and_save(person_img, resized_person)
        self.img_preprocessor.resize_and_save(cloth_img, resized_cloth)

        return {
            "person": resized_person,
            "cloth": resized_cloth,
            "mask": mask_path,
            "pose": pose_path
        }


class AgnosticBuilder:
    """
    Build CP-VTON-style agnostic representation with 22 channels:
    - 18 pose heatmaps (OpenPose keypoints)
    - 1 body shape (from SCHP parsing)
    - 3 RGB head features (original image masked by head/hair/face)
    Output tensor shape: (1, 22, H, W) with values in [0,1]
    """

    def __init__(self, device: torch.device = torch.device("cpu"), image_size: Tuple[int, int] = (256, 192)):
        self.device = device
        self.image_size = image_size
        self.parser = HumanParser(device=str(device))
        try:
            self.pose = PoseEstimator(device=str(device))
        except Exception:
            self.pose = None

    def _gaussian_map(self, h: int, w: int, cx: float, cy: float, sigma: float = 4.0) -> np.ndarray:
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
        return g.astype(np.float32)

    def _build_pose_heatmaps(self, np_img: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        heatmaps = np.zeros((18, h, w), dtype=np.float32)
        if self.pose is None:
            return heatmaps
        # Estimate keypoints from array
        keypoints_all = self.pose.estimate_from_array((np_img * 255).astype(np.uint8)) if hasattr(self.pose, 'estimate_from_array') else []
        if isinstance(keypoints_all, list) or keypoints_all is None or len(np.shape(keypoints_all)) == 0:
            return heatmaps
        keypoints = keypoints_all[0]  # first person
        # If BODY_25, take first 18 points
        num_points = min(18, keypoints.shape[0])
        for i in range(num_points):
            x, y, c = keypoints[i]
            if c > 0.1:
                # Ensure within bounds
                cx = np.clip(x, 0, w - 1)
                cy = np.clip(y, 0, h - 1)
                heatmaps[i] = np.maximum(heatmaps[i], self._gaussian_map(h, w, cx, cy, sigma=4.0))
        # Normalize to [0,1]
        heatmaps = np.clip(heatmaps, 0.0, 1.0)
        return heatmaps

    def build(self, np_img: np.ndarray) -> torch.Tensor:
        # Ensure size and range
        h, w = self.image_size
        img = cv2.resize((np_img * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        img_float = img.astype(np.float32) / 255.0

        # Parsing
        parsing = self.parser.parse_array(img_float) if hasattr(self.parser, 'parse_array') else None
        if parsing is None:
            parsing = np.ones((h, w), dtype=np.uint8)

        # Body mask (any non-background)
        body_mask = (parsing != 0).astype(np.float32)
        body_mask = cv2.GaussianBlur(body_mask, (7, 7), 0)
        body_mask = np.clip(body_mask, 0.0, 1.0)

        # Head mask: indices for LIP {Hat=1, Hair=2, Face=13}
        head_mask = np.isin(parsing, [1, 2, 13]).astype(np.float32)
        head_mask = cv2.GaussianBlur(head_mask, (5, 5), 0)
        head_mask = np.clip(head_mask, 0.0, 1.0)

        # Pose heatmaps (18)
        pose_maps = self._build_pose_heatmaps(img)

        # Head RGB
        head_rgb = img_float * head_mask[..., None]

        # Stack channels: 18 + 1 + 3 = 22
        agnostic_np = np.concatenate([
            pose_maps,
            body_mask[None, ...],
            head_rgb.transpose(2, 0, 1)
        ], axis=0)

        agnostic_tensor = torch.from_numpy(agnostic_np).unsqueeze(0).to(self.device, dtype=torch.float32)
        return agnostic_tensor


class ClothMaskBuilder:
    """
    Build a binary cloth mask from product image using GrabCut.
    Returns float32 mask in [0,1] with shape (H,W).
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 192)):
        self.image_size = image_size

    def build(self, np_img: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        bgr = cv2.resize((np_img * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        if bgr.shape[2] == 3:
            bgr = bgr
        else:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_RGBA2BGR)

        mask = np.zeros((h, w), np.uint8)
        rect = (2, 2, w - 4, h - 4)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            mask_float = cv2.GaussianBlur(mask2.astype(np.float32), (5, 5), 0)
            mask_float = np.clip(mask_float, 0.0, 1.0)
            return mask_float.astype(np.float32)
        except Exception:
            return np.ones((h, w), dtype=np.float32)
