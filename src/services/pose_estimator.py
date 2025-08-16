import numpy as np
from typing import Tuple, List

from ..utils.logger import Logger

try:
    # MMPose optional
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False

try:
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False


class PoseEstimator:
    """COCO-18 heatmap generator using MMPose (preferred) with fallback to MediaPipe.

    MMPose returns 17 COCO joints; we synthesize the 18th (neck) as the midpoint of the shoulders.
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 192)) -> None:
        self.logger = Logger(__name__)
        self.out_h, self.out_w = image_size

        self.use_mmpose = False
        if _HAS_MMPOSE:
            try:
                # lightweight default model shipped with mmpose (config packaged)
                cfg = 'mmpose::td-hm_hrnet-w32_8xb64-210e_coco-256x192'
                ckpt = None  # mmpose hub will resolve
                self.detector = init_model(cfg, checkpoint=ckpt, device='cpu')
                self.use_mmpose = True
                self.logger.info('MMPose initialized for COCO keypoints')
            except Exception as e:
                self.logger.warning(f'Could not init MMPose: {e}. Falling back to MediaPipe if available.')
                self.use_mmpose = False

        if not self.use_mmpose and _HAS_MEDIAPIPE:
            self.mp_pose = mp.solutions.pose.Pose(static_image_mode=True)  # type: ignore
        else:
            self.mp_pose = None

    def generate_heatmaps(self, rgb01: np.ndarray) -> np.ndarray:
        assert rgb01.shape[:2] == (self.out_h, self.out_w)
        if self.use_mmpose:
            return self._heatmaps_mmpose(rgb01)
        return self._heatmaps_mediapipe(rgb01)

    # ---- backends ----
    def _heatmaps_mmpose(self, rgb01: np.ndarray) -> np.ndarray:
        import numpy as np
        # MMPose expects BGR uint8
        img_bgr = rgb01[:, :, ::-1]
        result = inference_topdown(self.detector, img_bgr)[0]
        data_sample = merge_data_samples(result)
        kpts = data_sample.pred_instances.keypoints[0]  # (17,2)
        # synthesize neck as midpoint of shoulders (indices 5 and 6 in COCO 17)
        l_sh, r_sh = kpts[5], kpts[6]
        neck = (l_sh + r_sh) / 2.0
        kpts18 = np.vstack([kpts, neck[None, :]])  # (18,2)
        return self._gaussian_maps(kpts18)

    def _heatmaps_mediapipe(self, rgb01: np.ndarray) -> np.ndarray:
        h, w, _ = rgb01.shape
        coords = [(-1, -1)] * 18
        if self.mp_pose is not None:
            res = self.mp_pose.process((rgb01 * 255).astype(np.uint8))  # type: ignore
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark  # type: ignore
                def pt(i):
                    return int(lm[i].x * w), int(lm[i].y * h)
                r_sh, l_sh = pt(12), pt(11)
                neck = ((r_sh[0] + l_sh[0]) // 2, (r_sh[1] + l_sh[1]) // 2)
                mapping = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 7, 8]
                for i, idx in enumerate(mapping):
                    if i == 1:
                        coords[i] = neck
                    else:
                        coords[i] = pt(idx) if idx is not None else (-1, -1)
        return self._gaussian_maps(np.array(coords))

    def _gaussian_maps(self, coords: np.ndarray, sigma: float = 6.0) -> np.ndarray:
        h, w = self.out_h, self.out_w
        yy, xx = np.mgrid[0:h, 0:w]
        heatmaps = np.zeros((18, h, w), dtype=np.float32)
        for i, (cx, cy) in enumerate(coords):
            if cx is None or cy is None:
                continue
            if cx >= 0 and cy >= 0:
                g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                g_max = g.max()
                if g_max > 0:
                    heatmaps[i] = g / g_max
        return heatmaps


