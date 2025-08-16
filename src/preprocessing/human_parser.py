#!/usr/bin/env python
# preprocessing/human_parser.py - OpenCV-only parser (no LIP/SCHP)

from PIL import Image
import numpy as np
import cv2


class HumanParser:
    """
    OpenCV-based lightweight human parser.
    Produces a simple label map: 0=background, 1=person, 13=head.
    Designed to provide masks needed by the CP-VTON agnostic builder
    (body mask and head mask) without SCHP/LIP.
    """

    def __init__(self, device="cpu"):
        # Haar cascade for face detection (falls back if unavailable)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

    def _person_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        rect = (max(1, w // 20), max(1, h // 20), w - max(2, w // 10), h - max(2, h // 10))
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        except Exception:
            mask[:] = 1  # fallback: everything foreground
        # Smooth and clean small holes
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def _head_mask(self, bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        head = np.zeros((h, w), np.uint8)
        try:
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            else:
                faces = []
        except Exception:
            faces = []

        if len(faces) > 0:
            # Take the most confident/largest face
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            # Expand upward to include hair/hat region
            y0 = max(0, int(y - 0.6 * fh))
            x0 = max(0, int(x - 0.2 * fw))
            x1 = min(w, int(x + 1.2 * fw))
            y1 = min(h, int(y + 0.9 * fh))
            head[y0:y1, x0:x1] = 1
        else:
            # Fallback: top region overlapped with person mask
            top = int(0.22 * h)
            head[:top, :] = 1

        head = (head & (person_mask > 0).astype(np.uint8)).astype(np.uint8)
        head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return head

    def parse(self, image_path, save_path):
        rgb = np.array(Image.open(image_path).convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        person = self._person_mask(bgr)
        head = self._head_mask(bgr, person)
        parsing = np.zeros(person.shape, dtype=np.uint8)
        parsing[person > 0] = 1
        parsing[head > 0] = 13
        cv2.imwrite(save_path, parsing)
        return save_path

    def parse_array(self, np_img: np.ndarray) -> np.ndarray:
        if np_img.dtype != np.uint8:
            img_uint8 = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = np_img
        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        person = self._person_mask(bgr)
        head = self._head_mask(bgr, person)
        parsing = np.zeros(person.shape, dtype=np.uint8)
        parsing[person > 0] = 1
        parsing[head > 0] = 13
        return parsing
