# pyright: reportMissingImports=false
# preprocessing/pose_estimator.py
import cv2
import json
import os
import numpy as np

class PoseEstimator:
    def __init__(self, device="cpu", model_folder="openpose/models/"):
        # Use pyopenpose (OpenPose Python API)
        try:
            import pyopenpose as op  # type: ignore
        except ImportError:
            raise ImportError("OpenPose Python API not found. Install pyopenpose first.")

        params = dict()
        params["model_folder"] = model_folder
        params["hand"] = False
        params["face"] = False

        self.op = op
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate(self, image_path, save_path):
        img = cv2.imread(image_path)
        datum = self.op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])

        keypoints = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else []

        with open(save_path, "w") as f:
            json.dump(keypoints, f)

        return save_path

    def estimate_keypoints(self, image_path):
        """Return pose keypoints as numpy array without saving to disk.
        Returns shape (num_people, num_keypoints, 3) or empty list if none.
        """
        img = cv2.imread(image_path)
        datum = self.op.Datum()
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop([datum])
        if datum.poseKeypoints is None:
            return []
        return np.array(datum.poseKeypoints)

    def estimate_from_array(self, np_img):
        """Estimate pose keypoints from an in-memory RGB image array [0,1] or [0,255]."""
        if np_img.dtype != np.uint8:
            img_uint8 = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = np_img
        # Convert RGB to BGR for OpenPose
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        datum = self.op.Datum()
        datum.cvInputData = img_bgr
        self.opWrapper.emplaceAndPop([datum])
        if datum.poseKeypoints is None:
            return []
        return np.array(datum.poseKeypoints)
