
#!/usr/bin/env python3
"""
Preprocessing pipeline for CP-VTON-style virtual try-on (object-oriented).

Features:
- Strict 256x192 resizing for person & cloth
- Hooks for Human Parsing (SCHP) and Pose Estimation (OpenPose)
- Cloth masking (threshold-based) with a simple heuristic
- Mock mode to run without external models (produces usable placeholders)
- Clean directory structure, logging, and CLI

Usage (mock demo):
    python preprocessing.py --person /path/person.jpg --cloth /path/cloth.jpg --out ./preprocessed --mock

To plug real SCHP/OpenPose:
    - Implement SCHPHumanParser.run()
    - Implement OpenPoseEstimator.run()
    - Remove --mock and provide paths to those binaries or python modules
"""

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# -------------------------------
# Utils
# -------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -------------------------------
# Human Parsing (SCHP)
# -------------------------------
class HumanParser:
    def __init__(self, model_path="pretrained/SCHP_LIP.pth", device="cpu"):
        from networks import resnet101  # from SCHP repo
        self.device = device

        self.model = resnet101(pretrained=False, num_classes=20)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.406, 0.456, 0.485),
                                 (0.225, 0.224, 0.229))
        ])

    def parse(self, img_path, out_path):
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((192, 256), Image.BILINEAR)
        tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)

        Image.fromarray(parsing).save(out_path)
        return out_path

# -------------------------------
# Pose Estimation (OpenPose - PyOpenPose wrapper)
# -------------------------------
class PoseEstimator:
    def __init__(self, model_dir="pretrained/openpose_models/"):
        try:
            import pyopenpose as op
        except ImportError:
            raise ImportError("You must install OpenPose with Python API first.")

        params = dict()
        params["model_folder"] = model_dir
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate(self, img_path, out_path):
        import pyopenpose as op
        imageToProcess = cv2.imread(img_path)
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])

        keypoints = datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else []
        with open(out_path, "w") as f:
            json.dump({"people": [{"pose_keypoints_2d": keypoints}]}, f)

        return out_path

# -------------------------------
# Cloth Mask (GrabCut)
# -------------------------------
class ClothMasker:
    def generate_mask(self, img_path, out_path):
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
        cv2.imwrite(out_path, mask2)
        return out_path

# -------------------------------
# Main Preprocessor
# -------------------------------
class Preprocessor:
    def __init__(self, person_path, cloth_path, out_dir="./preprocessed",
                 parser_weights="pretrained/SCHP_LIP.pth",
                 pose_models="pretrained/openpose_models/"):
        self.person_path = person_path
        self.cloth_path = cloth_path
        self.out_dir = out_dir
        ensure_dir(out_dir)

        self.parser = HumanParser(model_path=parser_weights, device="cpu")
        self.pose = PoseEstimator(model_dir=pose_models)
        self.masker = ClothMasker()

    def run(self):
        # resize person + cloth
        person_img = Image.open(self.person_path).convert("RGB").resize((192, 256), Image.BILINEAR)
        cloth_img = Image.open(self.cloth_path).convert("RGB").resize((192, 256), Image.BILINEAR)

        person_out = os.path.join(self.out_dir, "person.jpg")
        cloth_out = os.path.join(self.out_dir, "cloth.jpg")
        person_img.save(person_out)
        cloth_img.save(cloth_out)

        # human parsing
        parsing_out = os.path.join(self.out_dir, "parsing.png")
        self.parser.parse(person_out, parsing_out)

        # pose estimation
        pose_out = os.path.join(self.out_dir, "person_pose.json")
        self.pose.estimate(person_out, pose_out)

        # cloth mask
        cloth_mask_out = os.path.join(self.out_dir, "cloth_mask.png")
        self.masker.generate_mask(cloth_out, cloth_mask_out)

        print(f"✅ Preprocessing complete. Results in {self.out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=str, required=True)
    parser.add_argument("--cloth", type=str, required=True)
    parser.add_argument("--out", type=str, default="./preprocessed")
    parser.add_argument("--parser_weights", type=str, default="pretrained/SCHP_LIP.pth")
    parser.add_argument("--pose_models", type=str, default="pretrained/openpose_models/")
    args = parser.parse_args()

    pre = Preprocessor(args.person, args.cloth, args.out, args.parser_weights, args.pose_models)
    pre.run()
