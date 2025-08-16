# preprocessing/image_preprocessor.py
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, size=(256, 192)):
        self.size = size
        self.transform_to_tensor = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])

    def resize_and_save(self, image_path, save_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.size, Image.BILINEAR)
        img.save(save_path)
        return save_path

    def preprocess_person_image(self, image_path: str) -> np.ndarray:
        """
        Load, resize to (256,192), convert to RGB float32 numpy in [0,1].
        Returns HWC numpy array.
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform_to_tensor(img)  # [0,1], CHW
        # Ensure target size
        if tensor.shape[1] != self.size[0] or tensor.shape[2] != self.size[1]:
            img_resized = img.resize(self.size, Image.BILINEAR)
            tensor = transforms.ToTensor()(img_resized)
        np_img = tensor.permute(1, 2, 0).numpy().astype(np.float32)  # HWC
        return np_img

    def preprocess_clothing_image(self, image_path: str) -> np.ndarray:
        """
        Same processing as person image. Returns HWC float32 numpy in [0,1].
        """
        return self.preprocess_person_image(image_path)

    def get_target_size(self):
        return self.size
