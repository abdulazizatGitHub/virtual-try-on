import torch
import os

from . import log_config, path_config

device_str = os.getenv("DEVICE", "auto")

if device_str == 'auto':
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)

