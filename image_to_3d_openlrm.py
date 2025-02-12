import torch
import torch.nn as nn
import wget
import json
import os

IMAGE_TO_3D_FOLDER = "./ImageTo3DModel"
IMAGE_TO_3D_MODEL_WEIGHTS = "pytorch_model.bin"
IMAGE_TO_3D_CONFIG = "config.json"
IMAGE_TO_3D_MODEL_URL = "https://huggingface.co/zxhezexin/openlrm-obj-base-1.1/resolve/main/pytorch_model.bin"
IMAGE_TO_3D_CONFIG_URL = "https://huggingface.co/zxhezexin/openlrm-obj-base-1.1/resolve/main/config.json"
IMAGE_TO_3D_FILES_URLS = [
    (IMAGE_TO_3D_MODEL_URL, IMAGE_TO_3D_MODEL_WEIGHTS),
    (IMAGE_TO_3D_CONFIG_URL, IMAGE_TO_3D_CONFIG),
]

def ensure_image_to_3d_files_exist():
    os.makedirs(IMAGE_TO_3D_FOLDER, exist_ok=True)
    for url, filename in IMAGE_TO_3D_FILES_URLS:
        filepath = os.path.join(IMAGE_TO_3D_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class OpenLRM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits