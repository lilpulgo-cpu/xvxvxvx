import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import wget
import json
import os

MUSICGEN_FOLDER = "./MusicGenModel"
MUSICGEN_MODEL_NAME = "melody"
MUSICGEN_MODEL_WEIGHTS = "pytorch_model.bin"
MUSICGEN_CONFIG = "config.json"
MUSICGEN_SAMPLE_RATE = 32000
MUSICGEN_DURATION = 8
MUSICGEN_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/musicgen-small/resolve/main/pytorch_model.bin"
MUSICGEN_CONFIG_URL = "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json"
MUSICGEN_FILES_URLS = [
    (MUSICGEN_MODEL_WEIGHTS_URL, MUSICGEN_MODEL_WEIGHTS),
    (MUSICGEN_CONFIG_URL, MUSICGEN_CONFIG),
]

def ensure_musicgen_files_exist():
    os.makedirs(MUSICGEN_FOLDER, exist_ok=True)
    for url, filename in MUSICGEN_FILES_URLS:
        filepath = os.path.join(MUSICGEN_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class MusicGenModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits