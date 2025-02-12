import torch
import torch.nn as nn
import wget
import json
import os

TEXT_TO_VIDEO_FOLDER = "./TextToVideoModel"
TEXT_TO_VIDEO_MODEL_WEIGHTS = "pytorch_model.bin"
TEXT_TO_VIDEO_CONFIG = "config.json"
TEXT_TO_VIDEO_VOCAB = "vocab.json"
TEXT_TO_VIDEO_MODEL_WEIGHTS_URL = "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/pytorch_model.bin"
TEXT_TO_VIDEO_CONFIG_URL = "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/config.json"
TEXT_TO_VIDEO_VOCAB_URL = "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/vocab.json"
TEXT_TO_VIDEO_FILES_URLS = [
    (TEXT_TO_VIDEO_MODEL_WEIGHTS_URL, TEXT_TO_VIDEO_MODEL_WEIGHTS),
    (TEXT_TO_VIDEO_CONFIG_URL, TEXT_TO_VIDEO_CONFIG),
    (TEXT_TO_VIDEO_VOCAB_URL, TEXT_TO_VIDEO_VOCAB),
]

def ensure_text_to_video_files_exist():
    os.makedirs(TEXT_TO_VIDEO_FOLDER, exist_ok=True)
    for url, filename in TEXT_TO_VIDEO_FILES_URLS:
        filepath = os.path.join(TEXT_TO_VIDEO_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class Clip4ClipModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits