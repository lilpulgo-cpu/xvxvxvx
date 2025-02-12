import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import wget
import json
import os

STT_FOLDER = "./STTModel"
STT_MODEL_NAME = "wav2vec2"
STT_MODEL_WEIGHTS = "pytorch_model.bin"
STT_CONFIG = "config.json"
STT_VOCAB = "vocab.json"
STT_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin"
STT_CONFIG_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json"
STT_VOCAB_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json"
STT_FILES_URLS = [
    (STT_MODEL_WEIGHTS_URL, STT_MODEL_WEIGHTS),
    (STT_CONFIG_URL, STT_CONFIG),
    (STT_VOCAB_URL, STT_VOCAB),
]

def ensure_stt_files_exist():
    os.makedirs(STT_FOLDER, exist_ok=True)
    for url, filename in STT_FILES_URLS:
        filepath = os.path.join(STT_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class Wav2Vec2ForCTC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 39 * 40, num_classes) # Adjusted input size

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits