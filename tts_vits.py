import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import wget
import json
import os

TTS_FOLDER = "./TTSModel"
TTS_MODEL_NAME = "vits"
TTS_MODEL_CONFIG = "config.json"
TTS_MODEL_WEIGHTS = "pytorch_model.bin"
TTS_VOCAB = "vocab.json"
TTS_CONFIG_URL = "https://huggingface.co/kakao-enterprise/vits-vctk/resolve/main/config.json"
TTS_MODEL_WEIGHTS_URL = "https://huggingface.co/kakao-enterprise/vits-vctk/resolve/main/pytorch_model.bin"
TTS_VOCAB_URL = "https://huggingface.co/kakao-enterprise/vits-vctk/resolve/main/vocab.json"
TTS_FILES_URLS = [
    (TTS_CONFIG_URL, TTS_MODEL_CONFIG),
    (TTS_MODEL_WEIGHTS_URL, TTS_MODEL_WEIGHTS),
    (TTS_VOCAB_URL, TTS_VOCAB),
]

def ensure_tts_files_exist():
    os.makedirs(TTS_FOLDER, exist_ok=True)
    for url, filename in TTS_FILES_URLS:
        filepath = os.path.join(TTS_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class VITS(nn.Module):
    def __init__(self, spec_channels, segment_size, num_speakers, num_languages, num_symbols):
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.num_speakers = num_speakers
        self.num_languages = num_languages
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols, 192)
        self.decoder = Generator(spec_channels)

    def forward(self, text):
        x = self.embedding(text)
        audio = self.decoder(x)
        return audio

class Generator(nn.Module):
    def __init__(self, spec_channels):
        super().__init__()
        self.spec_channels = spec_channels
        self.initial_conv = nn.ConvTranspose2d(192, spec_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.final_conv = nn.Conv2d(spec_channels, 1, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, encoder_outputs):
        x = encoder_outputs.unsqueeze(2)
        x = self.initial_conv(x)
        x = self.final_conv(x)
        return x.squeeze(1)