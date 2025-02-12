import torch
import torch.nn as nn
import wget
import json
import os

SUMMARIZATION_FOLDER = "./SummarizationModel"
SUMMARIZATION_MODEL_WEIGHTS = "pytorch_model.bin"
SUMMARIZATION_CONFIG = "config.json"
SUMMARIZATION_VOCAB = "vocab.json"
SUMMARIZATION_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/pytorch_model.bin"
SUMMARIZATION_CONFIG_URL = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json"
SUMMARIZATION_VOCAB_URL = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json"
SUMMARIZATION_FILES_URLS = [
    (SUMMARIZATION_MODEL_WEIGHTS_URL, SUMMARIZATION_MODEL_WEIGHTS),
    (SUMMARIZATION_CONFIG_URL, SUMMARIZATION_CONFIG),
    (SUMMARIZATION_VOCAB_URL, SUMMARIZATION_VOCAB),
]

def ensure_summarization_files_exist():
    os.makedirs(SUMMARIZATION_FOLDER, exist_ok=True)
    for url, filename in SUMMARIZATION_FILES_URLS:
        filepath = os.path.join(SUMMARIZATION_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class BartForConditionalGeneration(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits