import torch
import torch.nn as nn
import wget
import os

LIPSYNC_FOLDER = "./LipSyncModel"
LIPSYNC_MODEL_WEIGHTS = "lipsync_expert.pth"
LIPSYNC_MODEL_WEIGHTS_URL = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth"
LIPSYNC_FILES_URLS = [
    (LIPSYNC_MODEL_WEIGHTS_URL, LIPSYNC_MODEL_WEIGHTS),
]

WAV2LIP_FOLDER = "./Wav2LipModel"
WAV2LIP_MODEL_WEIGHTS = "wav2lip_gan.pth"
WAV2LIP_MODEL_WEIGHTS_URL = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth"
WAV2LIP_FILES_URLS = [
    (WAV2LIP_MODEL_WEIGHTS_URL, WAV2LIP_MODEL_WEIGHTS),
]

def ensure_lipsync_files_exist():
    os.makedirs(LIPSYNC_FOLDER, exist_ok=True)
    for url, filename in LIPSYNC_FILES_URLS:
        filepath = os.path.join(LIPSYNC_FOLDER, filename)
        if not os.path.exists(filepath):
            try:
                wget.download(url, out=filepath)
            except Exception as e:
                print(f"Warning: Download for {filename} failed, likely due to link restrictions. You may need to download it manually.")

def ensure_wav2lip_files_exist():
    os.makedirs(WAV2LIP_FOLDER, exist_ok=True)
    for url, filename in WAV2LIP_FILES_URLS:
        filepath = os.path.join(WAV2LIP_FOLDER, filename)
        if not os.path.exists(filepath):
            try:
                wget.download(url, out=filepath)
            except Exception as e:
                print(f"Warning: Download for {filename} failed, likely due to link restrictions. You may need to download it manually.")


class LipSyncModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class Wav2LipModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits