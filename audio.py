import librosa
import numpy as np
import sounddevice as sd
import requests
import wave
import os
from dotenv import load_dotenv
try:
    load_dotenv()
except Exception as e:
    print(f"Error: {e}")
API_KEY = os.getenv("HF_TOKEN")


WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(samplerate)
    wavefile.writeframes(audio.tobytes())
    wavefile.close()
    print("Recording complete.")

def transcribe_audio(filename="input.wav"):
    with open(filename, "rb") as f:
        response = requests.post(
            WHISPER_API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            data=f.read()
        )
    return response.json().get("text", "")

if __name__ == "__main__":
    record_audio()
    text = transcribe_audio()
    print("Transcribed Text:", text)