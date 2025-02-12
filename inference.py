import requests
import os

LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
from dotenv import load_dotenv
try:
    load_dotenv()
except Exception as e:
    print(f"Error: {e}")
API_KEY = os.getenv("HF_TOKEN")

def generate_response(text):
    response = requests.post(
        LLAMA_API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"inputs": text}
    )
    return response.json().get("generated_text", "")

def text_to_speech(text, output_filename="response.wav"):
    response = requests.post(
        TTS_API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"inputs": text}
    )
    with open(output_filename, "wb") as f:
        f.write(response.content)

if __name__ == "__main__":
    input_text = "Hello, how can I help you?"
    response_text = generate_response(input_text)
    print("AI Response:", response_text)
    text_to_speech(response_text)