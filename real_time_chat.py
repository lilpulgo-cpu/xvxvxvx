 
import os
import requests
import speech_recognition as sr
from moviepy.editor import VideoFileClip, AudioFileClip
import subprocess

# Configuración de APIs
LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

def audio_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di algo...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="es-ES")
    except sr.UnknownValueError:
        return "No se entendió el audio."
    except sr.RequestError:
        return "Error en el servicio de reconocimiento."

def generate_response(text):
    response = requests.post(LLAMA_API_URL, headers=HEADERS, json={"inputs": text})
    return response.json().get("generated_text", "No tengo respuesta.")

def text_to_speech(text, output_filename="response.wav"):
    response = requests.post(TTS_API_URL, headers=HEADERS, json={"inputs": text})
    with open(output_filename, "wb") as f:
        f.write(response.content)
    return output_filename

def sync_lips(video_path, audio_path, output_video="output.mp4"):
    command = f"python Wav2Lip/inference.py --checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth --face {video_path} --audio {audio_path} --outfile {output_video}"
    subprocess.run(command, shell=True)
    return output_video

if __name__ == "__main__":
    user_input = audio_to_text()
    print(f"Usuario: {user_input}")

    ai_response = generate_response(user_input)
    print(f"AI: {ai_response}")

    audio_file = text_to_speech(ai_response)
    
    video_file = "input_video.mp4"  # Video base para sincronización
    output_video = sync_lips(video_file, audio_file)

    print(f"Video generado: {output_video}")
