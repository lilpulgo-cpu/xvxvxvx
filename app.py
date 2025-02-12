import os
import zipfile
import requests
import shutil
import subprocess
import speech_recognition as sr
from flask import Flask, request, jsonify, send_file

# ============================
# Parte 1: Descarga y extracción del ZIP
# ============================

# Configuración del ZIP y rutas
ZIP_URL = "https://huggingface.co/Kfjjdjdjdhdhd/jjjhsjjdjdjd/resolve/main/fjhfdhjfhfhhfhfhfh.zip"
ZIP_PATH = "extra_files.zip"
EXTRACT_PATH = "extracted_files"
SOURCE_FOLDER_NAME = "sdsdsdsds"  # Carpeta contenida en el ZIP
DEST_DIR = "wav2lip"             # Carpeta destino para mover el contenido

# Crear la carpeta de destino si no existe
os.makedirs(DEST_DIR, exist_ok=True)

# Descargar el archivo ZIP si no existe
if not os.path.exists(ZIP_PATH):
    print("Descargando archivo ZIP adicional...")
    response = requests.get(ZIP_URL, stream=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
else:
    print("El archivo ZIP ya fue descargado.")

# Extraer el ZIP si aún no se ha hecho
if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    print("Extrayendo el archivo ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
else:
    print("El archivo ZIP ya fue extraído.")

# Ruta de la carpeta "sdsdsdsds" dentro del contenido extraído
source_folder = os.path.join(EXTRACT_PATH, SOURCE_FOLDER_NAME)
if not os.path.exists(source_folder):
    print(f"La carpeta '{SOURCE_FOLDER_NAME}' no se encontró en el contenido extraído del ZIP.")
else:
    print(f"Moviendo el contenido de '{SOURCE_FOLDER_NAME}' a '{DEST_DIR}' sin sobrescribir archivos existentes...")

    # Recorrer de forma recursiva todas las carpetas, subcarpetas y archivos
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            src_file = os.path.join(root, file)
            # Calcular la ruta relativa respecto a la carpeta SOURCE_FOLDER_NAME
            relative_path = os.path.relpath(src_file, source_folder)
            dest_file = os.path.join(DEST_DIR, relative_path)
            dest_file_dir = os.path.dirname(dest_file)
            os.makedirs(dest_file_dir, exist_ok=True)

            # Si el archivo destino ya existe, se asigna un nombre alternativo para evitar sobrescribir
            final_dest_file = dest_file
            if os.path.exists(dest_file):
                base, ext = os.path.splitext(dest_file)
                i = 1
                while os.path.exists(final_dest_file):
                    final_dest_file = f"{base}_{i}{ext}"
                    i += 1

            shutil.move(src_file, final_dest_file)
            print(f"Movido: {src_file} -> {final_dest_file}")

    print("Movimiento completado.")

# ============================
# Parte 2: Aplicación Flask
# ============================

app = Flask(__name__)

# Configuración de APIs
LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

# Carpetas para los videos y salidas
VIDEO_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def audio_to_text():
    """Convierte audio capturado desde el micrófono a texto usando Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language="es-ES")
    except sr.UnknownValueError:
        return "No se entendió el audio."
    except sr.RequestError:
        return "Error en el servicio de reconocimiento."

def generate_response(text):
    """Genera respuesta a partir de un modelo de lenguaje alojado en Hugging Face."""
    response = requests.post(LLAMA_API_URL, headers=HEADERS, json={"inputs": text})
    return response.json().get("generated_text", "No tengo respuesta.")

def text_to_speech(text, output_filename="outputs/response.wav"):
    """Convierte texto a voz utilizando una API de TTS alojada en Hugging Face."""
    response = requests.post(TTS_API_URL, headers=HEADERS, json={"inputs": text})
    with open(output_filename, "wb") as f:
        f.write(response.content)
    return output_filename

def sync_lips(video_path, audio_path, output_video="outputs/output.mp4"):
    """
    Sincroniza labios utilizando el script de inferencia de Wav2Lip.
    Se asume que el script 'inference.py' se encuentra en la carpeta 'Wav2Lip'
    y que los checkpoints se encuentran en la carpeta 'wav2lip/checkpoints'.
    """
    command = (
        f"python Wav2Lip/inference.py "
        f"--checkpoint_path {DEST_DIR}/checkpoints/wav2lip_gan.pth "
        f"--face {video_path} "
        f"--audio {audio_path} "
        f"--outfile {output_video}"
    )
    subprocess.run(command, shell=True)
    return output_video

@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Sube un video al servidor."""
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    video_path = os.path.join(VIDEO_FOLDER, video_file.filename)
    video_file.save(video_path)

    return jsonify({"message": "Video uploaded successfully", "video_path": video_path})

@app.route("/process", methods=["GET"])
def process_video():
    """
    Procesa el primer video encontrado en la carpeta de uploads:
      1. Convierte audio a texto.
      2. Genera respuesta con el modelo Llama.
      3. Convierte la respuesta a audio.
      4. Sincroniza labios en el video.
    """
    video_files = os.listdir(VIDEO_FOLDER)
    if not video_files:
        return jsonify({"error": "No video file found"}), 400

    video_path = os.path.join(VIDEO_FOLDER, video_files[0])
    user_input = audio_to_text()
    ai_response = generate_response(user_input)
    audio_file = text_to_speech(ai_response)
    output_video = sync_lips(video_path, audio_file)

    return jsonify({"message": "Processing complete", "output_video": output_video})

@app.route("/download", methods=["GET"])
def download_video():
    """Descarga el video de salida."""
    output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")
    if not os.path.exists(output_path):
        return jsonify({"error": "No output video available"}), 400

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
