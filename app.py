import os
import zipfile
import requests
import shutil
import subprocess
import speech_recognition as sr
from flask import Flask, request, jsonify, send_file, Response, render_template, redirect
import time
from inference import main

# ============================
# Parte 1: Descarga y extracción del ZIP
# ============================

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

# Extraer el ZIP en la carpeta 'extracted_files' si aún no se ha hecho
if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    print("Extrayendo el archivo ZIP en la carpeta 'extracted_files'...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
else:
    print("El archivo ZIP ya fue extraído en la carpeta 'extracted_files'.")

# Extraer el ZIP en el directorio actual (./) si aún no se ha hecho
ROOT_EXTRACT_MARKER = "extracted_in_root_marker"
if not os.path.exists(ROOT_EXTRACT_MARKER):
    print("Extrayendo el archivo ZIP en el directorio actual...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("./")
    with open(ROOT_EXTRACT_MARKER, "w") as marker:
        marker.write("Extraction in root completed.")
else:
    print("El archivo ZIP ya fue extraído en el directorio actual.")

# Mover el contenido de la carpeta 'sdsdsdsds' a 'wav2lip' sin sobrescribir archivos existentes
source_folder = os.path.join(EXTRACT_PATH, SOURCE_FOLDER_NAME)
if not os.path.exists(source_folder):
    print(f"La carpeta '{SOURCE_FOLDER_NAME}' no se encontró en el contenido extraído del ZIP.")
else:
    print(f"Moviendo el contenido de '{SOURCE_FOLDER_NAME}' a '{DEST_DIR}' sin sobrescribir archivos existentes...")
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            src_file = os.path.join(root, file)
            relative_path = os.path.relpath(src_file, source_folder)
            dest_file = os.path.join(DEST_DIR, relative_path)
            dest_file_dir = os.path.dirname(dest_file)
            os.makedirs(dest_file_dir, exist_ok=True)

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
# Parte 1.1: Copiar contenidos de 'openvino_model' y 'checkpoints'
#         a las carpetas destino 'openvino_model' y 'checkpoints'
#         de modo que ambas tengan la unión de los contenidos.
# ============================

DEST_OPENVINO = "openvino_model"
DEST_CHECKPOINTS = "checkpoints"
os.makedirs(DEST_OPENVINO, exist_ok=True)
os.makedirs(DEST_CHECKPOINTS, exist_ok=True)

for folder in ["openvino_model", "checkpoints"]:
    src_folder = os.path.join(EXTRACT_PATH, folder)
    if not os.path.exists(src_folder):
        print(f"La carpeta '{folder}' no se encontró en el contenido extraído del ZIP.")
    else:
        print(f"Copiando el contenido de '{folder}' a '{DEST_OPENVINO}' y '{DEST_CHECKPOINTS}'...")
        for root, dirs, files in os.walk(src_folder):
            for file in files:
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(src_file, src_folder)

                # Destino para openvino_model
                dest_file_openvino = os.path.join(DEST_OPENVINO, relative_path)
                os.makedirs(os.path.dirname(dest_file_openvino), exist_ok=True)
                final_dest_file_openvino = dest_file_openvino
                if os.path.exists(dest_file_openvino):
                    base, ext = os.path.splitext(dest_file_openvino)
                    i = 1
                    while os.path.exists(final_dest_file_openvino):
                        final_dest_file_openvino = f"{base}_{i}{ext}"
                        i += 1
                shutil.copy2(src_file, final_dest_file_openvino)
                print(f"Copiado: {src_file} -> {final_dest_file_openvino} (openvino)")

                # Destino para checkpoints
                dest_file_checkpoints = os.path.join(DEST_CHECKPOINTS, relative_path)
                os.makedirs(os.path.dirname(dest_file_checkpoints), exist_ok=True)
                final_dest_file_checkpoints = dest_file_checkpoints
                if os.path.exists(dest_file_checkpoints):
                    base, ext = os.path.splitext(dest_file_checkpoints)
                    i = 1
                    while os.path.exists(final_dest_file_checkpoints):
                        final_dest_file_checkpoints = f"{base}_{i}{ext}"
                        i += 1
                shutil.copy2(src_file, final_dest_file_checkpoints)
                print(f"Copiado: {src_file} -> {final_dest_file_checkpoints} (checkpoints)")
        print(f"Copiado el contenido de '{folder}' completado.")

# ============================
# Parte 2: Aplicación Flask - Funcionalidades de video, TTS y sincronización de labios
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
    """
    Convierte audio capturado desde el micrófono a texto usando Google Speech Recognition.
    """
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
    """
    Genera respuesta a partir de un modelo de lenguaje alojado en Hugging Face.
    """
    response = requests.post(LLAMA_API_URL, headers=HEADERS, json={"inputs": text})
    return response.json().get("generated_text", "No tengo respuesta.")

def text_to_speech(text, output_filename="outputs/response.wav"):
    """
    Convierte texto a voz utilizando una API de TTS alojada en Hugging Face.
    """
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
    """
    Sube un video al servidor.
    """
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
    """
    Descarga el video de salida.
    """
    output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")
    if not os.path.exists(output_path):
        return jsonify({"error": "No output video available"}), 400

    return send_file(output_path, as_attachment=True)

# ============================
# Parte 3: Aplicación Flask - Funcionalidades de imagen y video feed
# ============================

app.config['IMAGE_DIR'] = './assets/uploaded_images/'
app.config['Filename'] = ''

def remove_files_in_directory(directory):
    # Eliminar todos los archivos del directorio indicado
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Limpiar la carpeta de imágenes
    remove_files_in_directory(app.config['IMAGE_DIR'])

    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    app.config['Filename'] = file.filename
    file.save(os.path.join(app.config['IMAGE_DIR'], file.filename))
    return redirect("/")

global flag
flag = 0

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global flag
    try:
        if request.method == 'POST':
            if request.form.get('start') == 'Start':
                flag = 1
            elif request.form.get('stop') == 'Stop':
                flag = 0
            elif request.form.get('clear') == 'clear':
                flag = 0
            print(f"Flag value {flag}")
            time.sleep(2)
        elif request.method == 'GET':
            return render_template('index.html')
    except Exception as e:
        print(e)
    return render_template("index.html")

@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    global flag
    try:
        if app.config['Filename'] != '':
            return Response(main(os.path.join(app.config['IMAGE_DIR'], app.config['Filename']), flag),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(e)
    return ""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
