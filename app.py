import os
import zipfile
import requests
import shutil
import subprocess
import speech_recognition as sr
from flask import Flask, request, jsonify, send_file, Response, render_template, redirect
import time
from inference import *

ZIP_URL = "https://huggingface.co/Kfjjdjdjdhdhd/jjjhsjjdjdjd/resolve/main/fjhfdhjfhfhhfhfhfh.zip"
ZIP_PATH = "extra_files.zip"
EXTRACT_PATH = "extracted_files"
SOURCE_FOLDER_NAME = "sdsdsdsds"
DEST_DIR = "wav2lip"

os.makedirs(DEST_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    response = requests.get(ZIP_URL, stream=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
else:
    print("El archivo ZIP ya fue descargado.")

if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
else:
    print("El archivo ZIP ya fue extraído.")

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

app = Flask(__name__)
LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
VIDEO_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['IMAGE_DIR'] = './assets/uploaded_images/'
app.config['Filename'] = ''
os.makedirs(app.config['IMAGE_DIR'], exist_ok=True)

def audio_to_text():
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
    response = requests.post(LLAMA_API_URL, headers=HEADERS, json={"inputs": text})
    return response.json().get("generated_text", "No tengo respuesta.")

def text_to_speech(text, output_filename="outputs/response.wav"):
    response = requests.post(TTS_API_URL, headers=HEADERS, json={"inputs": text})
    with open(output_filename, "wb") as f:
        f.write(response.content)
    return output_filename

def sync_lips(video_path, audio_path, output_video="outputs/output.mp4"):
    command = f"python Wav2Lip/inference.py --checkpoint_path {DEST_DIR}/checkpoints/wav2lip_gan.pth --face {video_path} --audio {audio_path} --outfile {output_video}"
    subprocess.run(command, shell=True)
    return output_video

def remove_files_in_directory(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    video_file = request.files["video"]
    video_path = os.path.join(VIDEO_FOLDER, video_file.filename)
    video_file.save(video_path)
    return jsonify({"message": "Video uploaded successfully", "video_path": video_path})

@app.route("/process", methods=["GET"])
def process_video():
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
    output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")
    if not os.path.exists(output_path):
        return jsonify({"error": "No output video available"}), 400
    return send_file(output_path, as_attachment=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    remove_files_in_directory(app.config['IMAGE_DIR'])
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    app.config['Filename'] = file.filename
    file.save(os.path.join(app.config['IMAGE_DIR'], file.filename))
    return redirect("/")

flag = 0

@app.route('/requests', methods=['POST','GET'])
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
            return Response((os.path.join(app.config['IMAGE_DIR'], app.config['Filename']), flag), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(e)
    return ""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
