#!/usr/bin/env python3
"""
Este script integra dos funcionalidades:
1. Uso de APIs de Hugging Face para generar respuestas y sintetizar voz.
2. Inferencia de lip-sync en videos/imágenes utilizando el modelo Wav2Lip.
"""

# ===============================
#       Importaciones
# ===============================
import os
import math
import time
import argparse
import requests

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import openvino as ov
import pyaudio
from PIL import Image, ImageTk

from dotenv import load_dotenv

import audio
# from face_detect import face_rect  # Descomenta si dispones de este módulo
from models import Wav2Lip
from batch_face import RetinaFace

# ===============================
#       Constantes Globales
# ===============================
LLAMA_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"

# Cargar variables de entorno
try:
    load_dotenv()
except Exception as e:
    print(f"Error al cargar .env: {e}")

API_KEY = os.getenv("HF_TOKEN")

# ===============================
#       Funciones API
# ===============================
def generate_response(text):
    """Genera una respuesta utilizando el modelo Llama."""
    response = requests.post(
        LLAMA_API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={"inputs": text}
    )
    return response.json().get("generated_text", "")

def text_to_speech(text, output_filename="response.wav"):
    """Convierte el texto a voz usando el modelo SpeechT5."""
    response = requests.post(
        TTS_API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={"inputs": text}
    )
    with open(output_filename, "wb") as f:
        f.write(response.content)

# ===============================
#       Argumentos para Wav2Lip
# ===============================
parser = argparse.ArgumentParser(
    description='Código de inferencia para lip-sync en videos/imágenes utilizando el modelo Wav2Lip'
)
parser.add_argument(
    '--checkpoint_path',
    type=str,
    default="./Wav2Lip/checkpoints/wav2lip_gan.pth",
    help='Ruta del checkpoint para cargar los pesos del modelo',
    required=False
)
parser.add_argument(
    '--face',
    type=str,
    default="Elon_Musk.jpg",
    help='Ruta del video/imagen que contiene las caras a utilizar',
    required=False
)
parser.add_argument(
    '--audio',
    type=str,
    help='Ruta del archivo de video/audio a utilizar como fuente de audio',
    required=False
)
parser.add_argument(
    '--outfile',
    type=str,
    default='results/result_voice.mp4',
    help='Ruta donde se guardará el video de salida'
)
parser.add_argument(
    '--static',
    type=bool,
    default=False,
    help='Si es True, se usa solo el primer frame para la inferencia'
)
parser.add_argument(
    '--fps',
    type=float,
    default=15.,
    help='FPS a usar cuando la entrada es una imagen estática',
    required=False
)
parser.add_argument(
    '--pads',
    nargs='+',
    type=int,
    default=[0, 10, 0, 0],
    help='Padding (arriba, abajo, izquierda, derecha) para incluir al menos la barbilla'
)
parser.add_argument(
    '--wav2lip_batch_size',
    type=int,
    default=8,
    help='Tamaño de batch para los modelos Wav2Lip'
)
parser.add_argument(
    '--resize_factor',
    default=1,
    type=int,
    help='Reduce la resolución en este factor. Mejores resultados a 480p o 720p'
)
parser.add_argument(
    '--out_height',
    default=480,
    type=int,
    help='Altura del video de salida. Mejores resultados a 480 o 720'
)
parser.add_argument(
    '--crop',
    nargs='+',
    type=int,
    default=[0, -1, 0, -1],
    help='Recorta el video a una región más pequeña (arriba, abajo, izquierda, derecha)'
)
parser.add_argument(
    '--box',
    nargs='+',
    type=int,
    default=[-1, -1, -1, -1],
    help='Bounding box constante para la cara. Usar solo si la detección falla'
)
parser.add_argument(
    '--rotate',
    default=False,
    action='store_true',
    help='Si el video está girado, lo rota 90° a la derecha'
)
parser.add_argument(
    '--nosmooth',
    default=False,
    action='store_true',
    help='Previene el suavizado de las detecciones faciales en ventanas temporales cortas'
)

# ===============================
#       Clase Wav2LipInference
# ===============================
class Wav2LipInference:
    
    def __init__(self, args) -> None:
        self.args = args
        # Configuración del audio
        self.CHUNK = 1024         # Número de frames por buffer
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1         # Audio monofónico
        self.RATE = 16000         # Frecuencia de muestreo
        self.RECORD_SECONDS = 0.5 # Duración de grabación por captura
        self.mel_step_size = 16   # Paso de frecuencia Mel
        self.audio_fs = 16000     # Frecuencia de muestreo del audio
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Usando {self.device} para inferencia.')
        
        # Cargar modelo y detector facial
        self.model = self.load_model()
        self.detector = self.load_batch_face_model()
        
        self.face_detect_cache_result = None
        self.img_tk = None

    def load_wav2lip_openvino_model(self):
        """
        Carga el modelo Wav2Lip usando OpenVINO (para CPU).

        Verifica que los archivos IR (.xml y .bin) existan y sean consistentes.
        """
        model_xml_path = os.path.join("./openvino_model/", "wav2lip_openvino_model.xml")
        model_bin_path = os.path.join("./openvino_model/", "wav2lip_openvino_model.bin")
        
        # Verificar existencia de los archivos
        if not os.path.exists(model_xml_path):
            raise FileNotFoundError(f"El archivo IR XML no existe: {model_xml_path}")
        if not os.path.exists(model_bin_path):
            raise FileNotFoundError(f"El archivo BIN no existe: {model_bin_path}")
        
        try:
            print("Cargando modelo OpenVINO para Wav2Lip...")
            core = ov.Core()
            devices = core.available_devices
            print(f"Dispositivo disponible: {devices[0]}")
            model = core.read_model(model=model_xml_path)
            compiled_model = core.compile_model(model=model, device_name=devices[0])
            return compiled_model
        except RuntimeError as e:
            # El error "Incorrect weights in bin file!" indica que el .bin no coincide con el .xml
            print("Error al cargar el modelo OpenVINO:", e)
            print("Verifica que el archivo .bin corresponda al archivo .xml y que no esté corrupto.")
            raise

    def load_model_weights(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_wav2lip_model(self, checkpoint_path):
        """Carga el modelo Wav2Lip y sus pesos desde el checkpoint."""
        model = Wav2Lip()
        print(f"Cargando checkpoint desde: {checkpoint_path}")
        checkpoint = self.load_model_weights(checkpoint_path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()

    def load_model(self):
        """Determina qué modelo cargar según el dispositivo."""
        if self.device == 'cpu':
            return self.load_wav2lip_openvino_model()
        else:
            return self.load_wav2lip_model(self.args.checkpoint_path)

    def load_batch_face_model(self):
        """Carga el detector facial."""
        if self.device == 'cpu':
            return RetinaFace(gpu_id=-1, model_path="checkpoints/mobilenet.pth", network="mobilenet")
        else:
            return RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
            
    def face_rect(self, images):
        """Genera las coordenadas de las caras detectadas en las imágenes."""
        face_batch_size = 64 * 8
        num_batches = math.ceil(len(images) / face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * face_batch_size: (i + 1) * face_batch_size]
            all_faces = self.detector(batch)
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret

    def record_audio_stream(self, stream):
        """Graba un fragmento de audio del stream."""
        stime = time.time()
        print("Grabando audio ...")
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            frames.append(stream.read(self.CHUNK))
        print("Grabación finalizada en {:.2f} segundos".format(time.time() - stime))
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data

    def get_mel_chunks(self, audio_data):
        """Convierte los datos de audio a trozos de espectrograma Mel."""
        stime = time.time()
        wav = audio_data
        mel = audio.melspectrogram(wav)
        print(f"Forma del mel: {mel.shape} en {time.time()-stime:.2f} segundos")
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('El espectrograma Mel contiene NaN. Prueba añadiendo un pequeño ruido.')
        stime = time.time()
        mel_chunks = []
        mel_idx_multiplier = 80. / self.args.fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1
        print(f"Total de chunks Mel: {len(mel_chunks)} en {time.time()-stime:.2f} segundos")
        return mel_chunks

    def get_smoothened_boxes(self, boxes, T):
        """Suaviza las cajas de detección facial en una ventana temporal."""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        """Realiza la detección facial en cada imagen."""
        results = []
        pady1, pady2, padx1, padx2 = self.args.pads
        s = time.time()
        for image, rect in zip(images, self.face_rect(images)):
            if rect is None:
                print("No se detectó la cara...")
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Cara no detectada. Asegúrate de que el video contenga una cara en cada frame.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])
        print('Tiempo de detección facial:', time.time() - s)
        boxes = np.array(results)
        if not self.args.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [
            [image[y1: y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]
        return results

    def datagen(self, frames, mels):
        """
        Genera batches de datos para la inferencia.
        Retorna:
            img_batch: lote de imágenes procesadas.
            mel_batch: lote de chunks de espectrograma Mel.
            frame_batch: frames originales.
            coords_batch: coordenadas de la región facial.
        """
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        if self.args.box[0] == -1:
            if not self.args.static:
                face_det_results = self.face_detect(frames)
            else:
                face_det_results = self.face_detect_cache_result
        else:
            print('Usando bounding box especificado en lugar de detección facial...')
            y1, y2, x1, x2 = self.args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        for i, m in enumerate(mels):
            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (self.args.img_size, self.args.img_size))
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch = np.asarray(img_batch)
                mel_batch = np.asarray(mel_batch)
                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size // 2:] = 0
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        if len(img_batch) > 0:
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch

# ===============================
#       Funciones Auxiliares
# ===============================
def update_frames(full_frames, stream, inference_pipeline):
    """Actualiza los frames con la predicción del modelo Wav2Lip."""
    stime = time.time()
    audio_data = inference_pipeline.record_audio_stream(stream)
    mel_chunks = inference_pipeline.get_mel_chunks(audio_data)
    print(f"Tiempo para procesar audio: {time.time() - stime:.2f} segundos")
    
    full_frames = full_frames[:len(mel_chunks)]
    batch_size = inference_pipeline.args.wav2lip_batch_size
    gen = inference_pipeline.datagen(full_frames.copy(), mel_chunks.copy())
    
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if inference_pipeline.device == 'cpu':
            img_batch = np.transpose(img_batch, (0, 3, 1, 2))
            mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))
            print(f"Batch shapes (CPU): {img_batch.shape}, {mel_batch.shape}")
            pred = inference_pipeline.model([mel_batch, img_batch])['output']
        else:
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(inference_pipeline.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(inference_pipeline.device)
            print(f"Batch shapes (GPU): {img_batch.shape}, {mel_batch.shape}")
            with torch.no_grad():
                pred = inference_pipeline.model(mel_batch, img_batch)
        print(f"Predicción shape: {pred.shape}")
        pred = pred.transpose(0, 2, 3, 1) * 255.
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            # Codificar la imagen a formato JPEG
            _, buffer = cv2.imencode('.jpg', f)
            buffer = np.array(buffer).tobytes()
            return (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n'
            )

def main(imagefilepath, flag):
    """
    Función principal para ejecutar la inferencia Wav2Lip.
    :param imagefilepath: Ruta del archivo de video/imagen.
    :param flag: Bandera para continuar (True) o detener (False) la inferencia.
    """
    args = parser.parse_args()
    args.img_size = 96
    args.face = imagefilepath
    inference_pipeline = Wav2LipInference(args)
    
    if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        args.static = True
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    elif os.path.isfile(args.face):
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Leyendo frames del video...')
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    else:
        raise ValueError('--face debe ser una ruta válida a un archivo de video/imagen')
    
    print(f"Número de frames disponibles para inferencia: {len(full_frames)}")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=inference_pipeline.FORMAT,
        channels=inference_pipeline.CHANNELS,
        rate=inference_pipeline.RATE,
        input=True,
        frames_per_buffer=inference_pipeline.CHUNK
    )
    
    inference_pipeline.face_detect_cache_result = inference_pipeline.face_detect([full_frames[0]])
    while True:
        if not flag:
            stream.stop_stream()
            stream.close()
            p.terminate()
            return b""
        print(f"Flag de inferencia: {flag}")
        yield update_frames(full_frames, stream, inference_pipeline)

# ===============================
#       Bloque Principal
# ===============================
if __name__ == "__main__":
    # Ejemplo de uso de las APIs de Hugging Face:
    input_text = "Hello, how can I help you?"
    response_text = generate_response(input_text)
    print("Respuesta AI:", response_text)
    text_to_speech(response_text)
    
    # Para ejecutar la inferencia Wav2Lip, llama a main con la ruta de la imagen/video y la bandera True.
    # Ejemplo:
    # for frame in main("Elon_Musk.jpg", True):
    #     # Aquí se puede procesar o mostrar el frame obtenido
    #     pass
