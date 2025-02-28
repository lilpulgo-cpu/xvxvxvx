import os
import zipfile
import requests
import shutil

# URLs y rutas
ZIP_URL = "https://huggingface.co/Kfjjdjdjdhdhd/jjjhsjjdjdjd/resolve/main/fjhfdhjfhfhhfhfhfh.zip"
ZIP_PATH = "extra_files.zip"
EXTRACT_PATH = "extracted_files"
SOURCE_FOLDER_NAME = "sdsdsdsds"  # Carpeta contenida en el ZIP
DEST_DIR = "wav2lip"             # Carpeta de destino

# Crear carpeta de destino si no existe
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

# Ruta de la carpeta "sdsdsdsds" dentro de los archivos extraídos
source_folder = os.path.join(EXTRACT_PATH, SOURCE_FOLDER_NAME)
if not os.path.exists(source_folder):
    print(f"La carpeta '{SOURCE_FOLDER_NAME}' no se encontró en el contenido extraído del ZIP.")
else:
    print(f"Moviendo el contenido de '{SOURCE_FOLDER_NAME}' a '{DEST_DIR}' sin sobrescribir archivos existentes...")

    # Recorrer recursivamente todas las carpetas, subcarpetas y archivos de la carpeta origen
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            src_file = os.path.join(root, file)
            # Calcular la ruta relativa respecto a la carpeta "sdsdsdsds"
            relative_path = os.path.relpath(src_file, source_folder)
            dest_file = os.path.join(DEST_DIR, relative_path)
            dest_file_dir = os.path.dirname(dest_file)
            os.makedirs(dest_file_dir, exist_ok=True)

            # Si el archivo de destino ya existe, generar un nombre alternativo para evitar sobrescribirlo
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

print("¡Operación completada!")
