import subprocess

def generate_lip_sync(input_audio="static/response.wav", input_face="static/face.jpg", output_video="static/output.mp4"):
    # Llama al script de inferencia de Wav2Lip con los parámetros necesarios
    command = [
        "python", "inference.py",
        "--checkpoint_path", "./Wav2Lip/checkpoints/wav2lip_gan.pth",
        "--face", input_face,
        "--audio", input_audio,
        "--outfile", output_video
    ]
    subprocess.run(command)

if __name__ == "__main__":
    generate_lip_sync()
