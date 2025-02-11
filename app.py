from flask import Flask, jsonify, request
import audio
import inference
import os

app = Flask(__name__)

@app.route("/talk", methods=["POST"])
def talk():
    audio.record_audio()
    transcribed_text = audio.transcribe_audio()
    response_text = inference.generate_response(transcribed_text)
    inference.text_to_speech(response_text, "static/response.wav")
    return jsonify({"response": response_text, "audio": "static/response.wav"})

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)