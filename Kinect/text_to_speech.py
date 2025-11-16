import os
import time
import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from datetime import datetime
import unicodedata
import re
import sounddevice as sd
import numpy as np


# === CONFIGURACIÓN ===
AUDIO_DIR = os.path.join(os.getcwd(), "audios")
os.makedirs(AUDIO_DIR, exist_ok=True)

# === CARGAR MODELO (MMS-TTS en español) ===
print("Cargando modelo MMS-TTS (español)...")
model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Modelo cargado en {device}")


# === NORMALIZACIÓN ===
def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    replacements = {
        r"\bduvan\b": "duv-ban",
        r"\bduván\b": "duv-ban",
        r"\bDuVan\b": "duv-ban",
        r"\bVuVán\b": "duv-ban",
        r"\bwifi\b": "uáifi",
        r"\bwhatsapp\b": "guásap",
        "@": "arroba",
        "#": "numeral",
        "%": "por ciento",
        r"°C": "grados Celsius",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text.strip()


# === GENERAR AUDIO ===
def generate_audio(text_data):
    text = normalize_text(text_data)
    print(f"Generando audio para: '{text_data[:60]}...'")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).waveform

    # Convertir a int16
    audio = output.squeeze().cpu().numpy()
    audio_int16 = (audio * 32767).astype("int16")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(AUDIO_DIR, f"voz_perfecta_{timestamp}.wav")

    sf.write(output_file, audio_int16, samplerate=16000)

    print(f"Audio generado: {output_file}")
    return output_file


# === REPRODUCIR AUDIO SIN PÉRDIDA (sounddevice) ===
def play_audio(file_path):
    try:
        print("Reproduciendo...")
        audio, sr = sf.read(file_path, dtype="int16")
        sd.play(audio, sr)
        sd.wait()
        print("Reproducción terminada.")
    except Exception as e:
        print(f"Error al reproducir: {e}")


# === PRINCIPAL ===
def text_to_speech_perfect(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        texto = f.read().strip()

    if not texto:
        print("Texto vacío.")
        return

    audio_file = generate_audio(texto)
    play_audio(audio_file)
