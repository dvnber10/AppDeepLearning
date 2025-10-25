from bark import SAMPLE_RATE, generate_audio, preload_models
import soundfile as sf

# Cargar modelos
preload_models()

# Texto a convertir desde archivo txt
with open("output.txt", "r", encoding="utf-8") as f:
    texto = f.read().strip()

# Generar audio
audio_array = generate_audio(texto, history_prompt="es_speaker_6")  # voz española preentrenada

# Guardar a .wav
sf.write("cancion.wav", audio_array, SAMPLE_RATE)