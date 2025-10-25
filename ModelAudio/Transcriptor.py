from vosk import Model, KaldiRecognizer
import wave
import json

def transcribe_with_vosk(audio_path):
    # Descargar el modelo de inglés (tamaño ~1.4 GB)
    model = Model("vosk-model-en-us-0.22-lgraph")  # Descarga desde: https://alphacephei.com/vosk/models
    
    with wave.open(audio_path, "rb") as wf:
        recognizer = KaldiRecognizer(model, wf.getframerate())
        
        text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text.append(result.get("text", ""))
        
        final_result = json.loads(recognizer.FinalResult())
        text.append(final_result.get("text", ""))
    
    return " ".join(text)

# Uso
audio_file = "Pruebas/Output/audio_extraido1.wav"
transcribed_text = transcribe_with_vosk(audio_file)
print("Texto extraído:", transcribed_text)