import pyttsx3

import pyttsx3

def text_to_speech_pyttsx3(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)   # velocidad
    engine.setProperty("volume", 1.0)

    # Buscar una voz en español
    voices = engine.getProperty("voices")
    for v in voices:
        if "spanish" in v.id.lower() or "es" in v.id.lower():
            print("✅ Usando voz:", v.id)
            engine.setProperty("voice", v.id)
            break

    engine.say(text)
    engine.runAndWait()


# Uso
def text_to_speech_file_pyttsx3(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text_to_speech_pyttsx3(text)

