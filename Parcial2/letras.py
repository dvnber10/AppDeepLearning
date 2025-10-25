import speech_recognition as sr
import serial
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
from vosk import Model, KaldiRecognizer
import json

# Configuración serial (ajusta el puerto según tu sistema)
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)  # Espera a que el puerto serial se estabilice
except Exception as e:
    print(f"Error al conectar con ESP32: {e}")
    ser = None

# Inicializar reconocedor y micrófono
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)  # Vosk requiere 16kHz
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# Cargar modelo Vosk (ajusta la ruta al modelo descargado)
model_path = "vosk-model-small-es-0.42"  # Cambia a la ruta donde descomprimiste
vosk_model = Model(model_path)

# Definir gramática para reconocer solo letras A-Z
grammar = ["a", "be", "ce", "de", "e", "efe", "ge", "hache", "i", "jota", "ka",
           "ele", "eme", "ene", "o", "pe", "cu", "erre", "ese", "te", "u", "ve",
           "doble ve", "equis", "ye", "zeta"]
vosk_recognizer = KaldiRecognizer(vosk_model, 16000, json.dumps(grammar))

# Estado global para controlar la escucha
listening = False
listen_thread = None

def is_valid_letter(text):
    """Valida que el texto sea una sola letra A-Z."""
    return len(text) == 1 and text.isalpha() and text.isupper()

def letter_to_uppercase(word):
    """Convierte el nombre de la letra (e.g., 'be') a la letra mayúscula (e.g., 'B')."""
    letter_map = {
        "a": "A", "ve": "B", "ce": "C", "de": "D", "e": "E", "efe": "F", "ge": "G",
        "hache": "H", "i": "I", "jota": "J", "ka": "K", "ele": "L", "eme": "M",
        "ene": "N", "o": "O", "pe": "P", "cu": "Q", "erre": "R", "ese": "S",
        "te": "T", "u": "U", "u ve": "V", "doble ve": "W", "equis": "X", "ye": "Y",
        "zeta": "Z"
    }
    return letter_map.get(word.lower(), "")

def listen_and_recognize(text_area):
    """Función que escucha el micrófono y detecta una sola letra."""
    global listening
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while listening:
            try:
                text_area.insert(tk.END, "Escuchando letra...\n")
                text_area.see(tk.END)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=2)  # 2s para letras
                # Procesar audio con Vosk
                vosk_recognizer.AcceptWaveform(audio.get_raw_data(convert_rate=16000, convert_width=2))
                result = vosk_recognizer.Result()
                texto = json.loads(result).get("text", "").lower()
                # Convertir nombre de letra a letra mayúscula
                letra = letter_to_uppercase(texto)
                if is_valid_letter(letra):
                    text_area.insert(tk.END, f"Letra detectada: {letra}\n")
                    text_area.see(tk.END)
                    # Enviar al ESP32 si está conectado
                    if ser and ser.is_open:
                        ser.write(letra.encode())
                        ser.write(b'#')
                        text_area.insert(tk.END, f"Enviado a ESP32: {letra}#\n")
                        text_area.see(tk.END)
                else:
                    text_area.insert(tk.END, f"Texto inválido (no es una letra): {texto}\n")
                    text_area.see(tk.END)
                time.sleep(1)  # Pausa antes de siguiente escucha
            except sr.WaitTimeoutError:
                text_area.insert(tk.END, "No se detectó audio, reintentando...\n")
                text_area.see(tk.END)
            except Exception as e:
                text_area.insert(tk.END, f"Error: {e}\n")
                text_area.see(tk.END)

def toggle_listening(text_area):
    """Alterna la escucha del micrófono y actualiza el botón."""
    global listening, listen_thread
    if not listening:
        listening = True
        button.config(text="Detener Micrófono")
        text_area.insert(tk.END, "Micrófono activado. Di una letra.\n")
        text_area.see(tk.END)
        # Iniciar hilo de escucha
        listen_thread = threading.Thread(target=listen_and_recognize, args=(text_area,), daemon=True)
        listen_thread.start()
    else:
        listening = False
        button.config(text="Activar Micrófono")
        text_area.insert(tk.END, "Micrófono detenido.\n")
        text_area.see(tk.END)

# Configuración de la GUI
root = tk.Tk()
root.title("Speech to Sign Language (Vosk - Letras)")
root.geometry("400x400")

# Área de texto para mostrar resultados
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=15)
text_area.pack(padx=10, pady=10)

# Botón para activar/desactivar micrófono
button = tk.Button(root, text="Activar Micrófono", command=lambda: toggle_listening(text_area))
button.pack(pady=5)

# Iniciar el bucle principal de tkinter
root.mainloop()

# Cerrar puerto serial al salir
if ser and ser.is_open:
    ser.close()