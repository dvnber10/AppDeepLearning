import speech_recognition as sr
import serial
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
from transformers import pipeline
import torch
import torchaudio
import numpy as np

# Configuración serial (ajusta el puerto: 'COM3' en Windows, '/dev/ttyUSB0' en Linux/Mac)
try:
    ser = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)  # Espera a que el puerto serial se estabilice
except Exception as e:
    print(f"Error al conectar con ESP32: {e}")
    ser = None

# Inicializar reconocedor y micrófono
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)  # Whisper requiere 16kHz
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# Cargar modelo Whisper desde Hugging Face
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1)

# Estado global para controlar la escucha
listening = False
listen_thread = None

def listen_and_recognize(text_area):
    """Función que escucha el micrófono y actualiza el área de texto usando Whisper."""
    global listening
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while listening:
            try:
                text_area.insert(tk.END, "Escuchando...\n")
                text_area.see(tk.END)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # 5s para nombres cortos
                # Convertir audio a formato compatible con Whisper
                audio_data = np.frombuffer(audio.get_raw_data(convert_rate=16000, convert_width=2), dtype=np.int16).astype(np.float32) / 32768.0
                # Procesar con Whisper
                result = whisper(audio_data, generate_kwargs={"language": "es"})
                texto = result["text"].upper().strip()
                if texto:
                    text_area.insert(tk.END, f"Texto detectado: {texto}\n")
                    text_area.see(tk.END)
                    # Enviar al ESP32 si está conectado
                    if ser and ser.is_open:
                        ser.write(texto.encode())
                        ser.write(b'#')
                        text_area.insert(tk.END, f"Enviado a ESP32: {texto}#\n")
                        text_area.see(tk.END)
                else:
                    text_area.insert(tk.END, "No se entendió el audio, reintentando...\n")
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
        text_area.insert(tk.END, "Micrófono activado.\n")
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
root.title("reconocimiento de voz a lenguaje de señas")
root.geometry("400x500")

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