import speech_recognition as sr
import serial
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import asyncio
import edge_tts  # Librería para Text-to-Speech de alta calidad
import os

# --- CONFIGURACIÓN PRINCIPAL ---
PORT_ESP32 = 'COM4'      # ⚠️ CAMBIA ESTO: Usa el puerto de tu ESP32 (e.g., 'COM4' o '/dev/ttyUSB0')
BAUDRATE = 115200        # Asegúrate de que coincida con el baudrate de tu ESP32
VOICE_TTS = "es-ES-ElviraNeural" # Voz de mujer, alta calidad. Puedes probar 'es-ES-AlvaroNeural'
OUTPUT_AUDIO_FILE = "nombre_tts.mp3"
OUTPUT_TEXT_FILE = "nombre_reconocido.txt"

# --- CONFIGURACIÓN SERIAL ---
ser = None
try:
    # 1. Intentar conectar al puerto serial
    ser = serial.Serial(PORT_ESP32, BAUDRATE, timeout=1)
    time.sleep(2)  # Espera a que el puerto serial se estabilice
    print(f"Conexión con ESP32 en {PORT_ESP32} exitosa.")
except Exception as e:
    print(f"ERROR al conectar con ESP32 en {PORT_ESP32}: {e}")
    print("La comunicación serial NO estará disponible.")

# --- CONFIGURACIÓN RECONOCIMIENTO DE VOZ ---
recognizer = sr.Recognizer()
# Usaremos el micrófono con la configuración por defecto de 16kHz, compatible con Google
mic = sr.Microphone()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

def send_to_esp32(text, text_area):
    """Envía el texto completo a la ESP32 para que lo deletree."""
    if ser and ser.is_open:
        try:
            # Quitamos los espacios y convertimos a mayúsculas para la ESP32
            # El código de la ESP32 debe estar programado para leer byte a byte
            clean_text = text.replace(" ", "").upper()
            
            # Enviamos cada letra seguida de un marcador de fin de letra (e.g., '#') 
            # o simplemente una pausa, según lo requiera tu código ESP32.
            # Usaremos un marcador de fin de transmisión para que sepa cuándo parar.
            
            # Enviamos el texto limpio, carácter por carácter
            for char in clean_text:
                ser.write(char.encode('ascii'))
                # Puedes agregar un pequeño retardo si los servos de la ESP32 son lentos
                time.sleep(0.3) 
                text_area.insert(tk.END, f"-> Enviado: {char}\n")
                text_area.see(tk.END)

            # Enviamos un marcador de fin de palabra/nombre, si tu ESP32 lo necesita
            # ser.write(b'\n') 
            
            text_area.insert(tk.END, f"✅ Nombre ({clean_text}) enviado completamente a ESP32.\n")
            text_area.see(tk.END)

        except Exception as e:
            text_area.insert(tk.END, f"❌ Error al enviar datos a ESP32: {e}\n")
            text_area.see(tk.END)
    else:
        text_area.insert(tk.END, "⚠️ ESP32 no conectada. No se pudo enviar el texto.\n")
        text_area.see(tk.END)

def generate_tts_audio(text, text_area):
    """
    Genera el archivo de audio TTS ultra-realista usando edge-tts.
    Se usa asyncio para ejecutar la generación de audio de forma asíncrona.
    """
    try:
        text_area.insert(tk.END, f"⏳ Generando audio de alta calidad: '{text}'...\n")
        text_area.see(tk.END)

        # Crear un objeto de comunicación TTS
        communicate = edge_tts.Communicate(text, VOICE_TTS)
        
        # Ejecutar la comunicación en un loop de asyncio
        # Esto se ejecuta de forma síncrona dentro del thread principal
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(communicate.save(OUTPUT_AUDIO_FILE))
        
        text_area.insert(tk.END, f"✅ Audio guardado como: {OUTPUT_AUDIO_FILE}\n")
        text_area.see(tk.END)
    except Exception as e:
        text_area.insert(tk.END, f"❌ Error al generar audio TTS: {e}\n")
        text_area.see(tk.END)


def process_name(text_area):
    """
    Función principal que ejecuta todo el flujo de trabajo:
    1. Escucha el micrófono. 2. Reconoce el nombre. 3. Guarda el texto. 
    4. Genera el audio TTS. 5. Envía el texto a ESP32.
    """
    text_area.insert(tk.END, "🎤 Escuchando el nombre, por favor habla ahora...\n")
    text_area.see(tk.END)
    
    recognized_text = None
    
    try:
        with mic as source:
            # Ajustar por ruido ambiental
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Escucha por un nombre (permitiendo una frase un poco más larga que una sola letra)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
        
        # Usar Google Speech Recognition para mayor precisión en nombres/palabras
        recognized_text = recognizer.recognize_google(audio, language="es-ES")
        recognized_text = recognized_text.strip()
        
        text_area.insert(tk.END, f"✨ Nombre reconocido: {recognized_text}\n")
        text_area.see(tk.END)

        if recognized_text:
            # --- TAREA 1: Guardar el texto en un TXT ---
            with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(recognized_text)
            text_area.insert(tk.END, f"💾 Texto guardado en: {OUTPUT_TEXT_FILE}\n")

            # --- TAREA 2: Generar el audio TTS ultra-realista ---
            generate_tts_audio(recognized_text, text_area)

            # --- TAREA 3: Enviar el texto a la ESP32 ---
            send_to_esp32(recognized_text, text_area)
            
        else:
            text_area.insert(tk.END, "⚠️ No se detectó un nombre válido.\n")

    except sr.WaitTimeoutError:
        text_area.insert(tk.END, "❌ Tiempo de escucha agotado. Por favor, inténtalo de nuevo.\n")
    except sr.UnknownValueError:
        text_area.insert(tk.END, "❌ No se pudo entender el audio.\n")
    except sr.RequestError as e:
        text_area.insert(tk.END, f"❌ Error de la API de Google: {e}\n")
    except Exception as e:
        text_area.insert(tk.END, f"❌ Ocurrió un error inesperado: {e}\n")
        
    text_area.see(tk.END)

def start_processing_thread(text_area):
    """Inicia la función process_name en un hilo para no bloquear la GUI."""
    button.config(state=tk.DISABLED, text="Procesando...")
    # Creamos un hilo para la ejecución
    thread = threading.Thread(target=lambda: (process_name(text_area), button.config(state=tk.NORMAL, text="Iniciar Proceso")))
    thread.start()

# --- Configuración de la GUI (Tkinter) ---
root = tk.Tk()
root.title("Speech-to-Sign & Realistic TTS")
root.geometry("450x450")

# Área de texto para mostrar resultados
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
text_area.pack(padx=10, pady=10)
text_area.insert(tk.END, "¡Listo para iniciar! Presiona 'Iniciar grabación' y di un nombre.\n")
text_area.insert(tk.END, f"Puerto Serial: {PORT_ESP32} | Baudrate: {BAUDRATE}\n")

# Botón para activar el proceso
button = tk.Button(root, text="Iniciar Grabación", command=lambda: start_processing_thread(text_area))
button.pack(pady=5)

# Iniciar el bucle principal de tkinter
try:
    root.mainloop()
except KeyboardInterrupt:
    pass

# Cerrar puerto serial al salir (si sigue abierto)
if ser and ser.is_open:
    ser.close()
    print("Puerto serial cerrado al salir.")