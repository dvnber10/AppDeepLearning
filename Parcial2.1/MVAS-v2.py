import speech_recognition as sr
import serial
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import asyncio
import edge_tts
import os  # ⬅️ Nueva importación para ejecutar comandos del sistema
import platform # Para detectar el sistema operativo
import unicodedata

# --- CONFIGURACIÓN PRINCIPAL ---
PORT_ESP32 = 'COM4'      # ⚠️ CAMBIA ESTO: Usa el puerto de tu ESP32 (e.g., 'COM4' o '/dev/ttyUSB0')
BAUDRATE = 115200        # Asegúrate de que coincida con el baudrate de tu ESP32
VOICE_TTS = "es-ES-ElviraNeural" 
OUTPUT_AUDIO_FILE = "nombre_tts.mp3"
# ⚠️ CAMBIA ESTO: Reemplaza 'X' con el número de índice de tu micrófono si lo sabes.
# Si no, déjalo vacío o en 'None' y el script intentará usar el micrófono por defecto.
MICROPHONE_INDEX = None 
OUTPUT_TEXT_FILE = "texto_extraido.txt"

# --- CONFIGURACIÓN SERIAL ---
ser = None
try:
    ser = serial.Serial(PORT_ESP32, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"Conexión con ESP32 en {PORT_ESP32} exitosa.")
except Exception as e:
    print(f"ERROR al conectar con ESP32 en {PORT_ESP32}: {e}")
    print("La comunicación serial NO estará disponible.")

# --- CONFIGURACIÓN RECONOCIMIENTO DE VOZ ---
recognizer = sr.Recognizer()
try:
    mic = sr.Microphone(device_index=MICROPHONE_INDEX) 
    print(f"Micrófono seleccionado (Índice {MICROPHONE_INDEX}).")
except ValueError:
    print("Usando micrófono por defecto (device_index no especificado o inválido).")
    mic = sr.Microphone()
    
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

def save_text_to_file(text, text_area):
    """
    Guarda el texto extraído en un archivo de texto.
    """
    try:
        with open(OUTPUT_TEXT_FILE, 'a', encoding='utf-8') as file:  # 'a' para append (agregar)
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {text}\n")
        text_area.insert(tk.END, f"📝 Texto guardado en: {OUTPUT_TEXT_FILE}\n")
        text_area.see(tk.END)
        return True
    except Exception as e:
        text_area.insert(tk.END, f"❌ Error al guardar texto en archivo: {e}\n")
        text_area.see(tk.END)
        return False

def normalize_text(text):
    """
    Normaliza el texto eliminando tildes y caracteres especiales.
    Convierte texto como 'José María' a 'JOSE MARIA'
    """
    # Normaliza el texto a la forma NFD (descomposición canónica)
    normalized = unicodedata.normalize('NFD', text)
    # Filtra solo los caracteres que no son marcas diacríticas (tildes)
    clean_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    # Convierte a mayúsculas y elimina espacios extra
    return clean_text.upper().strip()

def send_to_esp32(text, text_area):
    """Envía el texto completo a la ESP32."""
    if ser and ser.is_open:
        try:
            clean_text = text.replace(" ", "").upper()
            
            for char in clean_text:
                ser.write(char.encode('ascii'))
                time.sleep(0.3) 
                text_area.insert(tk.END, f"-> Enviado: {char}\n")
                text_area.see(tk.END)

            text_area.insert(tk.END, f"✅ Nombre ({clean_text}) enviado completamente a ESP32.\n")
            text_area.see(tk.END)
            return True # Éxito en el envío

        except Exception as e:
            text_area.insert(tk.END, f"❌ Error al enviar datos a ESP32: {e}\n")
            text_area.see(tk.END)
            return False
    else:
        text_area.insert(tk.END, "⚠️ ESP32 no conectada. No se pudo enviar el texto.\n")
        text_area.see(tk.END)
        return False

def generate_tts_audio(text, text_area):
    """Genera el archivo de audio TTS ultra-realista usando edge-tts."""
    try:
        text_area.insert(tk.END, f"⏳ Generando audio de alta calidad: '{text}'...\n")
        text_area.see(tk.END)
        communicate = edge_tts.Communicate(text, VOICE_TTS)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(communicate.save(OUTPUT_AUDIO_FILE))
        
        text_area.insert(tk.END, f"✅ Audio guardado como: {OUTPUT_AUDIO_FILE}\n")
        text_area.see(tk.END)
        return True
    except Exception as e:
        text_area.insert(tk.END, f"❌ Error al generar audio TTS: {e}\n")
        text_area.see(tk.END)
        return False

def play_audio(file_path, text_area):
    """
    Reproduce el archivo MP3 usando el reproductor de audio nativo del sistema.
    """
    try:
        text_area.insert(tk.END, f"🔈 Reproduciendo audio: {file_path}...\n")
        text_area.see(tk.END)

        sys_platform = platform.system()
        
        if sys_platform == "Windows":
            # Comando para Windows
            os.startfile(file_path)
        elif sys_platform == "Darwin":
            # Comando para macOS (requiere 'afplay')
            os.system(f"afplay {file_path}")
        elif sys_platform == "Linux":
            # Comando para Linux (usa 'xdg-open' o 'vlc', 'mpg123')
            # Asegúrate de tener un reproductor como 'mpg123' instalado: sudo apt install mpg123
            os.system(f"mpg123 {file_path}") # Opción recomendada para scripts
        else:
            text_area.insert(tk.END, "⚠️ Plataforma no soportada para reproducción automática.\n")
            return

        text_area.insert(tk.END, "✅ Reproducción finalizada.\n")

    except Exception as e:
        text_area.insert(tk.END, f"❌ Error al reproducir audio: {e}\n")
    finally:
        text_area.see(tk.END)


def process_name(text_area):
    """Ejecuta todo el flujo: Escucha -> Reconoce -> Genera TTS -> Envía a ESP32 -> Reproduce Audio."""
    text_area.insert(tk.END, "\n--- INICIANDO PROCESO ---\n")
    text_area.insert(tk.END, "🎤 Escuchando el nombre, por favor habla ahora...\n")
    text_area.see(tk.END)
    
    recognized_text = None
    
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Permite frases de hasta 4 segundos
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
        
        recognized_text = recognizer.recognize_google(audio, language="es-ES").strip()
        
        text_area.insert(tk.END, f"✨ Nombre reconocido: {recognized_text}\n")
        text_area.see(tk.END)

        if recognized_text:
            # 1. Generar el audio TTS
            tts_success = generate_tts_audio(recognized_text, text_area)

            # 2. Enviar el texto a la ESP32
            send_success = send_to_esp32(recognized_text, text_area)
            
            # 3. Reproducir el audio (Solo si la generación fue exitosa)
            if tts_success:
                # Damos una pequeña pausa para que la ESP32 empiece su tarea o para estabilizar
                time.sleep(1) 
                play_audio(OUTPUT_AUDIO_FILE, text_area)
            
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
        
    text_area.insert(tk.END, "--- PROCESO FINALIZADO ---\n")
    text_area.see(tk.END)

def start_processing_thread(text_area):
    """Inicia la función process_name en un hilo para no bloquear la GUI."""
    button.config(state=tk.DISABLED, text="Procesando...")
    thread = threading.Thread(target=lambda: (process_name(text_area), button.config(state=tk.NORMAL, text="Iniciar Proceso")))
    thread.start()

# --- Configuración de la GUI (Tkinter) ---
# ... (GUI setup remains the same)
root = tk.Tk()
root.title("Speech-to-Sign & Realistic TTS")
root.geometry("450x450")

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
text_area.pack(padx=10, pady=10)
text_area.insert(tk.END, "¡Listo! Presiona 'Iniciar Proceso', di un nombre, y el audio se reproducirá después del envío a la ESP32.\n")
text_area.insert(tk.END, f"Puerto Serial: {PORT_ESP32} | Baudrate: {BAUDRATE}\n")

button = tk.Button(root, text="Iniciar Proceso", command=lambda: start_processing_thread(text_area))
button.pack(pady=5)

try:
    root.mainloop()
except KeyboardInterrupt:
    pass

if ser and ser.is_open:
    ser.close()
    print("Puerto serial cerrado al salir.")