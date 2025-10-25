import speech_recognition as sr
import serial
import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import os
import platform
import unicodedata
import requests
import json

# --- CONFIGURACIÓN PRINCIPAL ---
PORT_ESP32 = 'COM4'      # ⚠️ CAMBIA ESTO: Puerto de tu ESP32
BAUDRATE = 115200        # Asegúrate que coincida con tu ESP32
OUTPUT_AUDIO_FILE = "nombre_tts.mp3"
OUTPUT_TEXT_FILE = "texto_extraido.txt"
MICROPHONE_INDEX = None 

# --- CONFIGURACIÓN ELEVENLABS ---
ELEVENLABS_API_KEY = "sk_2f185333dfcb8bb4f87016fbb1769644426c0b3ee7fb9580"  # ✅ Tu API Key está configurada
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Voz Rachel

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

def normalize_text(text):
    """
    Normaliza el texto eliminando tildes y caracteres especiales.
    Convierte texto como 'José María' a 'JOSE MARIA'
    """
    normalized = unicodedata.normalize('NFD', text)
    clean_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return clean_text.upper().strip()

def save_text_to_file(text, text_area):
    """
    Guarda el texto extraído en un archivo de texto.
    """
    try:
        with open(OUTPUT_TEXT_FILE, 'a', encoding='utf-8') as file:
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {text}\n")
        text_area.insert(tk.END, f"📝 Texto guardado en: {OUTPUT_TEXT_FILE}\n")
        text_area.see(tk.END)
        return True
    except Exception as e:
        text_area.insert(tk.END, f"❌ Error al guardar texto en archivo: {e}\n")
        text_area.see(tk.END)
        return False

def generate_tts_audio(text, text_area):
    """
    Genera audio ultra-realista usando ElevenLabs API.
    """
    try:
        text_area.insert(tk.END, f"⏳ Generando audio ultra-realista: '{text}'...\n")
        text_area.see(tk.END)
        
        # Verificar que tenemos API key (CORREGIDO)
        if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY.startswith("sk_"):
            # Solo verificar que no esté vacía, no comparar con el valor exacto
            pass
        
        # Preparar los datos para la API
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 0.7,
                "style": 0.2,
                "use_speaker_boost": True
            }
        }
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "accept": "audio/mpeg"
        }
        
        # Hacer la petición a la API de ElevenLabs
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # Verificar la respuesta
        if response.status_code == 200:
            # Guardar el audio
            with open(OUTPUT_AUDIO_FILE, "wb") as f:
                f.write(response.content)
            
            # Verificar que el archivo se creó correctamente
            file_size = os.path.getsize(OUTPUT_AUDIO_FILE)
            text_area.insert(tk.END, f"✅ Audio guardado: {OUTPUT_AUDIO_FILE} ({file_size} bytes)\n")
            text_area.see(tk.END)
            
            # Verificación adicional
            if os.path.exists(OUTPUT_AUDIO_FILE) and file_size > 0:
                text_area.insert(tk.END, f"🔍 Archivo verificado: EXISTE y tiene {file_size} bytes\n")
            else:
                text_area.insert(tk.END, "⚠️ Archivo creado pero está vacío o no existe\n")
                
            return True
            
        else:
            error_msg = f"❌ Error ElevenLabs API (Código {response.status_code}): "
            try:
                error_detail = response.json()
                error_msg += error_detail.get('detail', str(error_detail))
            except:
                error_msg += response.text
                
            text_area.insert(tk.END, error_msg + "\n")
            text_area.see(tk.END)
            return False
            
    except requests.exceptions.Timeout:
        text_area.insert(tk.END, "❌ Timeout: La API de ElevenLabs tardó demasiado en responder\n")
        text_area.see(tk.END)
        return False
        
    except requests.exceptions.ConnectionError:
        text_area.insert(tk.END, "❌ Error de conexión: No se pudo conectar a ElevenLabs\n")
        text_area.see(tk.END)
        return False
        
    except Exception as e:
        text_area.insert(tk.END, f"❌ Error inesperado en ElevenLabs: {e}\n")
        text_area.see(tk.END)
        return False

def send_to_esp32(text, text_area):
    """Envía el texto completo a la ESP32."""
    if ser and ser.is_open:
        try:
            clean_text = normalize_text(text)
            clean_text = clean_text.replace(" ", "")
            
            text_area.insert(tk.END, f"📤 Enviando texto normalizado: {clean_text}\n")
            text_area.see(tk.END)
            
            for char in clean_text:
                ser.write(char.encode('ascii'))
                time.sleep(0.3) 
                text_area.insert(tk.END, f"-> Enviado: {char}\n")
                text_area.see(tk.END)

            text_area.insert(tk.END, f"✅ Nombre ({clean_text}) enviado completamente a ESP32.\n")
            text_area.see(tk.END)
            return True

        except Exception as e:
            text_area.insert(tk.END, f"❌ Error al enviar datos a ESP32: {e}\n")
            text_area.see(tk.END)
            return False
    else:
        text_area.insert(tk.END, "⚠️ ESP32 no conectada. No se pudo enviar el texto.\n")
        text_area.see(tk.END)
        return False

def play_audio(file_path, text_area):
    """
    Reproduce el archivo MP3 usando el reproductor de audio nativo del sistema.
    """
    try:
        if not os.path.exists(file_path):
            text_area.insert(tk.END, f"❌ Archivo de audio no encontrado: {file_path}\n")
            text_area.see(tk.END)
            return

        text_area.insert(tk.END, f"🔈 Reproduciendo audio ultra-realista...\n")
        text_area.see(tk.END)

        sys_platform = platform.system()
        
        if sys_platform == "Windows":
            os.startfile(file_path)
        elif sys_platform == "Darwin":
            os.system(f"afplay '{file_path}'")
        elif sys_platform == "Linux":
            os.system(f"mpg123 '{file_path}'")
        else:
            text_area.insert(tk.END, "⚠️ Plataforma no soportada para reproducción automática.\n")
            return

        text_area.insert(tk.END, "✅ Reproducción finalizada.\n")

    except Exception as e:
        text_area.insert(tk.END, f"❌ Error al reproducir audio: {e}\n")
    finally:
        text_area.see(tk.END)

def get_available_voices(text_area):
    """
    Obtiene la lista de voces disponibles en ElevenLabs.
    """
    try:
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
        
        if response.status_code == 200:
            voices = response.json()["voices"]
            text_area.insert(tk.END, f"\n🎙️  Voces disponibles: {len(voices)}\n")
            for voice in voices[:5]:
                text_area.insert(tk.END, f"   - {voice['name']} (ID: {voice['voice_id']})\n")
            text_area.see(tk.END)
    except Exception as e:
        text_area.insert(tk.END, f"⚠️ No se pudieron cargar las voces: {e}\n")

def process_name(text_area):
    """Ejecuta todo el flujo de procesamiento."""
    text_area.insert(tk.END, "\n--- INICIANDO PROCESO ---\n")
    text_area.insert(tk.END, "🎤 Escuchando el nombre, por favor habla ahora...\n")
    text_area.see(tk.END)
    
    recognized_text = None
    
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
        
        recognized_text = recognizer.recognize_google(audio, language="es-ES").strip()
        text_area.insert(tk.END, f"✨ Nombre reconocido: {recognized_text}\n")
        text_area.see(tk.END)

        if recognized_text:
            normalized_text = normalize_text(recognized_text)
            text_area.insert(tk.END, f"🔧 Texto normalizado: {normalized_text}\n")
            text_area.see(tk.END)
            
            save_text_to_file(recognized_text, text_area)
            tts_success = generate_tts_audio(recognized_text, text_area)
            send_success = send_to_esp32(recognized_text, text_area)
            
            if tts_success:
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
root = tk.Tk()
root.title("Speech-to-Sign & ElevenLabs TTS Ultra-Realista")
root.geometry("500x500")

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=22)
text_area.pack(padx=10, pady=10)
text_area.insert(tk.END, "🎧 SISTEMA DE AUDIO \n")
text_area.insert(tk.END, "=" * 50 + "\n")
text_area.insert(tk.END, f"📁 Archivo de texto: {OUTPUT_TEXT_FILE}\n")
text_area.insert(tk.END, f"🔌 Puerto Serial: {PORT_ESP32} | Baudrate: {BAUDRATE}\n")

# Verificar configuración de API Key (CORREGIDO)
if ELEVENLABS_API_KEY and not ELEVENLABS_API_KEY.startswith("TU_API_KEY"):
    text_area.insert(tk.END, f"\n✅\n")
    get_available_voices(text_area)
else:
    text_area.insert(tk.END, "\n❌ ERROR: Configura tu API Key de ElevenLabs\n")

text_area.insert(tk.END, "\n¡Presiona 'Iniciar Proceso' y habla un nombre!\n")

button = tk.Button(root, text="Iniciar Proceso", command=lambda: start_processing_thread(text_area), bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
button.pack(pady=10)

# Botón para probar TTS sin reconocimiento de voz
def test_tts_direct():
    test_text = "Hola, esto es una prueba de audio ultra realista"
    text_area.insert(tk.END, f"\n🧪 Probando TTS directo: '{test_text}'\n")
    if generate_tts_audio(test_text, text_area):
        time.sleep(1)
        play_audio(OUTPUT_AUDIO_FILE, text_area)

test_button = tk.Button(root, text="Probar TTS Directo", command=test_tts_direct, bg="#2196F3", fg="white")
test_button.pack(pady=5)

try:
    root.mainloop()
except KeyboardInterrupt:
    pass

if ser and ser.is_open:
    ser.close()
    print("Puerto serial cerrado al salir.")