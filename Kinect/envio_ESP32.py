import serial
import time
import unicodedata

def normalize_text(text):
    """
    Normaliza el texto para eliminar problemas de codificación, convirtiendo a ASCII
    y eliminando caracteres problemáticos si es necesario.
    """
    try:
        # Normalizar a NFC (forma compuesta) para manejar caracteres Unicode
        normalized_text = unicodedata.normalize('NFC', text)
        # Convertir a ASCII, ignorando caracteres no-ASCII
        return normalized_text.encode('ascii', errors='ignore').decode('ascii')
    except UnicodeEncodeError:
        return text.encode('ascii', errors='ignore').decode('ascii')

def send_to_esp32(file_path):
    port='/dev/ttyUSB0'
    baudrate=115200
    """
    Lee un archivo de texto y envía su contenido a la ESP32 por puerto serial,
    letra por letra, con un retardo de 4 segundos entre cada carácter,
    mostrando el progreso en la consola.
    """
    try:
        # Configurar la conexión serial
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"✅ Conectado a ESP32 en {port} con baudrate {baudrate}")

        # Leer el archivo de texto
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print("⚠️ El archivo de texto está vacío.")
            ser.close()
            return False

        # Normalizar el texto y eliminar espacios
        clean_text = normalize_text(text)
        clean_text = clean_text.replace(" ", "")
        
        print(f"📤 Enviando texto normalizado: {clean_text}")

        # Enviar cada carácter a la ESP32 con un retardo de 4 segundos
        for char in clean_text:
            ser.write(char.encode('ascii'))
            print(f"-> Enviado: {char}")
            time.sleep(5)  # Esperar 4 segundos después de cada carácter

        print(f"✅ Texto ({clean_text}) enviado completamente a ESP32.")
        ser.close()
        return True

    except FileNotFoundError:
        print(f"❌ Error: El archivo {file_path} no se encontró.")
        return False
    except serial.SerialException as e:
        print(f"❌ Error al conectar con la ESP32: {e}")
        return False
    except Exception as e:
        print(f"❌ Error al enviar datos a ESP32: {e}")
        if 'ser' in locals() and ser.is_open:
            ser.close()
        return False
    