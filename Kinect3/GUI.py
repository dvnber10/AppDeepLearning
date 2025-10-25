import sys, time, math, os
import platform
import cv2
import numpy as np
import serial
import threading
import requests
import json
from PIL import Image
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget

# --- Librería de Kinect ---
try:
    import freenect
except ImportError:
    print("❌ ERROR: La librería 'freenect' no está instalada o configurada. El Kinect NO funcionará.")
    freenect = None

# --- Librerías de OCR ---
import pytesseract
# Importar la función de extracciónTexto (Aquí se define de forma simple)
def extract_text_from_image_1(qpixmap):
    """Guarda el QPixmap, usa pytesseract para OCR y devuelve el texto."""
    try:
        temp_path = "temp_ocr.png"
        qpixmap.save(temp_path, "PNG")
        # lang='spa' es para el español. Asegúrate de tener ese idioma instalado en Tesseract.
        text = pytesseract.image_to_string(Image.open(temp_path), lang='spa')
        os.remove(temp_path)
        return text.strip()
    except Exception as e:
        print(f"❌ Error en OCR: Verifique la instalación de Tesseract-OCR y su PATH/ruta. Error: {e}")
        return ""

# --- Librerías de Audio (Reemplazo de playsound) ---
try:
    from pydub import AudioSegment
    import simpleaudio as sa
except ImportError:
    print("ADVERTENCIA: pydub o simpleaudio no están instalados. El audio se guardará pero NO se reproducirá.")
    AudioSegment = None
    sa = None

# =========================
# CONFIGURACIÓN PRINCIPAL
# =========================
CANVAS_W, CANVAS_H = 1200, 700
SIDEBAR_W = 200
FPS = 15
CURSOR_RADIUS = 10
PEN_WIDTH = 20
ERASER_WIDTH = 26
BTN_H = 60
BTN_PADDING = 10
DWELL_MS = 1000

# --- Configuración Kinect (Valores predeterminados) ---
EMA_POS = 0.5
EMA_Z = 0.25
NEAR_MM = 300   # Distancia mínima de dibujo
FAR_MM = 1500   # Distancia máxima de dibujo
MIN_HAND_AREA = 150
MAX_HAND_AREA = 5000

COLOR_NAMES = ["Negro", "Azul", "Rojo", "Verde"]
COLORS = [Qt.black, Qt.blue, Qt.red, Qt.green]
TEXT_COLOR = QColor(255, 255, 255, 200)

# --- CONFIGURACIÓN TESSERACT OCR CROSS-PLATFORM ---
# ⚠️ PASO CRÍTICO: AJUSTA ESTAS RUTAS
if platform.system() == 'Windows':
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # RUTA EJEMPLO WINDOWS
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    except Exception:
        pass
else: # Linux/macOS
    os.environ['TESSDATA_PREFIX'] = '/usr/share/tessdata/'
    try:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' # RUTA EJEMPLO LINUX
    except Exception:
        pass


# --- CONFIGURACIÓN DE ARCHIVOS FIJOS ---
BASE_FILENAME = "CANVAS_OUTPUT"
IMAGE_FILE = f"{BASE_FILENAME}.png"
TEXT_FILE = f"{BASE_FILENAME}.txt"
AUDIO_FILE = f"{BASE_FILENAME}.mp3"

# --- CONFIGURACIÓN ELEVENLABS Y SERIAL ---
PORT_ESP32 = 'COM4'      # ⚠️ CAMBIA ESTO: Puerto de tu ESP32 (ej. 'COM3' o '/dev/ttyUSB0')
BAUDRATE = 115200
ELEVENLABS_API_KEY = "sk_2f185333dfcb8bb4f87016fbb1769644426c0b3ee7fb9580" # ⚠️ ¡COLOCA TU CLAVE REAL DE ELEVENLABS AQUÍ!
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Voz Rachel

# --- CONEXIÓN SERIAL ---
ser = None
try:
    ser = serial.Serial(PORT_ESP32, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"✅ Conexión serial con ESP32 en {PORT_ESP32} exitosa.")
except Exception as e:
    print(f"❌ ERROR al conectar con ESP32: {e}")
    print("La comunicación serial NO estará disponible.")

# =========================
# FUNCIONES DE UTILIDAD (TTS y SERIAL)
# =========================

def send_serial_command(text):
    """Envía el texto extraído al ESP32 a través del puerto serial."""
    global ser
    if ser:
        try:
            # Enviar el texto (fragmento) seguido de un salto de línea
            text_to_send = text if len(text) < 100 else text[:100] + "..."
            ser.write(f"{text_to_send}\n".encode('utf-8'))
            print(f"Comando serial enviado al ESP32. Texto (fragmento): {text_to_send}")
            return True
        except Exception as e:
            print(f"❌ Error al enviar comando serial: {e}")
            return False
    return False

def generate_and_play_tts_audio(text):
    """Genera audio con ElevenLabs API, lo guarda y lo reproduce con pydub/simpleaudio."""
    print(f"⏳ Generando audio con ElevenLabs para: '{text[:50]}...'")
    
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "TU_API_KEY":
        print("❌ ERROR: API Key de ElevenLabs no configurada.")
        return

    payload = {"text": text, "model_id": "eleven_multilingual_v2", 
               "voice_settings": {"stability": 0.3, "similarity_boost": 0.7}}
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json", "accept": "audio/mpeg"}
    
    try:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # 3. Guardar el archivo de audio con nombre fijo
            with open(AUDIO_FILE, "wb") as f:
                f.write(response.content)
            print(f"✅ Audio generado y guardado como {AUDIO_FILE}.")
            
            # 4. Reproducir audio con pydub/simpleaudio
            if AudioSegment and sa:
                try:
                    audio = AudioSegment.from_mp3(AUDIO_FILE)
                    wave_obj = sa.WaveObject(audio.raw_data, num_channels=audio.channels, 
                                            bytes_per_sample=audio.sample_width, 
                                            sample_rate=audio.frame_rate)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    print("🔊 Reproducción finalizada.")
                except Exception as e:
                    print(f"❌ Error de reproducción con simpleaudio. ¿FFmpeg instalado? Error: {e}")
            else:
                 print("🔊 Librerías de audio no disponibles, no se pudo reproducir.")

        else:
            error_msg = response.json().get('detail', response.text)
            print(f"❌ Error API ElevenLabs ({response.status_code}): {error_msg}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión con ElevenLabs: {e}")
    except Exception as e:
        print(f"❌ Error general en TTS: {e}")


# =========================
# CLASE PRINCIPAL (Kinect Drawing App)
# =========================
class DrawingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Drawing, OCR & Talk")
        self.setFixedSize(CANVAS_W + SIDEBAR_W, CANVAS_H)
        
        self.pixmap = QPixmap(CANVAS_W, CANVAS_H)
        self.pixmap.fill(Qt.white)

        self.mode = "draw"
        self.color_idx = 0
        self.cur_x, self.cur_y, self.cur_z = None, None, None
        self.is_fist = False
        self.drawing = False
        self.last_point = QPoint()
        self.hover_start = None
        
        self.buttons = self._init_buttons()
        self.button_labels = {1: "Cambiar Color", 2: "Borrar", 3: "Dibujar", 
                              4: "Limpiar Lienzo", 5: "Leer Texto (OCR)"}
        self.font_small = QFont("Arial", 12)

        self.kinect_timer = QTimer(self)
        self.kinect_timer.timeout.connect(self.update_kinect_frame)
        self.kinect_timer.start(1000 // FPS)

    def _init_buttons(self):
        buttons = {}
        for i in range(1, 6):
            y = BTN_PADDING + (i - 1) * (BTN_H + BTN_PADDING)
            buttons[i] = QRect(BTN_PADDING, y, SIDEBAR_W - 2 * BTN_PADDING, BTN_H)
        return buttons

    # --- LÓGICA DE PROCESAMIENTO KINECT (Mano y Dibujo) ---
    def _get_depth_frame(self):
        if freenect is None: return None
        depth = freenect.sync_get_depth()[0]
        valid_depth = np.where(depth > 0, depth, 2047)
        return valid_depth
    
    def _find_hand_contour(self, depth_frame):
        if depth_frame is None: return None, None, None, False
        depth_mm = 1000.0 / (depth_frame * -0.0030711016 + 3.3309495161)
        mask = np.logical_and(depth_mm > NEAR_MM, depth_mm < FAR_MM).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_LINEAR)
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None, None, False
            
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < MIN_HAND_AREA or area > MAX_HAND_AREA: return None, None, None, False
            
        M = cv2.moments(largest_contour)
        if M["m00"] == 0: return None, None, None, False
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calcular Z promedio
        contour_mask = np.zeros_like(depth_mm, dtype=np.uint8)
        scale_x, scale_y = depth_mm.shape[1] / 640, depth_mm.shape[0] / 480
        contour_original_size = (largest_contour * [scale_x, scale_y]).astype(int)
        cv2.drawContours(contour_mask, [contour_original_size], 0, 1, thickness=cv2.FILLED)
        z_avg_mm = depth_mm[contour_mask == 1].mean()
        
        # Detección de puño (solidez alta = puño cerrado)
        hull = cv2.convexHull(largest_contour)
        solidity = area / cv2.contourArea(hull)
        is_fist = solidity > 0.90 

        x_qt = int(cx * (CANVAS_W + SIDEBAR_W) / 640) 
        y_qt = int(cy * CANVAS_H / 480)
        
        return x_qt, y_qt, z_avg_mm, is_fist

    def update_kinect_frame(self):
        """Bucle principal de Kinect: procesa el frame y actualiza la UI."""
        
        x, y, z, is_fist = self._find_hand_contour(self._get_depth_frame())
        
        if x is not None:
            # Suavizado de posición (EMA)
            if self.cur_x is None:
                self.cur_x, self.cur_y, self.cur_z = x, y, z
            else:
                self.cur_x = int(self.cur_x * EMA_POS + x * (1.0 - EMA_POS))
                self.cur_y = int(self.cur_y * EMA_POS + y * (1.0 - EMA_POS))
                self.cur_z = self.cur_z * EMA_Z + z * (1.0 - EMA_Z)
            
            self.is_fist = is_fist
            
            # Lógica de Dibujo
            if self.cur_x >= SIDEBAR_W:
                canvas_x, canvas_y = self.cur_x - SIDEBAR_W, self.cur_y

                if self.is_fist:
                    if not self.drawing:
                        self.drawing = True
                        self.last_point = QPoint(canvas_x, canvas_y)
                    else:
                        self.draw_line(self.last_point, QPoint(canvas_x, canvas_y))
                        self.last_point = QPoint(canvas_x, canvas_y)
                else:
                    self.drawing = False
            else:
                 self.drawing = False
                 
            # Lógica de Botones (Hover)
            self.handle_button_hover(self.cur_x, self.cur_y)
            
        else:
            self.cur_x, self.cur_y, self.cur_z = None, None, None
            self.drawing = False
            self.hover_start = None

        self.update()

    def draw_line(self, start_point, end_point):
        painter = QPainter(self.pixmap)
        pen_color = COLORS[self.color_idx] if self.mode == 'draw' else Qt.white
        pen_width = PEN_WIDTH if self.mode == 'draw' else ERASER_WIDTH
            
        pen = QPen(pen_color, pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(start_point, end_point)

    def handle_button_hover(self, x, y):
        current_hovered_id = None
        for btn_id, rect in self.buttons.items():
            if rect.contains(QPoint(x, y)):
                current_hovered_id = btn_id
                break

        if current_hovered_id is not None:
            if self.hover_start is None:
                self.hover_start = time.time()
            elif (time.time() - self.hover_start) * 1000.0 >= DWELL_MS:
                self.handle_button_action(current_hovered_id)
                self.hover_start = None
        else:
            self.hover_start = None
            
    def handle_button_action(self, btn_id):
        if btn_id == 1: # Cambiar Color
            self.color_idx = (self.color_idx + 1) % len(COLORS)
            self.mode = 'draw'
            print(f"Cambiado a color: {COLOR_NAMES[self.color_idx]}")
        elif btn_id == 2: # Borrar
            self.mode = 'erase'
            print("Cambiado a modo: Borrar")
        elif btn_id == 3: # Dibujar
            self.mode = 'draw'
            print(f"Cambiado a modo: Dibujar con {COLOR_NAMES[self.color_idx]}")
        elif btn_id == 4: # Limpiar Lienzo
            self.pixmap.fill(Qt.white)
            print("Lienzo limpiado.")
        elif btn_id == 5: # Leer Texto (OCR) - Flujo de trabajo completo
            self.process_canvas_action()
            
    def process_canvas_action(self):
        """Flujo: Guardar Imagen -> OCR -> Guardar Texto -> ElevenLabs (Voz/Guardar MP3) -> Serial ESP32"""
        print("\n" + "=" * 50)
        
        # 1. Guardar la imagen con nombre fijo
        if self.pixmap.save(IMAGE_FILE, "PNG"):
            print(f"✅ 1. Imagen guardada como {IMAGE_FILE}.")
        
        # 2. Extraer Texto (OCR)
        text_to_read = extract_text_from_image_1(self.pixmap)
        
        if not text_to_read:
            final_text = "No se pudo reconocer ningún texto en el lienzo."
            print("⚠️ Proceso de voz/serial omitido debido a OCR fallido.")
        else:
            final_text = f"He leído el siguiente texto: {text_to_read}"
            print(f"✅ 2. Texto Extraído: {text_to_read[:50]}...")

            # 3. Guardar Texto (TXT) con nombre fijo
            with open(TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(text_to_read)
            print(f"✅ 3. Texto guardado en {TEXT_FILE}.")
            
            # 4. Generar Audio (ElevenLabs/MP3) en un hilo
            threading.Thread(target=generate_and_play_tts_audio, args=(final_text,), daemon=True).start()
            print("✅ 4. Audio ElevenLabs iniciado en segundo plano.")
            
            # 5. Enviar Texto a ESP32
            send_serial_command(text_to_read)
            print("✅ 5. Comando serial enviado.")
            
        print("=" * 50 + "\n")


    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Dibujar lienzo y barra lateral
        painter.drawPixmap(SIDEBAR_W, 0, self.pixmap)
        painter.fillRect(0, 0, SIDEBAR_W, CANVAS_H, QColor(40, 40, 40))
        
        # Dibujar botones y progreso del hover
        for btn_id, rect in self.buttons.items():
            painter.fillRect(rect, QColor(80, 80, 80))
            
            if self.hover_start is not None and rect.contains(QPoint(self.cur_x, self.cur_y)):
                frac = min(1.0, (time.time() - self.hover_start) * 1000.0 / DWELL_MS)
                prog_rect = QRect(rect.left(), rect.top(), int(rect.width() * frac), 4)
                painter.fillRect(prog_rect, QColor(60, 60, 60))
                
            painter.setPen(TEXT_COLOR)
            painter.setFont(self.font_small)
            painter.drawText(rect, Qt.AlignCenter, self.button_labels[btn_id])
        
        # HUD y Curson de Kinect
        painter.setFont(self.font_small)
        hud = f"Modo: {'Dibujar' if self.mode=='draw' else 'Borrar'} | Color: {COLOR_NAMES[self.color_idx]}"
        if self.cur_z is not None:
            hud += f" | Distancia: {self.cur_z:.0f}mm"
            hud += f" | Mano: {'Cerrada' if self.is_fist else 'Abierta'}"
        
        painter.setPen(Qt.black)
        painter.drawText(SIDEBAR_W + 12, 24, hud)
        
        if self.cur_x is not None and self.cur_y is not None:
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS, CURSOR_RADIUS)
            
            if self.is_fist and self.cur_x >= SIDEBAR_W:
                painter.setBrush(COLORS[self.color_idx] if self.mode == "draw" else Qt.white)
                painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS, CURSOR_RADIUS)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())