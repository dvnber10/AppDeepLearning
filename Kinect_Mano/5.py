import sys, time, math, os
import cv2
import numpy as np
import freenect
os.environ['TESSDATA_PREFIX'] = '/usr/share/tessdata/'  # Configurar TESSDATA_PREFIX para Arch Linux
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Ruta del ejecutable en Arch Linux
from PIL import Image
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

# =========================
# Configuración principal
# =========================
CANVAS_W, CANVAS_H = 1200, 700
SIDEBAR_W = 200
FPS = 30
CURSOR_RADIUS = 14
PEN_WIDTH = 20
ERASER_WIDTH = 26
BTN_H = 60
BTN_PADDING = 10
BTN_W = SIDEBAR_W - 2 * BTN_PADDING
DWELL_MS = 1000
EMA_POS = 0.25
EMA_Z = 0.25
NEAR_MM = 300
FAR_MM = 1500
MIN_HAND_AREA = 150
MAX_HAND_AREA = 5000
FIST_CONVEXITY_THRESHOLD = 0.85
COLORS = [
    QColor(0, 0, 0),
    QColor(220, 50, 47),
    QColor(38, 139, 210),
    QColor(133, 153, 0),
    QColor(211, 54, 130),
    QColor(42, 161, 152)
]
COLOR_NAMES = ["Negro", "Rojo", "Azul", "Verde", "Rosa", "Turquesa"]

# =========================
# Utilidades Kinect
# =========================
def get_depth_mm():
    for _ in range(3):
        try:
            depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
            if depth is not None:
                return depth.astype(np.uint16)
        except Exception:
            pass
        depth11, _ = freenect.sync_get_depth()
        if depth11 is not None:
            return (2047 - depth11).astype(np.uint16) * 2
    return None

def find_hand(depth_mm):
    if depth_mm is None:
        return None, None, None, None, False
    mask = np.zeros_like(depth_mm, dtype=np.uint8)
    mask[(depth_mm >= NEAR_MM) & (depth_mm <= FAR_MM)] = 255
    valid_pixels = depth_mm[(depth_mm >= NEAR_MM) & (depth_mm <= FAR_MM)]
    if valid_pixels.size == 0:
        return None, None, None, mask, False
    min_val = np.min(valid_pixels)
    hand_mask = np.zeros_like(depth_mm, dtype=np.uint8)
    hand_mask[(depth_mm >= min_val) & (depth_mm <= min_val + 50)] = 255
    hand_mask = cv2.medianBlur(hand_mask, 5)
    kernel = np.ones((5,5), np.uint8)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, mask, False
    valid_cnts = [cnt for cnt in cnts if MIN_HAND_AREA <= cv2.contourArea(cnt) <= MAX_HAND_AREA]
    if not valid_cnts:
        return None, None, None, mask, False
    best_cnt = max(valid_cnts, key=cv2.contourArea)
    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None, None, None, mask, False
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    h, w = depth_mm.shape
    r = 8
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    z_patch = depth_mm[y0:y1, x0:x1]
    if z_patch.size == 0:
        return None, None, None, mask, False
    z_val = int(np.median(z_patch[z_patch > 0])) if np.any(z_patch > 0) else None
    is_fist = False
    if len(best_cnt) >= 5:
        hull = cv2.convexHull(best_cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(best_cnt)
        if hull_area > 0:
            convexity = cnt_area / hull_area
            is_fist = convexity < FIST_CONVEXITY_THRESHOLD
    return cx, cy, z_val, mask, is_fist

# =========================
# Interfaz y lógica
# =========================
class KinectPaint(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Paint - Mano Abierta (Mover) / Cerrada (Dibujar)")
        self.setFixedSize(CANVAS_W + SIDEBAR_W, CANVAS_H)
        self.canvas = QPixmap(CANVAS_W, CANVAS_H)
        self.canvas.fill(Qt.white)
        self.view = QLabel(self)
        self.view.setGeometry(0, 0, CANVAS_W + SIDEBAR_W, CANVAS_H)
        self.mode = "draw"
        self.color_idx = 0
        self.pen_down = False
        self.last_pt = None
        self.cur_x = CANVAS_W // 2 + SIDEBAR_W
        self.cur_y = CANVAS_H // 2
        self.cur_z = None
        print("Initialized cur_x:", self.cur_x, "cur_y:", self.cur_y, "cur_z:", self.cur_z)
        self.hover_start = None
        self.hover_target = None
        self.last_button_press = 0
        self.button_cooldown = 1.0
        self.buttons = self._make_buttons()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / FPS))
        self.font_big = QFont("Sans", 12, QFont.Bold)
        self.font_small = QFont("Sans", 10)

    def _make_buttons(self):
        btns = []
        names = [
            ("Dibujar", "draw"),
            ("Borrar", "erase"),
            ("Guardar", "save"),
            ("Limpiar", "clear"),
            ("Color Negro", "color0"),
            ("Color Rojo", "color1"),
            ("Color Azul", "color2"),
            ("Color Verde", "color3"),
            ("Color Rosa", "color4"),
            ("Color Turquesa", "color5"),
        ]
        x = BTN_PADDING
        y = BTN_PADDING
        for label, action in names:
            rect = QRect(x, y, BTN_W, BTN_H)
            btns.append((rect, label, action))
            y += BTN_H + BTN_PADDING
        return btns

    def kinect_to_canvas(self, x_depth, y_depth, shape):
        dh, dw = shape
        total_width = CANVAS_W + SIDEBAR_W
        sx = total_width / float(dw)
        sy = CANVAS_H / float(dh)
        return int((dw - x_depth) * sx), int(y_depth * sy)

    def update_frame(self):
        depth_mm = get_depth_mm()
        hx, hy, hz, _mask, is_fist = find_hand(depth_mm)
        if hx is not None and hy is not None and hz is not None:
            cx, cy = self.kinect_to_canvas(hx, hy, depth_mm.shape)
            if not hasattr(self, 'cur_x'):
                self.cur_x = CANVAS_W // 2 + SIDEBAR_W
            if not hasattr(self, 'cur_y'):
                self.cur_y = CANVAS_H // 2
            self.cur_x = int(EMA_POS * cx + (1-EMA_POS) * self.cur_x)
            self.cur_y = int(EMA_POS * cy + (1-EMA_POS) * self.cur_y)
            if self.cur_z is None:
                self.cur_z = hz
            else:
                self.cur_z = int(EMA_Z * hz + (1-EMA_Z) * self.cur_z)
            self.apply_gestures(is_fist)
        else:
            self.pen_down = False
            self.last_pt = None
            self.hover_target = None
            self.hover_start = None
        self.render_scene(is_fist if hx is not None else None)

    def apply_gestures(self, is_fist):
        target_btn = None
        for rect, _, action in self.buttons:
            if rect.contains(self.cur_x, self.cur_y):
                target_btn = action
                break
        now = time.time()
        if now - self.last_button_press < self.button_cooldown:
            self.hover_target = None
            self.hover_start = None
            return
        if target_btn != self.hover_target:
            self.hover_target = target_btn
            self.hover_start = now if target_btn else None
        else:
            if self.hover_target is not None and self.hover_start is not None:
                dwell_ms = (now - self.hover_start) * 1000
                if dwell_ms >= DWELL_MS:
                    self.activate_action(self.hover_target)
                    self.hover_start = None
                    self.hover_target = None
                    self.last_button_press = now
        if self.cur_x >= SIDEBAR_W and self.hover_target is None:
            self.pen_down = is_fist
        else:
            self.pen_down = False
            self.last_pt = None
        if self.pen_down and self.cur_x >= SIDEBAR_W:
            canvas_x = self.cur_x - SIDEBAR_W
            self.draw_at(QPoint(canvas_x, self.cur_y))
        else:
            self.last_pt = None

    def activate_action(self, action):
        if action == "draw":
            self.mode = "draw"
        elif action == "erase":
            self.mode = "erase"
        elif action == "save":
            filename = self.auto_save()
            if filename:
                self.extract_text_from_image(filename)
        elif action.startswith("color"):
            self.color_idx = int(action[5:])
        elif action == "clear":
            self.clear_canvas()
        self.update_button_labels()

    def update_button_labels(self):
        self.buttons = self._make_buttons()

    def clear_canvas(self):
        self.canvas.fill(Qt.white)

    def draw_at(self, pt: QPoint):
        painter = QPainter(self.canvas)
        if self.mode == "erase":
            pen = QPen(Qt.white, ERASER_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        else:
            pen = QPen(COLORS[self.color_idx], PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        if self.last_pt is None:
            painter.drawPoint(pt)
        else:
            painter.drawLine(self.last_pt, pt)
        painter.end()
        self.last_pt = QPoint(pt)

    def auto_save(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = f"dibujo_{ts}.png"
        if self.canvas.save(out, "PNG"):
            print(f"Imagen guardada como: {out}")
            return out
        else:
            print(f"Error al guardar la imagen: {out}")
            return None

    def extract_text_from_image(self, image_path):
        try:
            print(f"Intentando abrir imagen: {image_path}")
            img = Image.open(image_path)
            print("Imagen cargada correctamente")
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)  # Escalar imagen
            img = img.convert('L')  # Convertir a escala de grises
            img = img.point(lambda p: p > 128 and 255)  # Binarización
            text = pytesseract.image_to_string(img, lang='eng')  # Especificar idioma inglés
            print(f"Texto extraído: {text}")
            txt_filename = os.path.splitext(image_path)[0] + ".txt"
            print(f"Guardando texto en: {txt_filename}")
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Texto extraído y guardado en: {txt_filename}")
            return text
        except Exception as e:
            print(f"Error al extraer texto de la imagen: {e}")
            return None

    def render_scene(self, is_fist):
        frame = QPixmap(self.width(), self.height())
        frame.fill(QColor(245, 245, 245))
        painter = QPainter(frame)
        sidebar_rect = QRect(0, 0, SIDEBAR_W, CANVAS_H)
        painter.fillRect(sidebar_rect, QColor(240, 240, 240))
        painter.drawPixmap(SIDEBAR_W, 0, self.canvas)
        for rect, label, action in self.buttons:
            bg = QColor(230, 230, 230)
            if action == "draw" and self.mode == "draw":
                bg = QColor(210, 235, 210)
            elif action == "erase" and self.mode == "erase":
                bg = QColor(235, 210, 210)
            elif action.startswith("color") and int(action[5:]) == self.color_idx:
                bg = COLORS[int(action[5:])].lighter(150)
            if self.hover_target == action:
                bg = bg.lighter(120)
            painter.fillRect(rect, bg)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawRect(rect)
            painter.setPen(Qt.black)
            painter.setFont(self.font_big)
            painter.drawText(rect, Qt.AlignCenter, label)
            if self.hover_target == action and self.hover_start is not None:
                frac = min(1.0, (time.time() - self.hover_start) * 1000.0 / DWELL_MS)
                prog_rect = QRect(rect.left(), rect.top(), int(rect.width() * frac), 4)
                painter.fillRect(prog_rect, QColor(60, 60, 60))
        painter.setFont(self.font_small)
        hud = f"Modo: {'Dibujar' if self.mode=='draw' else 'Borrar'} | Color: {COLOR_NAMES[self.color_idx]} | "
        if self.cur_z is not None:
            hud += f"Distancia: {self.cur_z}mm | "
            if is_fist is not None:
                hud += f"Mano: {'Cerrada (Dibujando)' if is_fist else 'Abierta (Moviendo)'}"
        painter.setPen(Qt.black)
        painter.drawText(SIDEBAR_W + 12, 24, hud)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS, CURSOR_RADIUS)
        if is_fist and self.cur_x >= SIDEBAR_W:
            painter.setBrush(COLORS[self.color_idx] if self.mode == "draw" else Qt.white)
            painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS - 4, CURSOR_RADIUS - 4)
        else:
            painter.drawLine(self.cur_x - 8, self.cur_y, self.cur_x + 8, self.cur_y)
            painter.drawLine(self.cur_x, self.cur_y - 8, self.cur_x, self.cur_y + 8)
        painter.end()
        self.view.setPixmap(frame)

if __name__ == "__main__":
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    w = KinectPaint()
    w.show()
    sys.exit(app.exec_())