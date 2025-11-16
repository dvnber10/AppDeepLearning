import sys, time, math, os
import cv2
import numpy as np
import freenect

from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

# =========================
# Configuración principal
# =========================
CANVAS_W, CANVAS_H = 1000, 700
FPS = 30
CURSOR_RADIUS = 14
PEN_WIDTH = 6
ERASER_WIDTH = 26

# Botones (barra inferior grande, fáciles de “clicar” con hover)
BTN_H = 90
BTN_PADDING = 12
BTN_W = 180

DWELL_MS = 1200  # tiempo para activar un botón por hover
EMA_POS = 0.25   # suavizado de posición mano
EMA_Z   = 0.15   # suavizado de distancia mano

# Detección de mano en profundidad
NEAR_MM = 300     # recorte mínimo (muy cerca del sensor)
FAR_MM  = 1500    # recorte máximo (ignoramos fondo lejano)
MIN_HAND_AREA = 150  # px mínimos para considerar un blob como mano

# Gesto “empujar para dibujar”:
PUSH_DELTA_MM = 120  # cuánta diferencia con respecto a z_base para “pen down”

# Colores disponibles
COLORS = [QColor(0,0,0), QColor(220, 50, 47)]  # negro, rojo (puedes añadir más)
COLOR_NAMES = ["Negro", "Rojo"]

# =========================
# Utilidades Kinect
# =========================
def get_depth_mm():
    """Obtiene mapa de profundidad en milímetros (si es posible)."""
    try:
        depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)
        if depth is None:
            raise RuntimeError("Kinect depth is None")
        return depth.astype(np.uint16)
    except Exception:
        # Fallback a 11 bits, convertir aproximado a mm
        depth11, _ = freenect.sync_get_depth()
        if depth11 is None:
            return None
        # Escala aproximada: valores 0..2047 -> ~0..(11 bits). No lineal real,
        # pero sirve para detectar “cercanía”. Convertimos a pseudo-mm:
        return (2047 - depth11).astype(np.uint16) * 2

def find_hand(depth_mm):
    """
    Encuentra la “mano” como el mayor blob dentro de un rango cercano.
    Retorna (cx, cy, z_mm) suavizados y una máscara binaria para debug opcional.
    """
    if depth_mm is None:
        return None, None, None, None

    # Recorta a rango útil
    mask = (depth_mm >= NEAR_MM) & (depth_mm <= FAR_MM)
    mask = mask.astype(np.uint8) * 255

    # Suavizado y morfología para limpiar ruido
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos y quedarse con el más grande
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, mask

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_HAND_AREA:
        return None, None, None, mask

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, None, None, mask

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    # z local promediando en una ventanita
    h, w = depth_mm.shape
    r = 8
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    z_patch = depth_mm[y0:y1, x0:x1]
    if z_patch.size == 0:
        return None, None, None, mask
    z_mm = int(np.median(z_patch[z_patch > 0])) if np.any(z_patch > 0) else None
    return cx, cy, z_mm, mask

# =========================
# Interfaz y lógica
# =========================
class KinectPaint(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Paint - Hover & Push Gestures")
        self.setFixedSize(CANVAS_W, CANVAS_H + BTN_H)

        # Canvas donde pintamos
        self.canvas = QPixmap(CANVAS_W, CANVAS_H)
        self.canvas.fill(Qt.white)

        # Etiqueta de render (solo 1 widget, dibujamos todo encima)
        self.view = QLabel(self)
        self.view.setGeometry(0, 0, CANVAS_W, CANVAS_H + BTN_H)

        # Estado
        self.mode = "draw"     # draw | erase
        self.color_idx = 0
        self.pen_down = False
        self.last_pt = None

        # Cursor (coordenadas en canvas)
        self.cur_x = CANVAS_W // 2
        self.cur_y = CANVAS_H // 2
        self.cur_z = None

        # Calibración de z base (distancia de referencia)
        self.z_base = None
        self.base_ready = False
        self.base_accum = []
        self.base_warmup_frames = FPS * 2  # ~2 s

        # Hover dwell
        self.hover_start = None
        self.hover_target = None

        # Pre-cálculo de rects de botones
        self.buttons = self._make_buttons()

        # Timer de actualización
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / FPS))

        # Fuente
        self.font_big = QFont("Sans", 18, QFont.Bold)
        self.font_small = QFont("Sans", 12)

    def _make_buttons(self):
        btns = []
        names = [
            ("Dibujar", "draw"),
            ("Borrar", "erase"),
            ("Guardar", "save"),
            (f"Color: {COLOR_NAMES[0]}", "color0"),
            (f"Color: {COLOR_NAMES[1]}", "color1"),
        ]
        x = BTN_PADDING
        y = CANVAS_H + (BTN_PADDING // 2)
        for label, action in names:
            rect = QRect(x, y, BTN_W, BTN_H - BTN_PADDING)
            btns.append((rect, label, action))
            x += BTN_W + BTN_PADDING
        return btns

    def kinect_to_canvas(self, x_depth, y_depth, shape):
        """Mapea coords del depth al canvas."""
        dh, dw = shape
        sx = CANVAS_W / float(dw)
        sy = CANVAS_H / float(dh)
        return int((dw - x_depth) * sx), int(y_depth * sy)

    def update_frame(self):
        # 1) Leer profundidad
        depth_mm = get_depth_mm()

        # 2) Encontrar mano
        hx, hy, hz, _mask = find_hand(depth_mm)

        if hx is not None and hy is not None and hz is not None:
            # Suavizar posición
            cx, cy = self.kinect_to_canvas(hx, hy, depth_mm.shape)
            self.cur_x = int(EMA_POS * cx + (1-EMA_POS) * self.cur_x)
            self.cur_y = int(EMA_POS * cy + (1-EMA_POS) * self.cur_y)

            # Suavizar z
            if self.cur_z is None:
                self.cur_z = hz
            else:
                self.cur_z = int(EMA_Z * hz + (1-EMA_Z) * self.cur_z)

            # Calibración base (primeros segundos)
            if not self.base_ready:
                self.base_accum.append(self.cur_z)
                if len(self.base_accum) >= self.base_warmup_frames:
                    self.z_base = int(np.median(self.base_accum))
                    self.base_ready = True

            # 3) Gestos
            self.apply_gestures()
        else:
            # Si perdemos la mano, levantamos el lápiz para evitar rayones
            self.pen_down = False
            self.last_pt = None

        # 4) Render de todo
        self.render_scene()

    def apply_gestures(self):
        # 3.1) Dwell para botones (si el cursor está sobre alguno)
        target_btn = None
        for rect, _, action in self.buttons:
            if rect.contains(self.cur_x, self.cur_y):
                target_btn = action
                break

        now = time.time()
        if target_btn != self.hover_target:
            # Cambió de objetivo
            self.hover_target = target_btn
            self.hover_start = now if target_btn else None
        else:
            # Misma diana
            if self.hover_target is not None and self.hover_start is not None:
                dwell_ms = (now - self.hover_start) * 1000
                if dwell_ms >= DWELL_MS:
                    self.activate_action(self.hover_target)
                    # Reinicio (para no activar en bucle)
                    self.hover_start = None
                    self.hover_target = None

        # 3.2) Push para “pen down” (solo si estamos en el canvas, no sobre botones)
        if self.cur_y < CANVAS_H:
            if self.base_ready and self.cur_z is not None:
                # Pen down si nos acercamos respecto a z_base
                self.pen_down = (self.z_base - self.cur_z) >= PUSH_DELTA_MM
            else:
                self.pen_down = False
        else:
            self.pen_down = False
            self.last_pt = None

        # 3.3) Dibujo continuo si pen_down
        if self.pen_down and self.cur_y < CANVAS_H:
            self.draw_at(QPoint(self.cur_x, self.cur_y))
        else:
            self.last_pt = None

    def activate_action(self, action):
        if action == "draw":
            self.mode = "draw"
        elif action == "erase":
            self.mode = "erase"
        elif action == "save":
            self.auto_save()
        elif action == "color0":
            self.color_idx = 0
        elif action == "color1":
            self.color_idx = 1
        # Actualizar etiquetas de color en botones
        self.buttons = self._make_buttons()

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
        self.canvas.save(out, "PNG")

    def render_scene(self):
        # Fondo final
        frame = QPixmap(self.width(), self.height())
        frame.fill(QColor(245, 245, 245))

        painter = QPainter(frame)

        # 1) Canvas de dibujo
        painter.drawPixmap(0, 0, self.canvas)

        # 2) Barra inferior con botones
        for rect, label, action in self.buttons:
            # Estado visual (activo)
            bg = QColor(230,230,230)
            if action == "draw" and self.mode == "draw":
                bg = QColor(210,235,210)
            if action == "erase" and self.mode == "erase":
                bg = QColor(235,210,210)
            if action.startswith("color") and int(action[-1]) == self.color_idx:
                bg = QColor(210,225,245)
            painter.fillRect(rect, bg)
            painter.setPen(Qt.black)
            painter.setFont(self.font_big)
            painter.drawText(rect, Qt.AlignCenter, label)

            # Indicador de dwell (progreso circular en el botón apuntado)
            if self.hover_target == action and self.hover_start is not None:
                frac = min(1.0, (time.time() - self.hover_start) * 1000.0 / DWELL_MS)
                # Dibujar una barra superior como progreso
                prog_rect = QRect(rect.left(), rect.top(), int(rect.width() * frac), 6)
                painter.fillRect(prog_rect, QColor(60,60,60))

        # 3) HUD (estado)
        painter.setFont(self.font_small)
        hud = f"Modo: {'Dibujar' if self.mode=='draw' else 'Borrar'} | Color: {COLOR_NAMES[self.color_idx]} | "
        if self.base_ready and self.cur_z is not None:
            hud += f"z_base={self.z_base}mm  z={self.cur_z}mm  Δ={(self.z_base - self.cur_z) if self.z_base else 0}mm"
        else:
            hud += "Calibrando distancia base... mantén la mano estable frente al sensor"
        painter.setPen(Qt.black)
        painter.drawText(12, 24, hud)

        # 4) Cursor (solo sobre el canvas)
        if self.cur_y < CANVAS_H:
            # círculo externo
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS, CURSOR_RADIUS)
            # relleno si pen down
            if self.pen_down:
                painter.setBrush(COLORS[self.color_idx] if self.mode == "draw" else Qt.white)
                painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS - 4, CURSOR_RADIUS - 4)

        painter.end()
        self.view.setPixmap(frame)


if __name__ == "__main__":
    # Acelera PyQt en algunos drivers
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    w = KinectPaint()
    w.show()
    sys.exit(app.exec_())
