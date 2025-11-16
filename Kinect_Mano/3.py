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
CANVAS_W, CANVAS_H = 1200, 700
SIDEBAR_W = 200  # Ancho del sidebar
FPS = 25
CURSOR_RADIUS = 14
PEN_WIDTH = 6
ERASER_WIDTH = 26

# Botones en sidebar izquierdo
BTN_H = 60
BTN_PADDING = 10
BTN_W = SIDEBAR_W - 2 * BTN_PADDING

DWELL_MS = 1000  # tiempo para activar un botón por hover
EMA_POS = 0.25   # suavizado de posición mano
EMA_Z   = 0.25   # suavizado de distancia mano

# Detección de mano en profundidad
NEAR_MM = 300     # recorte mínimo (muy cerca del sensor)
FAR_MM  = 1500    # recorte máximo (ignoramos fondo lejano)
MIN_HAND_AREA = 150  # px mínimos para considerar un blob como mano
MAX_HAND_AREA = 5000  # px máximos para evitar detectar todo el cuerpo

# Gesto "empujar para dibujar" reemplazado con detección de puño
FIST_CONVEXITY_THRESHOLD = 0.85  # Umbral para detectar puño (convexidad baja = puño)

# Colores disponibles
COLORS = [
    QColor(0, 0, 0),        # Negro
    QColor(220, 50, 47),    # Rojo
    QColor(38, 139, 210),   # Azul
    QColor(133, 153, 0),    # Verde
    QColor(211, 54, 130),   # Rosa
    QColor(42, 161, 152)    # Turquesa
]
COLOR_NAMES = ["Negro", "Rojo", "Azul", "Verde", "Rosa", "Turquesa"]

# =========================
# Utilidades Kinect
# =========================
def get_depth_mm():
    for _ in range(3):  # Intenta 3 veces
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
    """
    Encuentra la "mano" como el punto más cercano dentro del rango.
    Retorna (cx, cy, z_mm) suavizados, una máscara binaria, y si es puño.
    """
    if depth_mm is None:
        return None, None, None, None, False

    # Crear máscara con solo los puntos dentro del rango
    mask = np.zeros_like(depth_mm, dtype=np.uint8)
    mask[(depth_mm >= NEAR_MM) & (depth_mm <= FAR_MM)] = 255

    # Encontrar el punto más cercano (valor mínimo de profundidad)
    valid_pixels = depth_mm[(depth_mm >= NEAR_MM) & (depth_mm <= FAR_MM)]
    if valid_pixels.size == 0:
        return None, None, None, mask, False
        
    min_val = np.min(valid_pixels)
    
    # Crear máscara para el punto más cercano y sus alrededores
    hand_mask = np.zeros_like(depth_mm, dtype=np.uint8)
    hand_mask[(depth_mm >= min_val) & (depth_mm <= min_val + 50)] = 255
    
    # Aplicar operaciones morfológicas para limpiar la máscara
    hand_mask = cv2.medianBlur(hand_mask, 5)
    kernel = np.ones((5,5), np.uint8)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos en la máscara de la mano
    cnts, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, mask, False
    
    # Filtrar contornos por área
    valid_cnts = [cnt for cnt in cnts if MIN_HAND_AREA <= cv2.contourArea(cnt) <= MAX_HAND_AREA]
    if not valid_cnts:
        return None, None, None, mask, False
    
    # Encontrar el contorno más grande que cumpla con los criterios
    best_cnt = max(valid_cnts, key=cv2.contourArea)
    
    # Calcular centroide
    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None, None, None, mask, False
        
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    
    # Obtener profundidad en el centroide
    h, w = depth_mm.shape
    r = 8
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    z_patch = depth_mm[y0:y1, x0:x1]
    
    if z_patch.size == 0:
        return None, None, None, mask, False
        
    z_val = int(np.median(z_patch[z_patch > 0])) if np.any(z_patch > 0) else None
    
    # Determinar si es puño o mano abierta usando convexidad
    is_fist = False
    if len(best_cnt) >= 5:  # Necesitamos al menos 5 puntos para convexHull
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

        # Canvas donde pintamos (área derecha)
        self.canvas = QPixmap(CANVAS_W, CANVAS_H)
        self.canvas.fill(Qt.white)

        # Etiqueta de render (solo 1 widget, dibujamos todo encima)
        self.view = QLabel(self)
        self.view.setGeometry(0, 0, CANVAS_W + SIDEBAR_W, CANVAS_H)

        # Estado
        self.mode = "draw"     # draw | erase
        self.color_idx = 0
        self.pen_down = False
        self.last_pt = None

        # Cursor (coordenadas en canvas)
        self.cur_x = CANVAS_W // 2 + SIDEBAR_W
        self.cur_y = CANVAS_H // 2
        self.cur_z = None

        # Hover dwell para botones
        self.hover_start = None
        self.hover_target = None
        self.last_button_press = 0
        self.button_cooldown = 1.0  # segundos entre pulsaciones

        # Pre-cálculo de rects de botones
        self.buttons = self._make_buttons()

        # Timer de actualización
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / FPS))

        # Fuente
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
        """Mapea coords del depth al canvas completo (incluyendo sidebar)."""
        dh, dw = shape
        total_width = CANVAS_W + SIDEBAR_W
        sx = total_width / float(dw)
        sy = CANVAS_H / float(dh)
        return int((dw - x_depth) * sx), int(y_depth * sy)

    def update_frame(self):
        # 1) Leer profundidad
        depth_mm = get_depth_mm()

        # 2) Encontrar mano
        hx, hy, hz, _mask, is_fist = find_hand(depth_mm)

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

            # 3) Gestos - mano cerrada para dibujar, abierta para moverse
            self.apply_gestures(is_fist)
        else:
            # Si perdemos la mano, levantamos el lápiz para evitar rayones
            self.pen_down = False
            self.last_pt = None
            self.hover_target = None
            self.hover_start = None

        # 4) Render de todo
        self.render_scene(is_fist if hx is not None else None)

    def apply_gestures(self, is_fist):
        # 3.1) Dwell para botones (si el cursor está sobre alguno)
        target_btn = None
        for rect, _, action in self.buttons:
            if rect.contains(self.cur_x, self.cur_y):
                target_btn = action
                break

        now = time.time()
        
        # Cooldown para evitar pulsaciones múltiples rápidas
        if now - self.last_button_press < self.button_cooldown:
            self.hover_target = None
            self.hover_start = None
            return

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
                    self.last_button_press = now

        # 3.2) Mano cerrada para dibujar, abierta solo para moverse
        # Solo dibujar si estamos en el área del canvas (no en el sidebar)
        if self.cur_x >= SIDEBAR_W and self.hover_target is None:
            self.pen_down = is_fist
        else:
            self.pen_down = False
            self.last_pt = None

        # 3.3) Dibujo continuo si pen_down
        if self.pen_down and self.cur_x >= SIDEBAR_W:
            # Ajustar coordenadas para el área de dibujo (restar el ancho del sidebar)
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
            self.auto_save()
        elif action.startswith("color"):
            self.color_idx = int(action[5:])  # Extraer el número del color
        elif action == "clear":
            self.clear_canvas()
        
        # Actualizar etiquetas de botones
        self.update_button_labels()

    def update_button_labels(self):
        # Actualizar las etiquetas de los botones de color
        self.buttons = self._make_buttons()

    def clear_canvas(self):
        self.canvas.fill(Qt.white)\
    def draw_at(self, pt: QPoint):
        painter = QPainter(self.canvas)
        if self.mode == "erase":
            pen = QPen(Qt.white, ERASER_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        else:
            pen = QPen(COLORS[self.color_idx], PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        if self.last_pt is not None:
            painter.drawLine(self.last_pt, pt)

        painter.end()
        self.last_pt = QPoint(pt)


    #def draw_at(self, pt: QPoint):
    #    painter = QPainter(self.canvas)
    #    if self.mode == "erase":
    #        pen = QPen(Qt.white, ERASER_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    #    else:
    #        pen = QPen(COLORS[self.color_idx], PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    #    painter.setPen(pen)
#
    #    if self.last_pt is None:
    #        painter.drawPoint(pt)
    #    else:
    #        painter.drawLine(self.last_pt, pt)
#
    #    painter.end()
    #    self.last_pt = QPoint(pt)

    def auto_save(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = f"dibujo_{ts}.png"
        self.canvas.save(out, "PNG")
        print(f"Imagen guardada como: {out}")

    def render_scene(self, is_fist):
        # Fondo final
        frame = QPixmap(self.width(), self.height())
        frame.fill(QColor(245, 245, 245))

        painter = QPainter(frame)

        # 1) Dibujar sidebar con fondo gris
        sidebar_rect = QRect(0, 0, SIDEBAR_W, CANVAS_H)
        painter.fillRect(sidebar_rect, QColor(240, 240, 240))
        
        # 2) Canvas de dibujo (área derecha)
        painter.drawPixmap(SIDEBAR_W, 0, self.canvas)

        # 3) Botones en el sidebar
        for rect, label, action in self.buttons:
            # Estado visual (activo)
            bg = QColor(230, 230, 230)
            
            # Resaltar botón activo
            if action == "draw" and self.mode == "draw":
                bg = QColor(210, 235, 210)
            elif action == "erase" and self.mode == "erase":
                bg = QColor(235, 210, 210)
            elif action.startswith("color") and int(action[5:]) == self.color_idx:
                bg = COLORS[int(action[5:])].lighter(150)  # Color más claro
                
            # Resaltar botón bajo el cursor
            if self.hover_target == action:
                bg = bg.lighter(120)  # Hacer más claro
                
            painter.fillRect(rect, bg)
            
            # Borde del botón
            painter.setPen(QPen(Qt.black, 1))
            painter.drawRect(rect)
            
            # Texto del botón
            painter.setPen(Qt.black)
            painter.setFont(self.font_big)
            painter.drawText(rect, Qt.AlignCenter, label)

            # Indicador de dwell (progreso en el botón apuntado)
            if self.hover_target == action and self.hover_start is not None:
                frac = min(1.0, (time.time() - self.hover_start) * 1000.0 / DWELL_MS)
                # Dibujar una barra superior como progreso
                prog_rect = QRect(rect.left(), rect.top(), int(rect.width() * frac), 4)
                painter.fillRect(prog_rect, QColor(60, 60, 60))

        # 4) HUD (estado) - en la parte superior del área de dibujo
        painter.setFont(self.font_small)
        hud = f"Modo: {'Dibujar' if self.mode=='draw' else 'Borrar'} | Color: {COLOR_NAMES[self.color_idx]} | "
        
        if self.cur_z is not None:
            hud += f"Distancia: {self.cur_z}mm | "
            if is_fist is not None:
                hud += f"Mano: {'Cerrada (Dibujando)' if is_fist else 'Abierta (Moviendo)'}"
        
        painter.setPen(Qt.black)
        painter.drawText(SIDEBAR_W + 12, 24, hud)

        # 5) Cursor (en toda el área, incluyendo sidebar)
        # círculo externo
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS, CURSOR_RADIUS)
        
        # relleno y forma según el estado de la mano
        if is_fist and self.cur_x >= SIDEBAR_W:
            # Mano cerrada - círculo relleno (solo en área de dibujo)
            painter.setBrush(COLORS[self.color_idx] if self.mode == "draw" else Qt.white)
            painter.drawEllipse(QPoint(self.cur_x, self.cur_y), CURSOR_RADIUS - 4, CURSOR_RADIUS - 4)
        else:
            # Mano abierta - cruz
            painter.drawLine(self.cur_x - 8, self.cur_y, self.cur_x + 8, self.cur_y)
            painter.drawLine(self.cur_x, self.cur_y - 8, self.cur_x, self.cur_y + 8)

        painter.end()
        self.view.setPixmap(frame)


if __name__ == "__main__":
    # Acelera PyQt en algunos drivers
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    w = KinectPaint()
    w.show()
    sys.exit(app.exec_())