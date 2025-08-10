import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


# ------------------------------
# Configuration
# ------------------------------
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
SMOOTHING_ALPHA = 0.35
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 28

HAAR_FACE = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
HAAR_NOSE = str(Path(cv2.data.haarcascades) / "haarcascade_mcs_nose.xml")


class ExponentialSmoother:
    def __init__(self, alpha: float, initial: Optional[Tuple[float, float]] = None):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.value_x = None if initial is None else float(initial[0])
        self.value_y = None if initial is None else float(initial[1])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self.value_x is None:
            self.value_x = x
            self.value_y = y
            return x, y
        self.value_x = self.alpha * x + (1.0 - self.alpha) * self.value_x
        self.value_y = self.alpha * y + (1.0 - self.alpha) * self.value_y
        return self.value_x, self.value_y


def clamp(val: float, vmin: float, vmax: float) -> float:
    return float(max(vmin, min(vmax, val)))


class NoseTrackerOpenCV:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(HAAR_FACE)
        self.nose_cascade = cv2.CascadeClassifier(HAAR_NOSE) if os.path.exists(HAAR_NOSE) else None
        self.smoother = ExponentialSmoother(alpha=SMOOTHING_ALPHA)

    def estimate_nose(self, gray: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
        if len(faces) == 0:
            return None, None
        rx, ry, rw, rh = max(faces, key=lambda r: r[2] * r[3])
        nose_pt: Optional[Tuple[int, int]] = None
        roi_gray = gray[ry:ry + rh, rx:rx + rw]
        if self.nose_cascade is not None:
            noses = self.nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(noses) > 0:
                nx, ny, nw, nh = max(noses, key=lambda n: n[2] * n[3])
                nose_pt = (rx + nx + nw // 2, ry + ny + nh // 2)
        if nose_pt is None:
            nose_pt = (rx + rw // 2, ry + int(rh * 0.58))
        h, w = gray.shape[:2]
        sm_x, sm_y = self.smoother.update(clamp(nose_pt[0] / w, 0.0, 1.0), clamp(nose_pt[1] / h, 0.0, 1.0))
        return (int(sm_x * (w - 1)), int(sm_y * (h - 1))), (rx, ry, rw, rh)


class HandDetectorOpenCV:
    @staticmethod
    def compute_skin_mask(bgr_img: np.ndarray, exclude_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        if exclude_rect is not None:
            rx, ry, rw, rh = exclude_rect
            cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), 0, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return mask

    @staticmethod
    def find_largest_contour(bin_img: np.ndarray, min_area: int = 6000) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            return None
        return contour

    @staticmethod
    def estimate_extended_fingers(contour: np.ndarray) -> int:
        if contour is None or len(contour) < 5:
            return 0
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return 0
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None:
            return 0
        finger_gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i, 0]
            if depth > 10000:
                finger_gaps += 1
        return int(max(0, min(5, finger_gaps + 1)))


class PaintEngine:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.background = (0, 0, 0)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_point: Optional[Tuple[int, int]] = None
        self.brush_thickness = BRUSH_THICKNESS
        self.eraser_thickness = ERASER_THICKNESS
        self.current_color = (57, 255, 20)  # neon green
        self.using_eraser = False

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        self.canvas = cv2.resize(self.canvas, (width, height), interpolation=cv2.INTER_NEAREST)

    def pen_down_draw(self, point: Tuple[int, int]) -> None:
        if self.last_point is not None:
            color = self.background if self.using_eraser else self.current_color
            thickness = self.eraser_thickness if self.using_eraser else self.brush_thickness
            cv2.line(self.canvas, self.last_point, point, color, thickness=thickness, lineType=cv2.LINE_AA)
        self.last_point = point

    def move_without_drawing(self, point: Tuple[int, int]) -> None:
        self.last_point = point

    def clear(self) -> None:
        self.canvas[:] = self.background

    def set_color(self, bgr: Tuple[int, int, int]) -> None:
        self.current_color = bgr
        self.using_eraser = False

    def set_eraser(self) -> None:
        self.using_eraser = True


def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class VideoPaintWidget(QtWidgets.QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setScaledContents(True)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #000000; border-radius: 12px;")
        self.setMouseTracking(True)
        self._last_mouse_pos = QtCore.QPoint(-1, -1)
        self.left_pressed = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._last_mouse_pos = event.pos()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.left_pressed = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.left_pressed = False
        super().mouseReleaseEvent(event)

    def get_mouse_pos(self) -> Tuple[int, int]:
        return self._last_mouse_pos.x(), self._last_mouse_pos.y()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nose Painting - Neon")
        self.setStyleSheet("""
            QMainWindow { background-color: #0c0c0f; }
            QLabel, QPushButton { color: #E0E0E0; font-size: 14px; }
            QPushButton { background-color: #141419; border: 1px solid #2a2a33; padding: 8px 12px; border-radius: 10px; }
            QPushButton:hover { border: 1px solid #39FF14; }
            QPushButton:pressed { background-color: #1b1b24; }
            QStatusBar { background: #0c0c0f; color: #8f9ba8; }
        """)

        self.video_label = VideoPaintWidget()

        # Toolbar panel
        self.toolbar = QtWidgets.QFrame()
        self.toolbar.setStyleSheet("QFrame { background-color: #0e0e12; border-radius: 12px; }")
        toolbar_layout = QtWidgets.QVBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(10, 10, 10, 10)
        toolbar_layout.setSpacing(10)

        # Color buttons (neon palette)
        self.color_buttons: list[QtWidgets.QPushButton] = []
        neon_colors = [
            ("Neon Green", (57, 255, 20)),
            ("Neon Pink", (149, 0, 255)),
            ("Neon Cyan", (255, 255, 0)),
            ("Neon Yellow", (0, 255, 255)),
            ("Neon Orange", (0, 110, 255)),
            ("Neon Blue", (255, 0, 0)),
            ("White", (255, 255, 255)),
        ]
        self.selected_color = neon_colors[0][1]

        def make_color_button(name: str, bgr: Tuple[int, int, int]) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton(name)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setStyleSheet(f"QPushButton {{ color: #E0E0E0; border: 2px solid rgba({bgr[2]}, {bgr[1]}, {bgr[0]}, 200); }}"
                              "QPushButton:hover { background-color: #181821; }")
            btn.clicked.connect(lambda: self.on_select_color(bgr))
            return btn

        for name, bgr in neon_colors:
            toolbar_layout.addWidget(make_color_button(name, bgr))

        # Eraser / Clear / Save buttons
        self.eraser_btn = QtWidgets.QPushButton("Eraser")
        self.eraser_btn.clicked.connect(self.on_eraser)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.on_save)

        for b in (self.eraser_btn, self.clear_btn, self.save_btn):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            toolbar_layout.addWidget(b)
        toolbar_layout.addStretch(1)

        # Mode toggle (Nose vs Mouse)
        self.mode_toggle = QtWidgets.QPushButton("Mode: Nose")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(False)
        self.mode_toggle.toggled.connect(self.on_toggle_mode)
        toolbar_layout.addWidget(self.mode_toggle)

        # Status panel
        self.status_panel = QtWidgets.QLabel("Mode: Idle | Color: Neon Green")
        self.statusBar().addPermanentWidget(self.status_panel)

        # Layout: toolbar left, video right
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.video_label, 1)
        self.setCentralWidget(central)

        # Capture and engines
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Camera Error", "Could not open webcam.")
            sys.exit(1)

        self.nose_tracker = NoseTrackerOpenCV()
        self.hand_detector = HandDetectorOpenCV()
        self.painter = PaintEngine(FRAME_WIDTH, FRAME_HEIGHT)

        self.pen_down = False
        self.use_mouse_mode = False

        # Timer for frame updates
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_frame)
        self.timer.start(1000 // 30)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, activated=self.on_clear)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=self.on_save)

        self.resize(1220, 760)

    def on_select_color(self, bgr: Tuple[int, int, int]) -> None:
        self.painter.set_color(bgr)
        self.selected_color = bgr
        self.status_panel.setText(f"Mode: {'Erase' if self.painter.using_eraser else ('Draw' if self.pen_down else 'Idle')} | Color: {bgr}")

    def on_eraser(self) -> None:
        self.painter.set_eraser()
        self.status_panel.setText("Mode: Erase | Color: -")

    def on_clear(self) -> None:
        self.painter.clear()

    def on_save(self) -> None:
        out_path = Path(f"painting_{time.strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(str(out_path), self.painter.canvas)
        self.statusBar().showMessage(f"Saved to {out_path}", 3000)

    def on_toggle_mode(self, checked: bool) -> None:
        self.use_mouse_mode = checked
        self.mode_toggle.setText("Mode: Mouse" if checked else "Mode: Nose")

    def on_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self.painter.resize(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_mouse_mode:
            mx, my = self.video_label.get_mouse_pos()
            mx = int(clamp(mx, 0, self.video_label.width() - 1))
            my = int(clamp(my, 0, self.video_label.height() - 1))
            pix_w, pix_h = self.video_label.width(), self.video_label.height()
            if pix_w > 0 and pix_h > 0:
                sx = w / pix_w
                sy = h / pix_h
                px, py = int(mx * sx), int(my * sy)
            else:
                px, py = w // 2, h // 2
            nose_point = (px, py)
            self.pen_down = self.video_label.left_pressed
            face_rect = None
        else:
            nose_point, face_rect = self.nose_tracker.estimate_nose(gray)
            self.pen_down = False
            if nose_point is not None:
                skin_mask = HandDetectorOpenCV.compute_skin_mask(frame, face_rect)
                hand_contour = HandDetectorOpenCV.find_largest_contour(skin_mask)
                if hand_contour is not None:
                    extended = HandDetectorOpenCV.estimate_extended_fingers(hand_contour)
                    self.pen_down = extended <= 1
                    cv2.drawContours(frame, [hand_contour], -1, (255, 0, 255), 2)

        if nose_point is not None:
            if self.pen_down:
                self.painter.pen_down_draw(nose_point)
            else:
                self.painter.move_without_drawing(nose_point)
            cv2.circle(frame, nose_point, 6, (0, 255, 0), -1)
        if face_rect is not None:
            rx, ry, rw, rh = face_rect
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (60, 60, 60), 1)

        composed = cv2.addWeighted(frame, 0.6, self.painter.canvas, 1.0, 0)

        mode_text = "Erase" if self.painter.using_eraser else ("Draw" if self.pen_down else "Idle")
        cv2.putText(composed, f"{mode_text}", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (57, 255, 20) if self.pen_down else (200, 200, 200), 2, cv2.LINE_AA)

        qimg = bgr_to_qimage(composed)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Nose Painting - Neon")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


