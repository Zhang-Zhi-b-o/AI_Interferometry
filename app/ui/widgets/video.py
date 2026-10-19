"""主画面：显示相机帧、绘制叠加层、支持拖拽框选 ROI。

只做显示与坐标换算，不碰相机本身——画面由 ``MainWindow`` 推给它。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class VideoView(QWidget):
    roi_changed = Signal(object)   # (x, y, w, h) 或 None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._frame_size = (0, 0)
        self._overlay: dict = {}
        self._roi: tuple[int, int, int, int] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._show_roi = True
        self.setStyleSheet("background: #11161f; border-radius: 12px;")

    # ------------------------------------------------------------------
    def set_frame(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            return
        height, width = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self._image = image.copy()
        self._frame_size = (width, height)
        self.update()

    def clear(self) -> None:
        self._image = None
        self.update()

    def set_overlay(self, **values) -> None:
        """支持的键：center_x / boxes / center_conf / status。"""
        self._overlay.update({k: v for k, v in values.items() if v is not None})
        self.update()

    def clear_overlay(self) -> None:
        self._overlay.clear()
        self.update()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self._roi = roi
        self.update()

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    # ------------------------------------------------------------------
    def _target_rect(self) -> QRect:
        if self._image is None:
            return QRect(0, 0, self.width(), self.height())
        view_w, view_h = self.width(), self.height()
        img_w, img_h = self._image.width(), self._image.height()
        if img_w <= 0 or img_h <= 0:
            return QRect(0, 0, view_w, view_h)
        scale = min(view_w / img_w, view_h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        return QRect((view_w - draw_w) // 2, (view_h - draw_h) // 2, draw_w, draw_h)

    def _to_image(self, point: QPoint) -> tuple[float, float] | None:
        rect = self._target_rect()
        if rect.width() <= 0 or rect.height() <= 0 or self._frame_size[0] == 0:
            return None
        img_w, img_h = self._frame_size
        x = (point.x() - rect.x()) * img_w / rect.width()
        y = (point.y() - rect.y()) * img_h / rect.height()
        return max(0.0, min(img_w - 1.0, x)), max(0.0, min(img_h - 1.0, y))

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#11161f"))
        rect = self._target_rect()

        if self._image is not None:
            # 圆角裁剪，和卡片的 12px 圆角保持一致
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), 10.0, 10.0)
            painter.save()
            painter.setClipPath(path)
            painter.drawImage(rect, self._image)
            painter.restore()
        else:
            painter.setPen(QColor("#7f8287"))
            font = QFont()
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "画面未打开\n\n在「实时」页点击「打开相机」")

        if self._image is not None:
            self._paint_overlay(painter, rect)
        self._paint_roi(painter, rect)
        self._paint_drag(painter)

    def _paint_overlay(self, painter: QPainter, rect: QRect) -> None:
        img_w, img_h = self._frame_size
        if img_w == 0 or img_h == 0:
            return
        sx = rect.width() / img_w
        sy = rect.height() / img_h

        # 检测框
        boxes = self._overlay.get("boxes")
        if boxes is not None and len(boxes):
            painter.setPen(QPen(QColor("#5686fe"), 2))
            for box in boxes:
                try:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                except Exception:
                    continue
                painter.drawRect(QRect(
                    int(rect.x() + x1 * sx), int(rect.y() + y1 * sy),
                    int((x2 - x1) * sx), int((y2 - y1) * sy)))

        # 画面中心参考线
        center_view_x = rect.x() + rect.width() // 2
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.PenStyle.DashLine))
        painter.drawLine(center_view_x, rect.y(), center_view_x, rect.bottom())

        # 中心条纹位置
        center_x = self._overlay.get("center_x")
        if center_x is not None:
            px = rect.x() + float(center_x) * sx
            painter.setPen(QPen(QColor("#22c55e"), 2))
            painter.drawLine(int(px), rect.y(), int(px), rect.bottom())
            conf = self._overlay.get("center_conf")
            text = "中心条纹" if conf is None else f"中心条纹 {float(conf):.2f}"
            painter.setPen(QColor("#22c55e"))
            font = QFont("Microsoft YaHei UI")
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(int(px) + 8, rect.y() + 20, text)

        status = self._overlay.get("status")
        if status:
            painter.setPen(QColor("#cfd3d6"))
            font = QFont("Consolas")
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(rect.x() + 10, rect.bottom() - 10, str(status))

    def _paint_roi(self, painter: QPainter, rect: QRect) -> None:
        if not self._show_roi or self._roi is None or self._image is None:
            return
        img_w, img_h = self._frame_size
        if img_w == 0 or img_h == 0:
            return
        sx = rect.width() / img_w
        sy = rect.height() / img_h
        x, y, w, h = self._roi
        painter.setPen(QPen(QColor("#f59e0b"), 2, Qt.PenStyle.DashLine))
        painter.drawRect(QRect(int(rect.x() + x * sx), int(rect.y() + y * sy),
                               int(w * sx), int(h * sy)))

    def _paint_drag(self, painter: QPainter) -> None:
        if self._drag_start is None or self._drag_current is None:
            return
        painter.setPen(QPen(QColor("#f59e0b"), 2))
        painter.drawRect(QRect(self._drag_start, self._drag_current).normalized())

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._image is not None:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._roi = None
            self.roi_changed.emit(None)
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is None or self._drag_current is None:
            return
        start, current = self._drag_start, self._drag_current
        self._drag_start = self._drag_current = None
        p1 = self._to_image(start)
        p2 = self._to_image(current)
        if p1 is None or p2 is None:
            self.update()
            return
        x1, x2 = sorted((p1[0], p2[0]))
        y1, y2 = sorted((p1[1], p2[1]))
        if (x2 - x1) < 12 or (y2 - y1) < 12:  # 太小的框当作误触
            self.update()
            return
        self._roi = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        self.roi_changed.emit(self._roi)
        self.update()
