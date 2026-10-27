"""Keep wheel input for scrolling and video zoom, never form value changes."""
from PySide6.QtCore import QEvent, QObject, QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QApplication,
    QComboBox, QScrollBar, QTabBar, QWidget,
)


class FormWheelFilter(QObject):
    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() != QEvent.Type.Wheel or not isinstance(watched, QWidget):
            return False
        control = watched
        while control is not None:
            if isinstance(control, QScrollBar):
                return False
            if isinstance(control, (QAbstractSpinBox, QComboBox, QAbstractSlider, QTabBar)):
                break
            # Editable children of inputs need the same protection. Scrollable
            # text fields and open dropdown lists retain ordinary list scrolling.
            if isinstance(control, QAbstractScrollArea):
                return False
            control = control.parentWidget()
        if control is None:
            return False
        parent = control.parentWidget()
        while parent is not None and not isinstance(parent, QAbstractScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            viewport = parent.viewport()
            forwarded = QWheelEvent(
                QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
                event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                event.buttons(), event.modifiers(), event.phase(), event.inverted(),
                event.source(),
            )
            QApplication.sendEvent(viewport, forwarded)
        event.accept()
        return True


def install_form_wheel_filter():
    app = QApplication.instance()
    if not hasattr(app, "_form_wheel_filter"):
        app._form_wheel_filter = FormWheelFilter(app)
        app.installEventFilter(app._form_wheel_filter)
