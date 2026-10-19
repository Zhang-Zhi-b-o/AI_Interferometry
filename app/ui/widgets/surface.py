"""可旋转、导出的 Qt 3D 形貌视图，按需加载 Matplotlib。"""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton, QVBoxLayout

from app.services.surface_data import surface_mesh


class SurfaceDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        metrics = result.get("metrics") or {}
        anchored = bool(metrics.get("has_absolute_anchor"))
        calibrated = result.get("mode", metrics.get("mode")) == "calibrated"
        if anchored:
            title, zlabel = "有独立锚点的厚度形貌", "Thickness (μm)"
        elif calibrated:
            title, zlabel = "标定相对厚度形貌", "Calibrated relative height (μm)"
        else:
            title, zlabel = "相对厚度起伏（未经物理精度验证）", "Relative height (μm)"
        if metrics.get("data_source") == "demo":
            title = "演示数据 · " + title
        self.setWindowTitle(title)
        self.resize(820, 640)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title + "；X/Y 为分析区域像素坐标，拖动可旋转。"))
        self.figure = Figure(figsize=(7.5, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)
        x, y, z, triangles = surface_mesh(result.get("thickness"), result.get("mask"))
        self.axes = self.figure.add_subplot(111, projection="3d")
        surface = self.axes.plot_trisurf(x, y, z, triangles=triangles, cmap="turbo",
                                        linewidth=0, antialiased=True)
        self.axes.set(xlabel="ROI X (px)", ylabel="ROI Y (px)", zlabel=zlabel)
        self.figure.colorbar(surface, ax=self.axes, shrink=0.65, label=zlabel)
        self.canvas.draw()
        save = QPushButton("保存当前视角 PNG…")
        save.clicked.connect(self._save)
        layout.addWidget(save)
        self.status = QLabel("")
        layout.addWidget(self.status)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存 3D 形貌", "", "PNG (*.png)")
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=180)
            self.status.setText("已保存：" + path)
        except OSError as exc:
            self.status.setText("保存失败：" + str(exc))
