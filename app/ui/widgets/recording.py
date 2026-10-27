"""旧版录制来源和帧率控制的 Qt 入口。"""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QFileDialog, QSpinBox

from app.ui.widgets.common import Card, form_grid, hstack, secondary_button, label


class RecordingCard(Card):
    def __init__(self, ctx):
        super().__init__("视频录制", "录制不改变识别与测量输入；视频旁保存来源信息。")
        self.ctx = ctx
        self.source = QComboBox()
        for title, source in (("主相机原始画面", "raw"), ("旋转校正画面", "corrected"),
                              ("检测标注画面", "annotated"), ("读数相机画面", "reading")):
            self.source.addItem(title, source)
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(15)
        self.add(form_grid([("来源", self.source), ("输出帧率", self.fps)]))
        self.start_button = secondary_button("开始录制", self._start)
        self.stop_button = secondary_button("停止并保存", self._stop)
        self.add(hstack(self.start_button, self.stop_button, spacing=8))
        self.status = label("未录制", name="Hint", wrap=True)
        self.add(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)
        self.refresh()

    def _start(self):
        source = self.source.currentData()
        if source == "reading" and not self.ctx.reading.connected:
            self.ctx.log("请先启动读数相机")
            return
        if source != "reading" and self.ctx.frame() is None:
            self.ctx.log("请先打开主相机")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存录像", "", "MP4 (*.mp4);;AVI (*.avi)")
        if not path:
            return
        try:
            _, message = self.ctx.media.start(path, fps=self.fps.value(), source=source,
                metadata={"data_source": "demo" if self.ctx.demo_mode else "live",
                          "rotation_deg": getattr(self.ctx.camera, "rotation", 0),
                          "roi": self.ctx.roi})
            self.ctx.log(message)
        except Exception as exc:
            self.ctx.log(f"无法开始录制：{exc}")
        self.refresh()

    def _stop(self):
        self.ctx.run_async(self.ctx.media.stop, on_done=lambda _: self.refresh(), name="record-stop")

    def refresh(self):
        active = self.ctx.media.active
        self.source.setEnabled(not active)
        self.fps.setEnabled(not active)
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self.status.setText(f"{'录制中' if active else '未录制'} · {self.ctx.media.frames} 帧 · 丢弃 {self.ctx.media.dropped} 帧")
