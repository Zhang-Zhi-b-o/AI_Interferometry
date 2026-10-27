"""⑥ 厚度分布：单帧分析（相对 / 标定）＋ 颜色→光程差标定采集。"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.state import AppState
from app.services.calibration import opd_um_from_readings
from app.ui.pages.base import Page
from app.ui.widgets.common import (
    Card,
    Expander,
    StatusPill,
    field_row,
    form_grid,
    hstack,
    label,
    primary_button,
    secondary_button,
)


class MapPage(Page):
    key = "map"
    title = "⑥ 厚度分布"
    hint = "相对模式观察厚度起伏；定量解释需要颜色标定与独立测量验证。"

    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(ctx, parent)
        self._calibration_path: str = ""
        self._preview: QPixmap | None = None
        self._last_result: dict | None = None
        self._surface_dialog = None
        self._zero_mm: float | None = None
        self._baseline = None
        self._baseline_metadata = None
        self._analysis_metadata = {}
        self._analysis_busy = False
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_analyze_tab(), "分析")
        self.tabs.addTab(self._build_calibration_tab(), "标定采集")
        self.add(self.tabs)
        self.add_stretch()

    # ==================================================================
    # 分析
    # ==================================================================
    def _build_analyze_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        card = Card("参数", "标定模式需要先采一张标定表；相对模式只能看起伏，不能报绝对厚度。")
        self.wavelength = QDoubleSpinBox()
        self.wavelength.setDecimals(1)
        self.wavelength.setRange(350.0, 1200.0)
        self.wavelength.setValue(float(self.ctx.config.get("measurement.wavelength_nm", 589.3)))
        self.film_n = QDoubleSpinBox()
        self.film_n.setDecimals(3)
        self.film_n.setRange(1.001, 3.0)
        self.film_n.setValue(float(self.ctx.config.get("measurement.film_refractive_index", 1.523)))
        self.mode = QComboBox()
        self.mode.addItem("相对模式（不需要标定表）", "relative")
        self.mode.addItem("标定模式（需要标定 CSV）", "calibrated")
        card.add(form_grid([("波长 (nm)", self.wavelength), ("薄膜折射率", self.film_n)], columns=2))
        card.add(field_row("模式", self.mode))

        self.calibration_label = label("未选择标定表", name="Muted", wrap=True)
        self.pick_calibration = secondary_button("选择标定 CSV…", self._pick_calibration)
        card.add(hstack(self.pick_calibration, spacing=8))
        card.add(self.calibration_label)

        self.invert = QCheckBox("反转厚度方向")
        self.whole_region = QCheckBox("整幅都当有效区域（先框选 ROI 再勾选）")
        card.add(self.invert)
        card.add(self.whole_region)

        self.baseline_status = label("无膜基准：未捕获", name="Muted")
        advanced = Expander("无膜基准、绝对锚点与平滑")
        advanced.add(hstack(secondary_button("捕获无膜基准", self._capture_baseline),
                        secondary_button("清除基准", self._clear_baseline)))
        advanced.add(self.baseline_status)
        self.anchor_enabled = QCheckBox("使用独立绝对厚度锚点（μm）")
        self.anchor = QDoubleSpinBox()
        self.anchor.setDecimals(5)
        self.anchor.setRange(0.00001, 100000)
        advanced.add(self.anchor_enabled)
        advanced.add(field_row("独立厚度锚点", self.anchor))
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0, 30)
        self.smooth.setValue(2)
        advanced.add(field_row("平滑 (px)", self.smooth))
        card.add(advanced)
        self.analyze_button = primary_button("分析当前画面", self._analyze)
        card.add(self.analyze_button)
        self.analyze_status = StatusPill("尚未分析", "idle")
        card.add(self.analyze_status)
        layout.addWidget(card)

        result = Card("结果")
        self.preview = QLabel("分析后这里显示伪彩厚度图")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(220)
        self.preview.setStyleSheet(
            "background: #f1f3f5; border: 1px solid #e1e5ee; border-radius: 10px;"
            "color: #979da6; font-size: 14px;")
        result.add(self.preview)
        self.metrics_label = label("", name="Hint", wrap=True)
        result.add(self.metrics_label)
        self.save_button = secondary_button("保存伪彩图", self._save_preview)
        self.save_button.setEnabled(False)
        self.surface_button = secondary_button("查看 3D 形貌…", self._show_surface)
        self.surface_button.setEnabled(False)
        result.add(hstack(self.save_button, self.surface_button))
        layout.addWidget(result)
        layout.addStretch(1)
        return page

    # ==================================================================
    # 标定采集
    # ==================================================================
    def _build_calibration_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        intro = Card("采集流程", "① 记零点 → ② 每移动一段距离按一次「采集」 → ③ 至少 4 点后保存。")
        self.zero_label = label("零点：未记录", name="Hint")
        self.set_zero = secondary_button("把当前读数记为零点", self._set_zero)
        intro.add(self.zero_label)
        intro.add(hstack(self.set_zero, spacing=8))
        layout.addWidget(intro)

        point = Card("采集一个点")
        self.opd_mode = QComboBox()
        self.opd_mode.addItem("按读数差自动算 OPD", "auto")
        self.opd_mode.addItem("手动填 OPD", "manual")
        self.opd_manual = QDoubleSpinBox()
        self.opd_manual.setDecimals(4)
        self.opd_manual.setRange(0.0, 100.0)
        self.opd_manual.setSuffix(" μm")
        point.add(form_grid([("OPD 来源", self.opd_mode)]))
        point.add(field_row("手动 OPD", self.opd_manual))
        self.capture_button = primary_button("采集这个点", self._capture)
        point.add(self.capture_button)
        self.capture_status = StatusPill("尚无标定点", "idle")
        point.add(self.capture_status)
        layout.addWidget(point)

        table_card = Card("标定点", "列：OPD(μm) 与画面中心线颜色 (r,g,b)。")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["OPD (μm)", "R", "G", "B"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(180)
        self.table.horizontalHeader().setStretchLastSection(True)
        table_card.add(self.table)
        self.remove_button = secondary_button("删除选中点", self._remove_selected)
        self.save_cal = primary_button("保存标定 CSV…", self._save_calibration)
        self.load_cal = secondary_button("载入已有 CSV…", self._load_calibration)
        table_card.add(hstack(self.remove_button, self.save_cal, spacing=8))
        table_card.add(self.load_cal)
        layout.addWidget(table_card)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    def _pick_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择颜色标定 CSV", str(self.ctx.config.get("paths.exports", "")),
            "CSV (*.csv);;所有文件 (*.*)")
        if not path:
            return
        self._calibration_path = path
        self.calibration_label.setText(f"标定表：{path}")
        self.mode.setCurrentIndex(1)

    def _analyze(self) -> None:
        if self._analysis_busy:
            return
        frame = self.ctx.frame()
        if frame is None:
            self.analyze_status.set_state("error", "没有画面，请先打开相机")
            return
        roi = getattr(self.ctx, "roi", None)
        if roi and self.whole_region.isChecked():
            x, y, w, h = roi
            frame = frame[y:y + h, x:x + w]
        baseline = None if self._baseline is None else self._baseline.copy()
        if baseline is not None and self._baseline_metadata and (
                self._baseline_metadata["camera_index"] != self.ctx.state.camera_index or
                self._baseline_metadata.get("rotation_deg", 0) != getattr(self.ctx.camera, "rotation", 0)):
            self.analyze_status.set_state("error", "相机或旋转校正已变化，请重新捕获无膜基准")
            return
        if baseline is not None and roi and self.whole_region.isChecked():
            x, y, w, h = roi
            baseline = baseline[y:y + h, x:x + w]
        if baseline is not None and baseline.shape != frame.shape:
            self.analyze_status.set_state("error", "基准与当前画面尺寸不同，请重新捕获基准")
            return
        calibration = self._calibration_path if self.mode.currentData() == "calibrated" else None
        if self.mode.currentData() == "calibrated" and not calibration:
            self.analyze_status.set_state("warn", "标定模式需要先选择标定 CSV")
            return
        params = dict(
            wavelength_nm=float(self.wavelength.value()),
            refractive_index=float(self.film_n.value()),
            calibration=calibration,
            invert=self.invert.isChecked(),
            reference_image=baseline,
            reference_thickness_um=float(self.anchor.value()) if self.anchor_enabled.isChecked() else None,
            smooth_px=float(self.smooth.value()),
            whole_region=bool(roi) and self.whole_region.isChecked(),
        )
        import time
        self._analysis_metadata = {
            "captured_at": time.time(), "data_source": "demo" if getattr(self.ctx, "demo_mode", False) else "live",
            "camera_index": self.ctx.state.camera_index, "roi": list(roi) if roi and self.whole_region.isChecked() else None,
            "rotation_deg": getattr(self.ctx.camera, "rotation", 0),
            "frame_shape": list(frame.shape), "baseline": self._baseline_metadata,
            "parameters": {key: value for key, value in params.items() if key != "reference_image"},
            "thickness_unit": "um", "software_version": __import__("app").__version__,
            "calibration_csv": calibration,
        }
        self._analysis_busy = True
        self.analyze_button.setEnabled(False)
        self.analyze_status.set_state("busy", "分析中…")

        def done(result):
            self._analysis_busy = False
            self.ctx.last_map = {"mode": "calibrated" if calibration else "relative",
                                 "calibration": calibration, "metrics": result.get("metrics", {})}
            self.ctx.timeline.add("thickness_map", parameters=self._analysis_metadata["parameters"],
                                  result=self.ctx.last_map)
            self.analyze_button.setEnabled(True)
            self._show_result(result)

        def failed(exc):
            self._analysis_busy = False
            self.analyze_button.setEnabled(True)
            self.analyze_status.set_state("error", f"分析失败：{exc}")

        self.ctx.run_async(lambda: self.ctx.thickness.analyze(frame, **params),
                           on_done=done, on_error=failed, name="thickness-map")

    def _show_result(self, result: dict) -> None:
        result["source_metadata"] = dict(self._analysis_metadata)
        self._last_result = result
        self.surface_button.setEnabled(result.get("thickness") is not None)
        if self._surface_dialog is not None:
            self._surface_dialog.close()
            self._surface_dialog.deleteLater()
            self._surface_dialog = None
        metrics = result.get("metrics") or {}
        lines = [f"模式：{result.get('mode_label', '')}"]
        for key, text in (("valid_pixels", "有效像素"),
                          ("min_um", "最小厚度 (μm)"),
                          ("max_um", "最大厚度 (μm)"),
                          ("pv_um", "峰谷值 PV (μm)"),
                          ("rms_um", "RMS 不均匀度 (μm)")):
            if key in metrics and metrics[key] is not None:
                value = metrics[key]
                lines.append(f"{text}：{value:.4g}" if isinstance(value, float) else f"{text}：{value}")
        for key, value in metrics.items():
            if key not in ("valid_pixels", "min_um", "max_um", "pv_um", "rms_um"):
                lines.append(f"{key}：{value}")
        self.metrics_label.setText("\n".join(lines))
        overlay = result.get("overlay")
        if overlay is not None:
            self._set_preview(overlay)
            self.save_button.setEnabled(True)
        self.analyze_status.set_state("ok", "分析完成")
        self.ctx.log("厚度分布分析完成")

    def _set_preview(self, bgr: np.ndarray) -> None:
        height, width = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._preview = QPixmap.fromImage(image)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self._preview is None:
            return
        self.preview.setPixmap(self._preview.scaled(
            self.preview.width() - 8, self.preview.height() - 8,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _show_surface(self) -> None:
        if self._last_result is None:
            return
        try:
            from app.ui.widgets.surface import SurfaceDialog

            if self._surface_dialog is None:
                self._surface_dialog = SurfaceDialog(self._last_result, self)
            self._surface_dialog.show()
            self._surface_dialog.raise_()
        except (ImportError, ValueError, RuntimeError) as exc:
            self.analyze_status.set_state("error", f"无法显示 3D：{exc}")

    def _save_preview(self) -> None:
        result = getattr(self, "_last_result", None)
        if not result:
            return
        try:
            import json
            import time
            from pathlib import Path
            import cv2
            directory = Path(self.ctx.record.out_dir) / ("thickness_" + time.strftime("%Y%m%d-%H%M%S") + f"_{time.time_ns() % 1000000:06d}")
            directory.mkdir(parents=True, exist_ok=True)
            for key in ("overlay", "heatmap", "heatmap_legend", "mask"):
                array = result.get(key)
                if array is None:
                    continue
                if key == "mask":
                    array = np.uint8(array) * 255
                ok, encoded = cv2.imencode(".png", array)
                if not ok:
                    raise ValueError(f"无法编码 {key}")
                encoded.tofile(str(directory / f"{key}.png"))
            for key in ("thickness", "normalized_thickness"):
                array = result.get(key)
                if array is not None:
                    np.save(directory / f"{key}.npy", array)
                    np.savetxt(directory / f"{key}.csv", array, delimiter=",")
            (directory / "summary.json").write_text(json.dumps({"metrics": result.get("metrics"),
                "source": result.get("source_metadata")}, ensure_ascii=False, indent=2,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf8")
            path = directory
        except Exception as exc:
            self.analyze_status.set_state("error", f"保存失败：{exc}")
            return
        self.ctx.log(f"已保存：{path}")

    # ------------------------------------------------------------------
    def _capture_baseline(self):
        frame = self.ctx.frame()
        if frame is None:
            self.baseline_status.setText("没有画面，无法捕获基准")
            return
        import time
        self._baseline = frame.copy()
        self._baseline_metadata = {"captured_at": time.time(), "camera_index": self.ctx.state.camera_index,
                                   "rotation_deg": getattr(self.ctx.camera, "rotation", 0),
                                   "frame_shape": list(frame.shape), "reading_mm": self.ctx.state.reading_mm}
        self.baseline_status.setText("无膜基准已捕获；分析时扣除同一区域")

    def _clear_baseline(self):
        self._baseline = None
        self._baseline_metadata = None
        self.baseline_status.setText("无膜基准：未捕获")

    def _set_zero(self) -> None:
        value = self.ctx.state.reading_mm
        if value is None:
            self.capture_status.set_state("warn", "没有读数，请先启动微分表读数")
            return
        self._zero_mm = float(value)
        self.zero_label.setText(f"零点：{self._zero_mm:.4f} mm")
        self.capture_status.set_state("ok", "零点已记录")

    def _capture(self) -> None:
        frame = self.ctx.frame()
        if frame is None:
            self.capture_status.set_state("error", "没有画面")
            return
        if self.opd_mode.currentData() == "auto":
            zero = getattr(self, "_zero_mm", None)
            current = self.ctx.state.reading_mm
            if zero is None or current is None:
                self.capture_status.set_state("warn", "自动模式需要零点与当前读数")
                return
            opd = opd_um_from_readings(zero, current)
        else:
            opd = float(self.opd_manual.value())
            if opd <= 0:
                self.capture_status.set_state("warn", "手动 OPD 必须大于 0")
                return
        try:
            colour = self.ctx.thickness.sample_colour(
                frame, self.ctx.state.center_x)
        except Exception as exc:
            self.capture_status.set_state("error", f"取色失败：{exc}")
            return
        self.ctx.calibration.add(opd_um=opd, colour=colour)
        self.ctx.timeline.add("calibration_point", result={"opd_um": opd, "rgb": list(colour)})
        self._refresh_table()
        self.capture_status.set_state(
            "ok", f"已采集 OPD={opd:.3f} μm  rgb={colour}")

    def _refresh_table(self) -> None:
        points = self.ctx.calibration.points
        self.table.setRowCount(len(points))
        for row, point in enumerate(points):
            for column, text in enumerate(
                    (f"{point.opd_um:.3f}", str(point.r), str(point.g), str(point.b))):
                self.table.setItem(row, column, QTableWidgetItem(text))
        if not points:
            self.capture_status.set_state("idle", "尚无标定点")
        else:
            issues = self.ctx.calibration.problems()
            self.capture_status.set_state(
                "ok" if not issues else "warn",
                f"{len(points)} 个点" + ("" if not issues else "：" + "；".join(issues)))

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.ctx.calibration.remove_at(row)
        self._refresh_table()

    def _save_calibration(self) -> None:
        issues = self.ctx.calibration.problems()
        if issues:
            self.capture_status.set_state("error", "；".join(issues))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存颜色标定 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            target = self.ctx.calibration.save(path)
        except Exception as exc:
            self.capture_status.set_state("error", f"保存失败：{exc}")
            return
        self._calibration_path = str(target)
        self.calibration_label.setText(f"标定表：{target}")
        self.capture_status.set_state("ok", f"已保存 {len(self.ctx.calibration)} 点到 {target}")

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "载入颜色标定 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            count = self.ctx.calibration.load(path)
        except Exception as exc:
            self.capture_status.set_state("error", f"载入失败：{exc}")
            return
        self._calibration_path = path
        self.calibration_label.setText(f"标定表：{path}")
        self._refresh_table()
        self.capture_status.set_state("ok", f"已载入 {count} 个标定点")

    # ------------------------------------------------------------------
    def on_state(self, state: AppState) -> None:
        has_reading = state.reading_mm is not None
        self.set_zero.setEnabled(has_reading)
        if has_reading and self.opd_mode.currentData() == "auto":
            zero = getattr(self, "_zero_mm", None)
            if zero is not None:
                self.opd_manual.setValue(opd_um_from_readings(zero, state.reading_mm))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_preview()
