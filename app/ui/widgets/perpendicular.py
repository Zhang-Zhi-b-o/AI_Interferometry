"""One-step, live extreme-position guidance without image or trajectory recording."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel
from app.ui.widgets.common import Card, form_grid, hstack, primary_button, secondary_button


def alignment_palette(state):
    """Shared colors for the card and the prominent video guidance."""
    if state.get('mode') == 'idle':
        return '#64748b', '#f1f5f9'
    if state.get('aligned'):
        return '#15803d', '#dcfce7'
    if state.get('approaching'):
        return '#b45309', '#fef3c7'
    return '#1d4ed8', '#dbeafe'


def alignment_text(state):
    if state['mode'] == 'idle':
        return state['message'], False
    extreme = state['extreme_label']
    note = ''
    if state.get('release_note'):
        side = '右' if state['extreme'] == 'right' else '左'
        note = f' · 上次记录的极限在稳定停留后仍不可复现，已回落到可复现位置，请继续向{side}缓慢确认'
    if state['current_x'] is None:
        reason = state.get('reject_reason') or '等待新鲜、可信的中心条纹'
        text, aligned = f'{reason} · 实时寻找{extreme}极限', False
    elif state['reference_x'] is None:
        text, aligned = (f"正在建立中心条纹位置集合 · {state['position_count']} 个位置"
                         ' · 等待极限附近至少三个观测'), False
    elif state['aligned']:
        text, aligned = f'✓ 已到当前{extreme}极限（允许误差内）', True
    elif state.get('approaching'):
        text, aligned = (f"快到垂直参考位置，请放慢转动"
                         f" · 距当前{extreme}极限 {abs(state['delta_px']):.1f} px"), False
    else:
        text, aligned = f"实时寻找{extreme}极限 · 距当前极限 {state['delta_px']:+.2f} px", False
    return text + note, aligned


class PerpendicularCard(Card):
    def __init__(self, ctx):
        super().__init__('玻璃片自动找垂直', '实时维护中心条纹位置集合，从有多帧支持的位置中寻找最左或最右极限；不保存图像。')
        self.ctx = ctx
        self.service = ctx.perpendicular
        self._error = ''
        self.extreme = QComboBox()
        self.extreme.addItem('最左为极限', 'left')
        self.extreme.addItem('最右为极限', 'right')
        self.extreme.setCurrentIndex(self.extreme.findData(self.service.extreme))
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(.1, 500)
        self.tolerance.setValue(self.service.tolerance_px)
        self.tolerance.setSuffix(' px')
        self.approach = QDoubleSpinBox()
        self.approach.setRange(.1, 1000)
        self.approach.setValue(self.service.approach_tolerance_px)
        self.approach.setSuffix(' px')
        self.add(form_grid([('垂直参考判据', self.extreme),
                            ('到位允许误差 ±', self.tolerance),
                            ('提前提醒范围 ±', self.approach)]))
        self.start = primary_button('开始 / 重置极限', self._start)
        self.stop = secondary_button('停止', self._stop)
        self.add(hstack(self.start, self.stop))
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.add(self.status)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.add(self.details)
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.extreme.currentIndexChanged.connect(self._extreme_changed)
        self.tolerance.valueChanged.connect(self._tolerance_changed)
        self.approach.valueChanged.connect(self._approach_changed)
        self.refresh()

    def _extreme_changed(self, *_):
        self.service.set_extreme(self.extreme.currentData())
        self._error = ''
        self.refresh()

    def _tolerance_changed(self, *_):
        self.service.configure_tolerance(self.tolerance.value())
        self._error = ''
        self.refresh()

    def _approach_changed(self, *_):
        self.service.configure_approach_tolerance(self.approach.value())
        self.refresh()

    def _start(self):
        try:
            if not self.ctx.camera.is_open:
                raise ValueError('请先打开干涉相机')
            if not self.ctx.detect.is_loaded:
                raise ValueError('请先加载条纹模型')
            if any(getattr(self.ctx, key).active for key in ('search', 'target')):
                raise ValueError('请先停止寻中和目标定位，再手动旋转玻璃片')
            self.service.start(self.tolerance.value())
            self._error = ''
        except ValueError as exc:
            self._error = str(exc)
            self.ctx.log(str(exc))
        self.refresh()

    def _stop(self):
        self._error = ''
        self.service.stop()
        self.refresh()

    def refresh(self):
        state = self.service.snapshot()
        text, aligned = alignment_text(state)
        text = self._error or text
        if self.status.text() != text:
            self.status.setText(text)
        color, background = alignment_palette(state)
        if self.status.property('alignment_color') != color:
            self.status.setProperty('alignment_color', color)
            self.status.setStyleSheet(f'color: {color}; background: {background}; border-radius: 6px; '
                                     'font-size: 19px; font-weight: 700; padding: 8px;')
        self.stop.setEnabled(self.service.active)
        reference = ('等待可信极限' if state['reference_x'] is None else
                     f"{state['extreme_label']}参考 {state['reference_x']:.2f} px")
        self.details.setText(f"{reference} · 位置集合 {state['position_count']} 个 · 已排除跳点 {state['rejected_positions']} 个"
                             f" · 允许误差 ±{state['tolerance_px']:.2f} px"
                             f" · 提前提醒 ±{state['approach_tolerance_px']:.1f} px"
                             f" · 重新捕获 {state['reacquisitions']} 次"
                             f" · 放宽不可达极限 {state['released_extremes']} 次。"
                             '保留最近两分钟的位置，多帧支持后更新极限；稳定停留后仍无法复现的旧极限会自动放宽；点重置可清除。')
