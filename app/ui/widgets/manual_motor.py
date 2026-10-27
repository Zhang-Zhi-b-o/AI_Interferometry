"""手动电机命令；操作前停止所有自动闭环。"""
from PySide6.QtWidgets import QSpinBox
from app.ui.widgets.common import Card, form_grid, hstack, secondary_button


class ManualMotor(Card):
    def __init__(self, ctx):
        super().__init__("手动电机", "R 正转 · r 反转 · S 停止 · D 换向 · + 加速 · - 减速")
        self.ctx = ctx
        self.gear = QSpinBox()
        self.gear.setRange(1, 10)
        self.gear.setValue(int(ctx.config.get("motor.default_gear", 10)))
        self.add(form_grid([("档位（1最快，10最慢）", self.gear)]))
        self.buttons = []
        for actions in ((("正转 R", "forward"),("反转 r", "reverse"),("停止 S", "stop")),
                        (("运行中换向 D", "toggle_direction"),("加速 +", "speed_up"),("减速 -", "speed_down"))):
            buttons = [secondary_button(title, lambda checked=False, op=op: self.command(op))
                       for title, op in actions]
            self.buttons.extend(buttons)
            self.add(hstack(*buttons))
        self.add(hstack(secondary_button("设置档位", lambda: self.command("set_speed")),
                        secondary_button("查询 JSON 状态", self.query)))

    def command(self, operation):
        if not self.ctx.motion.connected:
            self.ctx.log("请先连接电机")
            return
        # 停止旧闭环并等待其线程退出，避免 finally 中的停车覆盖手动命令。
        if operation == "stop":
            self.ctx.stop_motion()
            return
        with self.ctx.motion.control_lock:
            if any(
                    getattr(getattr(self.ctx, name, None), "active", False)
                    for name in ("search", "target", "backlash")):
                self.ctx.stop_motion()
                self.ctx.log("已停止自动闭环，请待停止完成后再次发送手动命令")
                return
            if operation == "set_speed":
                self.ctx.motion.set_speed(self.gear.value())
            else:
                getattr(self.ctx.motion, operation)()
        self.ctx.log("手动电机命令：" + operation)

    def query(self):
        self.ctx.run_async(self.ctx.motion.query_status,
                           on_done=lambda result: self.ctx.log("电机状态：" + str(result)),
                           on_error=lambda exc: self.ctx.log("查询电机失败：" + str(exc)),
                           name="motor-status")
