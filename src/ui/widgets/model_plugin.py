"""YOLO 模型加载与预测插件（鼠标拖拽ROI + 预测结果显示）"""
from __future__ import annotations
import tkinter as tk


class ModelPluginPanel(tk.LabelFrame):
    """模型控制：参数/加载/预测/停止 + ROI框选 + 结果展示"""

    def __init__(self, parent: tk.Widget, confidence: float = 0.5,
                 iou: float = 0.45, imgsz: int = 640):
        super().__init__(parent, text="模型与预测", bg="#ffffff", fg="#000000")
        btn = dict(relief=tk.FLAT, bd=0, bg="#111111", fg="#ffffff",
                   activebackground="#0b0b0b", cursor="hand2")
        sm_btn = dict(relief=tk.FLAT, bd=0, bg="#444444", fg="#ffffff",
                      activebackground="#333333", cursor="hand2")

        self.conf_var = tk.StringVar(value=str(confidence))
        self.iou_var = tk.StringVar(value=str(iou))
        self.imgsz_var = tk.StringVar(value=str(imgsz))

        # ROI 状态
        self.roi_mode_var = tk.BooleanVar(value=False)
        self.roi_pixels: tuple[int,int,int,int] | None = None  # (x1,y1,x2,y2) 像素坐标

        # 预测结果展示
        self.result_var = tk.StringVar(value="")

        # -- 参数 --
        for label, var in [("置信度阈值", self.conf_var), ("IoU阈值", self.iou_var), ("推理尺寸", self.imgsz_var)]:
            r = tk.Frame(self, bg="#fff")
            r.pack(fill=tk.X, padx=8, pady=3)
            tk.Label(r, text=label, bg="#fff", fg="#000").pack(side=tk.LEFT)
            tk.Entry(r, textvariable=var, width=8).pack(side=tk.LEFT, padx=(8,0))

        # -- 预测按钮 --
        for text, cmd in [("加载YOLO模型","load"),("开始预测","start"),("单帧预测","single"),("停止预测","stop")]:
            tk.Button(self, text=text, command=lambda c=cmd: self._emit(c), **btn).pack(fill=tk.X, padx=8, pady=2)

        # -- 分隔 --
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=6)

        # -- ROI 框选 --
        rr = tk.Frame(self, bg="#fff")
        rr.pack(fill=tk.X, padx=8, pady=2)
        tk.Checkbutton(rr, text="鼠标框选ROI（在视频上拖拽画框）", variable=self.roi_mode_var,
                       command=lambda: self._emit("roi_toggle"),
                       bg="#fff", fg="#000", activebackground="#fff", selectcolor="#fff").pack(side=tk.LEFT)
        tk.Button(rr, text="清除ROI", command=lambda: self._emit("roi_clear"), **sm_btn).pack(side=tk.LEFT, padx=(8,0))

        # -- 预测结果 --
        tk.Frame(self, bg="#e5e5e5", height=1).pack(fill=tk.X, padx=8, pady=6)
        tk.Label(self, text="预测结果", bg="#fff", fg="#000",
                 font=("Microsoft YaHei UI", 9, "bold")).pack(anchor="w", padx=8, pady=(2,0))
        tk.Label(self, textvariable=self.result_var, bg="#fff", fg="#333",
                 anchor="w", justify="left", font=("Consolas", 9)).pack(fill=tk.X, padx=8, pady=(0,6))

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def on_command(self, cmd: str):
        """cmd: load/start/single/stop/roi_toggle/roi_clear"""
        pass

    def _emit(self, cmd: str):
        self.on_command(cmd)

    # ------------------------------------------------------------------
    # 取值
    # ------------------------------------------------------------------
    @property
    def conf(self) -> float:
        try: return min(max(float(self.conf_var.get().strip() or "0.5"), 0.0), 1.0)
        except ValueError: return 0.5

    @property
    def iou(self) -> float:
        try: return min(max(float(self.iou_var.get().strip() or "0.45"), 0.0), 1.0)
        except ValueError: return 0.45

    @property
    def imgsz(self) -> int:
        try: return max(32, int(self.imgsz_var.get().strip() or "640"))
        except ValueError: return 640

    @property
    def roi_mode(self) -> bool:
        return self.roi_mode_var.get()

    def set_roi(self, x1: int, y1: int, x2: int, y2: int):
        """设置 ROI 像素坐标"""
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            self.roi_pixels = None
            self.roi_mode_var.set(False)
        else:
            self.roi_pixels = (x1, y1, x2, y2)

    def get_roi_xywh(self) -> tuple[int,int,int,int] | None:
        """返回 (x, y, w, h)"""
        if self.roi_pixels is None:
            return None
        x1, y1, x2, y2 = self.roi_pixels
        return (x1, y1, x2 - x1, y2 - y1)

    # ------------------------------------------------------------------
    # 更新预测结果
    # ------------------------------------------------------------------
    def update_results(self, class_conf: dict, box_count: int, recommend: str):
        """显示最新预测结果"""
        lines = [f"检测目标: {box_count} 个", f"建议: {recommend}"]
        for name, conf in sorted(class_conf.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {conf:.2f}")
        self.result_var.set("\n".join(lines))
