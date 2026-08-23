"""薄膜厚度分布结果弹窗：热力图 + 详细数据 + 可旋转三维厚度分布图。"""
from __future__ import annotations

import tkinter as tk

import numpy as np

FONT = "Microsoft YaHei UI"
_BG = "#f3f6fb"
_SURFACE = "#ffffff"
_TEXT = "#182230"
_MUTED = "#667085"
_BORDER = "#dfe7f1"


def _configure_matplotlib_chinese() -> None:
    """让 matplotlib 支持中文标签（找不到中文字体时静默回退英文）。"""
    try:
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Microsoft JhengHei",
            "Arial Unicode MS", "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


class ThicknessViewer(tk.Toplevel):
    """单帧厚度分布结果的独立窗口。

    ``result`` 来自 ``analyze_thickness_distribution`` 的返回字典；
    ``anchor_um`` 为可选的绝对厚度锚定基准（μm），用于在数据区说明来源。
    """

    def __init__(self, master: tk.Widget, result: dict,
                 anchor_um: float | None = None,
                 title: str = "单帧厚度分布"):
        super().__init__(master)
        self.title(title)
        self.configure(bg=_BG)
        self.resizable(True, True)
        self._canvas = None
        self._photo = None
        self._build(result, anchor_um)

    # ------------------------------------------------------------------
    def _build(self, result: dict, anchor_um: float | None) -> None:
        metrics = result.get("metrics", {})

        # -- 标题 --
        mode_text = "标定（颜色→光程差）" if result.get("mode") == "calibrated" \
            else "相对（颜色级次插值）"
        header = tk.Frame(self, bg=_SURFACE, padx=14, pady=10)
        header.pack(fill=tk.X)
        tk.Label(header, text="薄膜厚度分布结果", bg=_SURFACE, fg=_TEXT,
                 font=(FONT, 14, "bold")).pack(anchor="w")
        sub = [f"模式：{mode_text}"]
        if anchor_um is not None:
            sub.append(f"绝对厚度基准：{anchor_um:.3f} μm")
        if metrics.get("has_reference"):
            sub.append("已扣除无膜基准")
        tk.Label(header, text="   |   ".join(sub), bg=_SURFACE, fg=_MUTED,
                 font=(FONT, 9)).pack(anchor="w", pady=(2, 0))

        # -- 中段：左热力图 + 右详细数据 --
        mid = tk.Frame(self, bg=_BG)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        image_frame = tk.Frame(mid, bg=_SURFACE, padx=8, pady=8)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(image_frame, text="厚度热力图", bg=_SURFACE, fg=_TEXT,
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._image_label = tk.Label(image_frame, bg=_SURFACE)
        self._image_label.pack(fill=tk.BOTH, expand=True)
        self._show_heatmap(result.get("heatmap"))

        data_frame = tk.Frame(mid, bg=_SURFACE, padx=10, pady=8)
        data_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        tk.Label(data_frame, text="详细数据", bg=_SURFACE, fg=_TEXT,
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._data_text = tk.Text(
            data_frame, width=44, height=16, bg="#fafafa", fg=_TEXT,
            font=("Consolas", 10), relief=tk.FLAT, highlightthickness=1,
            highlightbackground=_BORDER, wrap=tk.NONE, state=tk.DISABLED)
        self._data_text.pack(fill=tk.BOTH, expand=True)
        self._show_metrics(metrics, anchor_um)

        # -- 底部：可旋转三维厚度分布 --
        plot_frame = tk.Frame(self, bg=_SURFACE, padx=10, pady=8)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        tk.Label(plot_frame, text="三维厚度分布（按住鼠标左键拖动旋转，滚轮缩放）",
                 bg=_SURFACE, fg=_TEXT, font=(FONT, 10, "bold")).pack(
            anchor="w", pady=(0, 2))
        self._plot_host = tk.Frame(plot_frame, bg=_SURFACE)
        self._plot_host.pack(fill=tk.BOTH, expand=True)
        self._show_3d(result.get("thickness"))

    # ------------------------------------------------------------------
    def _show_heatmap(self, bgr) -> None:
        if bgr is None:
            self._image_label.configure(text="（无热力图）", fg=_MUTED)
            return
        try:
            from PIL import Image, ImageTk
        except Exception:
            self._image_label.configure(text="（PIL 缺失，无法显示图像）", fg=_MUTED)
            return
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        pil = Image.fromarray(rgb)
        max_w = 520
        if pil.width > max_w:
            ratio = max_w / pil.width
            pil = pil.resize((max_w, max(1, int(pil.height * ratio))),
                             Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self._image_label.configure(image=self._photo, text="")

    def _show_metrics(self, metrics: dict, anchor_um: float | None) -> None:
        mode_text = "标定" if metrics.get("mode") == "calibrated" else "相对"
        lines = [
            f"模式            : {mode_text}",
            f"波长            : {metrics.get('wavelength_nm', 0):.1f} nm",
            f"折射率          : {metrics.get('refractive_index', 0):.4f}",
            f"绝对厚度基准    : {anchor_um:.3f} μm" if anchor_um is not None
            else "绝对厚度基准    : 未锚定（相对分布）",
            f"有效像素        : {metrics.get('valid_pixels', 0)}",
            f"均值            : {metrics.get('mean_um', 0):.4f} μm",
            f"中位数          : {metrics.get('median_um', 0):.4f} μm",
            f"稳健最小值(2%)  : {metrics.get('min_robust_um', 0):.4f} μm",
            f"稳健最大值(98%) : {metrics.get('max_robust_um', 0):.4f} μm",
            f"稳健峰谷值 PV   : {metrics.get('pv_robust_um', 0):.4f} μm",
            f"RMS 不均匀度    : {metrics.get('rms_um', 0):.4f} μm",
            f"中间 90% 跨度   : {metrics.get('p90_span_um', 0):.4f} μm",
            f"中位置信度      : {metrics.get('median_confidence', 0):.3f}",
        ]
        self._data_text.configure(state=tk.NORMAL)
        self._data_text.delete("1.0", tk.END)
        self._data_text.insert(tk.END, "\n".join(lines))
        self._data_text.configure(state=tk.DISABLED)

    def _show_3d(self, thickness) -> None:
        _configure_matplotlib_chinese()
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as exc:
            tk.Label(self._plot_host, text=f"无法初始化 3D 绘图：{exc}",
                     bg=_SURFACE, fg=_MUTED).pack(pady=20)
            return

        if thickness is None:
            tk.Label(self._plot_host, text="（无厚度数据，无法绘制 3D 形貌）",
                     bg=_SURFACE, fg=_MUTED).pack(pady=20)
            return
        valid = np.isfinite(thickness)
        if not np.any(valid):
            tk.Label(self._plot_host, text="（无有效厚度像素）",
                     bg=_SURFACE, fg=_MUTED).pack(pady=20)
            return

        fill_value = float(np.nanmedian(thickness[valid]))
        filled = np.where(valid, thickness, fill_value)

        # 下采样到合理尺寸，保证拖动旋转流畅。
        h, w = filled.shape
        max_side = 200
        stride = max(1, int(np.ceil(max(h, w) / max_side)))
        if stride > 1:
            filled = filled[::stride, ::stride]

        h, w = filled.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        fig = Figure(figsize=(5.6, 4.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            xx, yy, filled, cmap="turbo", linewidth=0, antialiased=True,
            rstride=1, cstride=1)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="厚度 (μm)")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_zlabel("厚度 (μm)")
        ax.set_title("三维厚度分布")

        self._canvas = FigureCanvasTkAgg(fig, master=self._plot_host)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
