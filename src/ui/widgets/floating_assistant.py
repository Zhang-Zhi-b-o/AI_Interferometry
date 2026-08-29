"""页面内可移动、缩放和收起的实验助手容器。"""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk

from src.ui.theme import FONT, NAVY, PRIMARY, SURFACE


@dataclass(frozen=True)
class FloatingGeometry:
    x: int
    y: int
    width: int
    height: int


def clamp_floating_geometry(
    geometry: FloatingGeometry,
    parent_width: int,
    parent_height: int,
    *,
    min_width: int = 360,
    min_height: int = 300,
) -> FloatingGeometry:
    """把浮窗尺寸和坐标限制在父页面可见范围内。"""
    available_width = max(1, int(parent_width))
    available_height = max(1, int(parent_height))
    effective_min_width = min(int(min_width), available_width)
    effective_min_height = min(int(min_height), available_height)
    width = max(effective_min_width, min(int(geometry.width), available_width))
    height = max(effective_min_height, min(int(geometry.height), available_height))
    x = max(0, min(int(geometry.x), available_width - width))
    y = max(0, min(int(geometry.y), available_height - height))
    return FloatingGeometry(x, y, width, height)


class FloatingAssistantWindow(tk.Frame):
    """嵌在工作页面内的非模态浮动窗口。"""

    TITLE_HEIGHT = 52
    MIN_WIDTH = 380
    MIN_HEIGHT = 480

    def __init__(
        self,
        parent: tk.Widget,
        *,
        width: int = 560,
        height: int = 760,
    ) -> None:
        super().__init__(
            parent,
            bg=SURFACE,
            highlightthickness=2,
            highlightbackground="#9fb3c8",
            bd=0,
        )
        self._geometry = FloatingGeometry(24, 24, width, height)
        self._expanded_height = height
        self._collapsed = False
        self._visible = False
        self._initial_position_pending = True
        self._geometry_job: str | None = None
        self._drag_origin: tuple[int, int, int, int] | None = None
        self._resize_origin: tuple[int, int, int, int] | None = None

        self.title_bar = tk.Frame(self, bg=NAVY, height=self.TITLE_HEIGHT, cursor="fleur")
        self.title_bar.pack(fill=tk.X)
        self.title_bar.pack_propagate(False)

        brand = tk.Label(
            self.title_bar, text="AI", bg=PRIMARY, fg="#ffffff",
            font=("Segoe UI", 11, "bold"), width=3,
        )
        brand.pack(side=tk.LEFT, padx=(10, 9), pady=8, ipady=4)
        title_group = tk.Frame(self.title_bar, bg=NAVY, cursor="fleur")
        title_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=6)
        title = tk.Label(
            title_group, text="统一实验助手", bg=NAVY, fg="#ffffff",
            font=(FONT, 11, "bold"), anchor="w", cursor="fleur",
        )
        title.pack(fill=tk.X)
        subtitle = tk.Label(
            title_group, text="条纹诊断 · 四阶段调节 · 数据与报告", bg=NAVY,
            fg="#b8cee8", font=(FONT, 8), anchor="w", cursor="fleur",
        )
        subtitle.pack(fill=tk.X)

        self.collapse_button = self._title_button("收回", self.toggle_collapsed, width=4)
        self.collapse_button.pack(side=tk.RIGHT, padx=(2, 8), pady=8)
        self._title_button("＋", lambda: self.resize_by(72, 64)).pack(
            side=tk.RIGHT, padx=2, pady=8)
        self._title_button("－", lambda: self.resize_by(-72, -64)).pack(
            side=tk.RIGHT, padx=2, pady=8)

        self.content = tk.Frame(self, bg="#f4f7fb")
        self.content.pack(fill=tk.BOTH, expand=True)

        self.resize_grip = tk.Label(
            self, text="◢", bg=SURFACE, fg="#7890ad", cursor="size_nw_se",
            font=("Segoe UI Symbol", 12), padx=1, pady=0,
        )
        self.resize_grip.place(relx=1.0, rely=1.0, anchor="se")

        for widget in (self.title_bar, title_group, title, subtitle, brand):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<ButtonRelease-1>", self._finish_pointer_action)
        self.resize_grip.bind("<ButtonPress-1>", self._start_resize)
        self.resize_grip.bind("<B1-Motion>", self._resize)
        self.resize_grip.bind("<ButtonRelease-1>", self._finish_pointer_action)
        self.bind("<Button-1>", lambda _event: self.lift(), add="+")
        parent.bind("<Configure>", self._on_parent_configure, add="+")

    def _title_button(self, text: str, command, *, width: int = 2) -> tk.Button:
        return tk.Button(
            self.title_bar, text=text, command=command, width=width,
            relief=tk.FLAT, bd=0, bg="#264766", fg="#ffffff",
            activebackground="#345d82", activeforeground="#ffffff",
            cursor="hand2", font=(FONT, 9, "bold"), pady=4,
        )

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    @property
    def floating_geometry(self) -> FloatingGeometry:
        return self._geometry

    def show(self, *, expand: bool = False) -> None:
        self._visible = True
        if expand and self._collapsed:
            self.set_collapsed(False)
        self._apply_geometry()
        self.lift()

    def hide(self) -> None:
        self._visible = False
        self.place_forget()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        if collapsed:
            self._expanded_height = max(self.MIN_HEIGHT, self._geometry.height)
            self.content.pack_forget()
            self.resize_grip.place_forget()
            self.collapse_button.configure(text="展开")
            self._geometry = FloatingGeometry(
                self._geometry.x, self._geometry.y,
                self._geometry.width, self.TITLE_HEIGHT,
            )
        else:
            self.content.pack(fill=tk.BOTH, expand=True)
            self.resize_grip.place(relx=1.0, rely=1.0, anchor="se")
            self.collapse_button.configure(text="收回")
            self._geometry = FloatingGeometry(
                self._geometry.x, self._geometry.y,
                self._geometry.width, self._expanded_height,
            )
        self._apply_geometry()

    def resize_by(self, width_delta: int, height_delta: int) -> None:
        if self._collapsed:
            self.set_collapsed(False)
        self._geometry = FloatingGeometry(
            self._geometry.x, self._geometry.y,
            self._geometry.width + int(width_delta),
            self._geometry.height + int(height_delta),
        )
        self._apply_geometry()

    def _start_drag(self, event) -> None:
        self.lift()
        self._drag_origin = (
            event.x_root, event.y_root, self._geometry.x, self._geometry.y)

    def _drag(self, event) -> None:
        if self._drag_origin is None:
            return
        root_x, root_y, start_x, start_y = self._drag_origin
        self._geometry = FloatingGeometry(
            start_x + event.x_root - root_x,
            start_y + event.y_root - root_y,
            self._geometry.width,
            self._geometry.height,
        )
        self._queue_geometry_update()

    def _start_resize(self, event) -> None:
        self.lift()
        self._resize_origin = (
            event.x_root, event.y_root,
            self._geometry.width, self._geometry.height)

    def _resize(self, event) -> None:
        if self._resize_origin is None or self._collapsed:
            return
        root_x, root_y, start_width, start_height = self._resize_origin
        self._geometry = FloatingGeometry(
            self._geometry.x,
            self._geometry.y,
            start_width + event.x_root - root_x,
            start_height + event.y_root - root_y,
        )
        self._queue_geometry_update()

    def _finish_pointer_action(self, _event=None) -> None:
        self._drag_origin = None
        self._resize_origin = None
        if self._geometry_job is not None:
            self.after_cancel(self._geometry_job)
            self._geometry_job = None
        self._apply_geometry()

    def _on_parent_configure(self, _event=None) -> None:
        if self._visible:
            self._queue_geometry_update()

    def _queue_geometry_update(self) -> None:
        """合并高频拖动事件，避免复杂文本区重复重排产生残影。"""
        if self._geometry_job is None:
            self._geometry_job = self.after(16, self._flush_geometry_update)

    def _flush_geometry_update(self) -> None:
        self._geometry_job = None
        self._apply_geometry()

    def _apply_geometry(self) -> None:
        self.master.update_idletasks()
        parent_width = max(1, self.master.winfo_width())
        parent_height = max(1, self.master.winfo_height())
        if parent_width < 100 or parent_height < 100:
            if self._visible:
                self.after(20, self._apply_geometry)
            return
        visible_width = max(
            1, min(parent_width,
                   self.winfo_screenwidth() - self.master.winfo_rootx()))
        visible_height = max(
            1, min(parent_height,
                   self.winfo_screenheight() - self.master.winfo_rooty()))
        min_height = self.TITLE_HEIGHT if self._collapsed else self.MIN_HEIGHT
        geometry = clamp_floating_geometry(
            self._geometry, visible_width, visible_height,
            min_width=self.MIN_WIDTH, min_height=min_height,
        )
        if self._initial_position_pending:
            preferred_x = int(visible_width * 0.36)
            geometry = FloatingGeometry(
                max(0, min(preferred_x, visible_width - geometry.width - 22)),
                22,
                geometry.width,
                geometry.height,
            )
            self._initial_position_pending = False
        self._geometry = geometry
        if self._visible:
            self.place(
                x=geometry.x, y=geometry.y,
                width=geometry.width, height=geometry.height,
            )
            # Windows 下 ScrolledText 在连续 place 缩放时可能延迟擦除旧区域；
            # 在当前几何帧结束前完成父页面与浮窗重绘。
            self.master.update_idletasks()
            self.update_idletasks()
            self.lift()
