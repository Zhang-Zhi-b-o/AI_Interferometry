"""Read DirectShow capabilities without starting a second video stream."""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    fps: float
    fourcc: str


def supported_modes(index: int) -> list[CameraMode]:
    if not sys.platform.startswith("win"):
        return []
    import comtypes
    from pygrabber.dshow_graph import FilterGraph

    # Called on the capture worker, so COM must be initialized on that thread.
    comtypes.CoInitialize()
    graph = None
    try:
        graph = FilterGraph()
        graph.add_video_input_device(index)
        formats = graph.get_input_device().get_formats()
        modes = []
        for fmt in formats:
            # pygrabber 0.2 labels the two interval-derived rates in reverse.
            fps = max(fmt["min_framerate"], fmt["max_framerate"])
            fourcc = fmt["media_type_str"]
            if (fmt["width"] > 0 and fmt["height"] > 0 and math.isfinite(fps)
                    and fps > 0 and fourcc in ("MJPG", "YUY2")):
                modes.append(CameraMode(fmt["width"], fmt["height"], round(fps, 3), fourcc))
        return modes
    finally:
        if graph is not None:
            graph.remove_filters()
            graph = None
        comtypes.CoUninitialize()


def preferred_mode(modes: list[CameraMode], target_fps: float, *,
                   max_pixels: int | None = None) -> CameraMode | None:
    """Maximize resolution without dropping below the requested preview rate."""
    if not modes:
        return None
    if max_pixels is not None:
        limited = [mode for mode in modes if mode.width * mode.height <= max_pixels]
        if limited:
            modes = limited
        else:
            smallest = min(mode.width * mode.height for mode in modes)
            modes = [mode for mode in modes if mode.width * mode.height == smallest]
    smooth = [mode for mode in modes if mode.fps + 0.01 >= target_fps]
    if smooth:
        return max(smooth, key=lambda m: (m.width * m.height, m.fps, m.fourcc == "MJPG"))
    return max(modes, key=lambda m: (m.fps, m.width * m.height, m.fourcc == "MJPG"))
