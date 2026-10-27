"""检测标注样式；Qt 预览、单帧图片及录像共享类别配色。"""
from __future__ import annotations

import math


STYLES = {
    "zero_order": ("零级条纹", "#22c55e"),
    "near_fringe": ("近场条纹", "#3b82f6"),
    "far_fringe": ("远场条纹", "#f59e0b"),
}


def detection_style(name, confidence=None, *, chinese=False, compact=False):
    name = str(name or "unknown")
    title, color = STYLES.get(name, (name, "#c084fc"))
    text = f"{title} ({name})" if chinese and title != name else name
    if compact:
        titles = ({"zero_order": "中心", "near_fringe": "非中心", "far_fringe": "远场"}
                  if chinese else {"zero_order": "C", "near_fringe": "N", "far_fringe": "F"})
        text = titles.get(name, name)
    if confidence is not None and math.isfinite(float(confidence)):
        text += f" {float(confidence):.2f}"
    return text, color


def detection_order(boxes, names):
    """Draw the center last, independently of model confidence/order."""
    return sorted(range(len(boxes)), key=lambda i: i < len(names) and names[i] == "zero_order")


def draw_detections(frame, boxes, names=(), confs=()):
    """在 BGR 图像上绘制模型原始类别名称，避免 OpenCV 中文字体缺字。"""
    import cv2

    height, width = frame.shape[:2]
    for index in detection_order(boxes, names):
        box = boxes[index]
        x1, y1, x2, y2 = map(int, box[:4])
        name = names[index] if index < len(names) else "unknown"
        confidence = confs[index] if index < len(confs) else None
        text, color = detection_style(name, confidence, compact=True)
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        bgr = rgb[::-1]
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)
        tx = max(0, min(x1, width-tw-8))
        ty = max(th+baseline+6, min(y1, height-1))
        cv2.rectangle(frame, (tx, ty-th-baseline-6), (min(width-1, tx+tw+8), ty), bgr, -1)
        cv2.putText(frame, text, (tx+4, ty-baseline-3), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (10, 15, 20), 1, cv2.LINE_AA)
    return frame
