"""YOLO 类别名称映射"""
ZERO_KEYWORDS = ("black", "zero", "order", "dark", "黑", "零级")
FRINGE_KEYWORDS = ("color", "near", "far", "fringe", "彩", "条纹")

# 反向映射（用于自动控制判断）
def get_class_confidences(result: dict) -> dict[str, float]:
    """从检测结果提取各类别最高置信度，映射为友好名称"""
    if len(result["class_names"]) == 0:
        return {"color": 0.0, "black": 0.0}
    out = {"color": 0.0, "black": 0.0}
    for name, conf in zip(result["class_names"], result["confs"]):
        normalized = str(name).lower()
        # zero_order 同时含有 order，必须优先映射为黑条/零级角色。
        if any(keyword in normalized for keyword in ZERO_KEYWORDS):
            mapped = "black"
        elif any(keyword in normalized for keyword in FRINGE_KEYWORDS):
            mapped = "color"
        else:
            continue
        out[mapped] = max(out[mapped], float(conf))
    return out


def get_non_center_guide(result: dict, frame_width: float,
                         previous_x: float | None = None) -> dict:
    """选择最适合引导搜索的非中心条纹框。

    优先使用 ``near`` 类别；同类多个框时选择最靠近画面水平中心的框。
    零级/黑色中心框始终排除。
    """
    candidates = []
    boxes = result.get("boxes_xyxy", [])
    names = result.get("class_names", [])
    confs = result.get("confs", [])
    for box, name, conf in zip(boxes, names, confs):
        normalized = str(name).lower()
        if any(keyword in normalized for keyword in ZERO_KEYWORDS):
            continue
        if not any(keyword in normalized for keyword in FRINGE_KEYWORDS):
            continue
        try:
            center_x = (float(box[0]) + float(box[2])) / 2.0
            confidence = float(conf)
        except (TypeError, ValueError, IndexError):
            continue
        candidates.append({
            "x": center_x,
            "confidence": confidence,
            "class_name": str(name),
            "near": "near" in normalized or "近" in normalized,
        })

    if not candidates:
        return {"x": None, "confidence": 0.0, "count": 0, "class_name": ""}
    preferred = [item for item in candidates if item["near"]] or candidates
    target = float(frame_width) / 2.0
    # 已经锁定路标后优先选择与上一帧位置连续的框，避免多个 near_fringe
    # 框之间跳换；跳变过大时才回退到最靠近画面中心的框。
    if previous_x is not None:
        continuous = min(preferred, key=lambda item: abs(item["x"] - previous_x))
        max_jump = max(80.0, float(frame_width) * 0.22)
        selected = (
            continuous if abs(continuous["x"] - previous_x) <= max_jump
            else min(preferred, key=lambda item: abs(item["x"] - target))
        )
    else:
        selected = min(preferred, key=lambda item: abs(item["x"] - target))
    return {
        "x": selected["x"],
        "confidence": selected["confidence"],
        "count": len(candidates),
        "class_name": selected["class_name"],
    }
