"""YOLO 类别名称映射"""
CLASS_NAMES = {
    0: "color",
    1: "black",
}

# 反向映射（用于自动控制判断）
def get_class_confidences(result: dict) -> dict[str, float]:
    """从检测结果提取各类别最高置信度，映射为友好名称"""
    if len(result["class_names"]) == 0:
        return {"color": 0.0, "black": 0.0}
    out = {"color": 0.0, "black": 0.0}
    for name, conf in zip(result["class_names"], result["confs"]):
        mapped = CLASS_NAMES.get(int(name) if isinstance(name, (int, float)) else -1, str(name))
        if mapped not in out or conf > out[mapped]:
            out[mapped] = float(conf)
    return out
