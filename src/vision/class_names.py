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
