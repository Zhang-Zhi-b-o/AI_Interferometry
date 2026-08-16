"""用 app 的 2D 轮廓算法分析弯曲条纹并生成标注图。

用法: python annotate_curved.py <输入.png> [输出.png]
"""
import sys

import cv2
import numpy as np

from src.vision.fringe_width import measure_center_fringe_width_2d


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "376f4856e9630cf43845911ec78d5dc1.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".png", "_annotated.png")

    bgr = cv2.imread(src)
    if bgr is None:
        raise SystemExit(f"无法读取图片: {src}")

    result = measure_center_fringe_width_2d(bgr)
    print(f"period_px = {result['period_px']}")
    print(f"num_bands  = {result['num_bands']} "
          f"(bright={result['num_bright']}, dark={result['num_dark']})")
    for i, b in enumerate(result["bands"]):
        pts = len(b.get("centerline", []))
        star = "★" if b is result["center_band"] else " "
        print(f"  [{i:2d}]{star} {b['kind']:6s} "
              f"center={b['center_x']:7.2f} width={b['width']:6.2f} "
              f"pts={pts}")

    out = bgr.copy()
    h, w = out.shape[:2]
    for b in result["bands"]:
        color = (74, 210, 255) if b["kind"] == "bright" else (255, 208, 154)
        line = b.get("centerline")
        if line and len(line) >= 2:
            pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2,
                          lineType=cv2.LINE_AA)
        # 宽度标注：标在每条条纹顶部（中心位置）
        cx = int(round(b["center_x"]))
        cv2.putText(out, f"{b['width']:.1f}", (cx - 16, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(dst, out)
    print(f"已保存: {dst}")


if __name__ == "__main__":
    main()
