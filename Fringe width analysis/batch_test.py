"""跨目录抽样测试条纹识别鲁棒性。

从 ``pre_annotation_results`` 下所有采集会话各抽一批图片，用
``analyze_fringes.analyze_image`` 识别，汇总识别率、条纹数量/周期分布、
颜色序列，并列出识别失败（0 亮纹 / 读图失败）的样本，用于判断算法
对各种条纹（数量、宽度、颜色不同）的覆盖能力。

运行：``python batch_test.py [pre_annotation_results根目录] [每会话抽样数]``
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import analyze_fringes as af

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(
    r"D:\Files\Work\光电\AI_Interferometry_prepare\make_dataset_new"
    r"\pre_annotation_results")


def sample_images(session: Path, k: int) -> list[Path]:
    """从一个会话目录里均匀抽样 k 张图（按序号间隔取，覆盖前中后）。"""
    images = sorted((session / "images").glob("*.jpg"))
    if not images:
        return []
    if k >= len(images):
        return images
    step = len(images) / k
    idx = {int(i * step) for i in range(k)}
    return [images[i] for i in sorted(idx)]


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv and len(argv) > 0 else ROOT
    per = int(argv[1]) if argv and len(argv) > 1 else 12

    sessions = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "images").is_dir()
    )

    total = success = 0
    fail_reasons: Counter = Counter()
    bright_counts: Counter = Counter()
    dark_counts: Counter = Counter()
    periods: list[float] = []
    colour_seqs: Counter = Counter()  # 亮纹颜色序列 -> 出现次数
    failed_samples: list[tuple[str, str]] = []

    print(f"{'会话':<32} {'抽样':>4} {'成功':>4}  亮纹数分布")
    for sess in sessions:
        samples = sample_images(sess, per)
        ok = 0
        per_session_bright: Counter = Counter()
        for p in samples:
            total += 1
            try:
                rep = af.analyze_image(p)
            except Exception as exc:  # 读图/处理异常
                fail_reasons[f"异常:{type(exc).__name__}"] += 1
                failed_samples.append((p.parent.name, f"{p.name} {exc}"))
                continue
            nb = rep["num_bright"]
            nd = rep["num_dark"]
            if nb == 0:
                fail_reasons["无亮纹"] += 1
                failed_samples.append((p.parent.name, f"{p.name} 无亮纹"))
                continue
            success += 1
            ok += 1
            bright_counts[nb] += 1
            dark_counts[nd] += 1
            periods.append(rep["period_px"])
            per_session_bright[nb] += 1
            # 亮纹颜色序列（相邻同色去重，只看颜色排布规律）
            seq = tuple(
                b["color_name"] for b in rep["bands"] if b["kind"] == "bright")
            colour_seqs[seq] += 1
        dist = " ".join(f"{n}×{c}" for n, c in sorted(per_session_bright.items()))
        print(f"{sess.name:<32} {len(samples):>4} {ok:>4}  {dist}")

    print(f"\n===== 汇总 =====")
    print(f"抽测 {total} 张，识别成功（≥1 亮纹）{success} 张，"
          f"成功率 {success / max(total, 1) * 100:.1f}%")
    print(f"\n亮纹数量分布: "
          f"{dict(sorted(bright_counts.items()))}")
    print(f"暗纹数量分布: {dict(sorted(dark_counts.items()))}")
    if periods:
        import statistics
        print(f"周期分布: min={min(periods):.0f} 中位={statistics.median(periods):.0f} "
              f"max={max(periods):.0f} px")
    print(f"\n识别失败样本数: {sum(fail_reasons.values())}  "
          f"({dict(fail_reasons)})")
    print("\n最常见的亮纹颜色序列 (前 6 种):")
    for seq, cnt in colour_seqs.most_common(6):
        print(f"  {cnt:>3} 次  {' → '.join(seq)}")

    if failed_samples:
        print(f"\n失败样本明细 (前 20):")
        for sess, msg in failed_samples[:20]:
            print(f"  [{sess}] {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
