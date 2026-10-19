"""记录与导出：CSV / JSON / Markdown。目录默认 ``exports/``（git 忽略）。"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from app.core.config import ROOT


class RecordService:
    def __init__(self, out_dir: str | Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir else ROOT / "exports"

    # ------------------------------------------------------------------
    def ensure_dir(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir

    def default_path(self, stem: str, suffix: str) -> Path:
        self.ensure_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.out_dir / f"{stem}_{stamp}.{suffix}"

    # ------------------------------------------------------------------
    def export_rows(self, rows: list[dict], path: str | Path | None = None,
                    *, stem: str = "measurements") -> Path:
        target = Path(path) if path else self.default_path(stem, "csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _cell(value) for key, value in row.items()})
        return target

    def export_json(self, payload: dict, path: str | Path | None = None,
                    *, stem: str = "record") -> Path:
        target = Path(path) if path else self.default_path(stem, "json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target

    def export_report(self, *, title: str, sections: list[tuple[str, str]],
                      path: str | Path | None = None, stem: str = "report") -> Path:
        target = Path(path) if path else self.default_path(stem, "md")
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {title}", "", f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for heading, body in sections:
            lines += [f"## {heading}", "", body.strip() or "（无）", ""]
        target.write_text("\n".join(lines), encoding="utf-8")
        return target


def _cell(value) -> object:
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value
