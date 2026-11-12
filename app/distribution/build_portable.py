"""Build a relocatable Windows release using uv/python-build-standalone.

Run with Python 3.11+: python app/distribution/build_portable.py --output <folder>
Requires uv on PATH. Never operates the attached measurement hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

EXCLUDED = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist",
            "exports", "logs", "data", ".git", "config.yaml", "secrets.yaml"}


def copy_source(source, destination):
    def ignore(directory, names):
        return [name for name in names if name in EXCLUDED or name.endswith((".egg-info", ".pyc", ".pyo", ".tmp"))
                or name.startswith(".env") and name != ".env.example"]
    shutil.copytree(source, destination, ignore=ignore)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--archive-only", action="store_true", help="Repackage a prepared, verified bundle")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to(source):
        raise SystemExit("Choose an output directory outside app/ to avoid copying the build into itself")
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "AI_Interferometry_Windows_x64"
    if args.archive_only:
        if not (bundle / "runtime" / "python.exe").is_file():
            raise SystemExit("Prepared runtime is missing")
        finish_bundle(bundle, output)
        return
    if bundle.exists():
        raise SystemExit(f"Refusing to overwrite an existing bundle: {bundle}")
    bundle.mkdir()
    managed = output / "managed-python"
    subprocess.run(["uv", "python", "install", args.python, "--install-dir", str(managed), "--no-bin"], check=True)
    candidates = sorted(managed.glob("cpython-*-windows-x86_64-none/python.exe"))
    candidates = [p for p in candidates if re.fullmatch(r"cpython-\d+\.\d+\.\d+-windows-x86_64-none", p.parent.name)]
    if not candidates:
        raise SystemExit("Managed Windows Python was not found")
    runtime = bundle / "runtime"
    shutil.copytree(candidates[-1].parent, runtime)
    copy_source(source, bundle / "app")
    dependencies = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    requirements = output / "dependencies.txt"
    lock = source / "distribution" / "requirements-windows-cpu.lock"
    if lock.is_file():
        # Drop entries pinned to the release machine's own wheel paths (pip/setuptools
        # shipped with the CI runtime); they cannot be downloaded anywhere else.
        lines = lock.read_text(encoding="utf-8").splitlines()
        lines = [line for line in lines if not re.match(r"(pip|setuptools)\s*@\s*file://", line)]
        requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        requirements.write_text("\n".join(dependencies) + "\n", encoding="utf-8")
    python = runtime / "python.exe"
    # This is our private runtime copy, not the user's managed/system interpreter.
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--break-system-packages", "--torch-backend", "cpu",
                    "-r", str(requirements)], check=True)
    finish_bundle(bundle, output)


def finish_bundle(bundle, output):
    python = bundle / "runtime" / "python.exe"
    frozen = subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True)
    (bundle / "DEPENDENCIES.txt").write_text(frozen, encoding="utf-8")
    for name, args_text in (("start.bat", ""), ("demo.bat", "--demo")):
        (bundle / name).write_bytes((f'@echo off\r\ncd /d "%~dp0"\r\ncall "app\\start.bat" {args_text}\r\n').encode("ascii"))
    (bundle / "remove-micrometer-filter.bat").write_bytes(
        b'@echo off\r\ncall "%~dp0app\\remove-micrometer-filter.bat"\r\n')
    (bundle / "README.txt").write_text(
        "白光干涉测量工作台 · Windows 10/11 x64 便携版\n\n"
        "先完整解压到可写目录，然后双击 start.bat；无需安装 Python 或 CUDA。\n"
        "没有设备时双击 demo.bat。\n"
        "实验助手：进入⑥实验助手 → API设置，填写自己的API Key。\n"
        "微分表：进入④读数 → 开始信号读取与拦截，首次按提示允许管理员配置。\n"
        "驱动可能要求重启 Windows；仅支持 C216:010B 信号设备，其他仪器可用相机OCR。\n"
        "详细连接、卸载和问题排查见 app/README.md。\n"
        "各第三方依赖许可证保留在 runtime/Lib/site-packages 的 dist-info 目录中。\n",
        encoding="utf-8-sig")
    records = []
    for path in sorted((bundle / "app").rglob("*")):
        if path.is_file():
            records.append({"path": path.relative_to(bundle).as_posix(), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (bundle / "MANIFEST.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output / "AI_Interferometry_Windows_x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(bundle.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zipped.write(path, path.relative_to(output))
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(archive)


if __name__ == "__main__":
    main()
