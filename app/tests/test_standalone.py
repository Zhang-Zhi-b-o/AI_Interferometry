"""Regression checks for using app without the original repository."""
import ast
from pathlib import Path

from app.core.config import ROOT, load_config
from app.services.knowledge_corpus import SOURCES, KB_DIR, GUIDE_DIR


def test_resources_are_bundled_inside_app():
    assert ROOT == Path(__import__('app').__file__).resolve().parent
    assert (ROOT / 'config.v2.yaml').is_file()
    assert load_config().model_path.is_file()
    for name in ('PP-OCRv6_rec_small.onnx', 'PP-OCRv6_det_small.onnx',
                 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'):
        assert (ROOT / 'models/micrometer' / name).is_file()
    for _, relative in SOURCES:
        assert (ROOT / relative).is_file(), relative
    assert list(KB_DIR.glob('*.md'))
    assert list(GUIDE_DIR.glob('*.md'))


def test_no_imports_from_original_src_package():
    for path in ROOT.rglob('*.py'):
        if 'tests' in path.parts or 'build' in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != 'src' and not (node.module or '').startswith('src.'), path
            elif isinstance(node, ast.Import):
                assert all(alias.name != 'src' and not alias.name.startswith('src.')
                           for alias in node.names), path
