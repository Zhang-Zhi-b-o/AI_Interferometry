"""3D 网格、掩膜与像素坐标回归；不验证物理测量准确度。"""
import numpy as np
import pytest

from app.services.surface_data import surface_mesh


def test_surface_keeps_original_coordinates_and_values():
    data = np.arange(800 * 600, dtype=float).reshape(800, 600)
    original = data.copy()
    x, y, z, triangles = surface_mesh(data)
    assert max(x) == 599 and max(y) == 799
    assert len(z) <= 160 * 160
    np.testing.assert_array_equal(z, data[y, x])
    np.testing.assert_array_equal(data, original)
    assert triangles.shape[1] == 3
    assert triangles.max() < len(z)


def test_map_png_export_in_chinese_directory(tmp_path):
    import cv2

    from app.services.thickness_map import ThicknessMapService

    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    path = ThicknessMapService.save({"overlay": image}, tmp_path / "中文结果")
    restored = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(restored, image)
    with pytest.raises(ValueError):
        ThicknessMapService.save({}, tmp_path)


def test_surface_never_fills_mask_or_nan_holes():
    data = np.zeros((5, 5))
    data[2, 2] = np.nan
    mask = np.ones_like(data, dtype=bool)
    mask[0, 0] = False
    x, y, z, triangles = surface_mesh(data, mask)
    assert np.isfinite(z).all()
    assert not ((x == 2) & (y == 2)).any()
    assert not ((x == 0) & (y == 0)).any()
    # 任何包含孔洞角点的单元都不绘制三角面。
    for indices in triangles:
        assert not (x[indices].min() <= 2 <= x[indices].max()
                    and y[indices].min() <= 2 <= y[indices].max())


@pytest.mark.parametrize("data,mask", [
    (np.ones(3), None), (np.ones((1, 3)), None),
    (np.full((3, 3), np.nan), None),
    (np.ones((3, 3)), np.ones((4, 4))),
])
def test_surface_rejects_unusable_matrix(data, mask):
    with pytest.raises(ValueError):
        surface_mesh(data, mask)
