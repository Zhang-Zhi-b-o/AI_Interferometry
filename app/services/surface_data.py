"""用于显示的有限三角网格；不填补无效厚度，不改变原测量矩阵。"""
from __future__ import annotations

import numpy as np


def surface_mesh(thickness, mask=None, *, max_side=160):
    data = np.asarray(thickness, dtype=float)
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("3D 形貌需要至少 2×2 的厚度矩阵")
    if max_side < 2:
        raise ValueError("网格边长上限至少为 2")
    valid = np.isfinite(data)
    if mask is not None:
        if np.shape(mask) != data.shape:
            raise ValueError("厚度矩阵和有效区域尺寸不一致")
        valid &= np.asarray(mask, dtype=bool)
    stride = max(1, int(np.ceil((max(data.shape) - 1) / (max_side - 1))))
    rows = np.unique(np.r_[np.arange(0, data.shape[0], stride), data.shape[0] - 1])
    cols = np.unique(np.r_[np.arange(0, data.shape[1], stride), data.shape[1] - 1])
    # 降采样块内只要有无效像素就舍弃，避免把小孔洞填成连续曲面。
    sampled_valid = np.logical_and.reduceat(
        np.logical_and.reduceat(valid, rows, axis=0), cols, axis=1)
    z = data[np.ix_(rows, cols)]
    x, y = np.meshgrid(cols, rows)
    indices = np.arange(z.size).reshape(z.shape)
    a, b = indices[:-1, :-1], indices[:-1, 1:]
    c, d = indices[1:, :-1], indices[1:, 1:]
    cells = (sampled_valid[:-1, :-1] & sampled_valid[:-1, 1:]
             & sampled_valid[1:, :-1] & sampled_valid[1:, 1:])
    triangles = np.concatenate((np.stack((a[cells], b[cells], c[cells]), axis=1),
                                np.stack((b[cells], d[cells], c[cells]), axis=1)))
    if not len(triangles):
        raise ValueError("有效区域不足以形成连续 3D 曲面")
    keep = sampled_valid.ravel()
    mapping = np.cumsum(keep) - 1
    return x.ravel()[keep], y.ravel()[keep], z.ravel()[keep], mapping[triangles]
