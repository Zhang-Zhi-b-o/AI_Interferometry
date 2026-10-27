"""任意方向直条纹的全局方向预估与刚性坐标变换。

现有二维中心线追踪器沿图像竖直方向逐带搜索，适合近竖直条纹。激光预调阶段
可能出现接近水平、对比度较低并带有显示辅助线的直条纹。本模块先在缩小图上
搜索“把条纹旋到竖直时，逐行平均后的周期调制度最大”的角度，再由原中心线
算法完成轮廓和间距测量。

``tilt_deg`` 沿用项目约定：相对竖直方向，正号为 ``\\`` 形（顺时针倾斜），
范围为 ``[-90, 90)``；``correction_deg = -tilt_deg``。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage, signal


def _gray_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.shape[2] == 1:
        return array[:, :, 0].astype(np.float32)
    bgr = array[:, :, :3].astype(np.float32)
    return 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]


def suppress_cyan_overlay(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """去除实拍画面中的细青色中心辅助线，返回处理图和遮罩占比。

    仅在彩色图像中处理高饱和青色像素；若青色覆盖超过画面的 5%，视为真实
    场景颜色而不处理，避免误删大面积白光彩色条纹。
    """
    image = np.asarray(bgr)
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        return image, 0.0
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (75, 90, 70), (105, 255, 255))
    fraction = float(np.count_nonzero(mask) / mask.size)
    if fraction <= 0.0 or fraction > 0.05:
        return image, 0.0
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8))
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA), fraction


def rotation_matrix_same_size(
    shape: tuple[int, ...], angle_deg: float,
) -> np.ndarray:
    """返回绕图像中心旋转、输出尺寸不变的 2×3 仿射矩阵。"""
    height, width = shape[:2]
    return cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), float(angle_deg), 1.0)


def rotate_same_size(
    image: np.ndarray, angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """旋转图像但保持尺寸，以反射边界避免黑三角被误认成条纹。"""
    height, width = image.shape[:2]
    matrix = rotation_matrix_same_size(image.shape, angle_deg)
    rotated = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated, matrix


def transform_points(
    points: list[list[float]] | list[tuple[float, float]],
    matrix: np.ndarray,
) -> list[list[float]]:
    """用 2×3 仿射矩阵变换二维点列。"""
    if not points:
        return []
    array = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((array[:, :2], np.ones(len(array))))
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return [[float(x), float(y)] for x, y in mapped]


def fit_line_orientation(
    points: list[list[float]] | list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """用正交 PCA 拟合任意方向中心线。

    返回 ``(tilt_deg, residual_px, length_px)``。与 ``x(y)`` 回归不同，正交
    拟合在接近水平时没有奇点；``length_px`` 是沿主方向的投影跨度。
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        return None
    centered = array[:, :2] - np.mean(array[:, :2], axis=0)
    if float(np.max(np.ptp(array[:, :2], axis=0))) < 1e-6:
        return None
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0]
    # 主向量统一朝图像下方；水平极限时朝右，避免 ±90°随机翻转。
    if direction[1] < -1e-9 or (
        abs(direction[1]) <= 1e-9 and direction[0] < 0
    ):
        direction = -direction
    normal = np.asarray([direction[1], -direction[0]])
    along = centered @ direction
    across = centered @ normal
    length = float(np.ptp(along))
    if length < 1e-6:
        return None
    residual = float(np.sqrt(np.mean(across ** 2)))
    tilt = float(np.degrees(np.arctan2(direction[0], direction[1])))
    if tilt >= 90.0:
        tilt -= 180.0
    if tilt < -90.0:
        tilt += 180.0
    return tilt, residual, length


def _projection_signal(
    gray: np.ndarray, tilt_deg: float,
) -> tuple[np.ndarray, float]:
    """返回角度校正后的法向高通剖面及二维局部起伏尺度。"""
    corrected, _ = rotate_same_size(gray, -float(tilt_deg))
    height, width = corrected.shape
    y0, y1 = int(0.16 * height), int(0.84 * height)
    x0, x1 = int(0.12 * width), int(0.88 * width)
    region = corrected[y0:y1, x0:x1].astype(np.float64)
    if min(region.shape) < 8:
        return np.asarray([], dtype=np.float64), 1.0
    profile = np.mean(region, axis=0)
    sigma = max(4.0, len(profile) / 14.0)
    signal = profile - ndimage.gaussian_filter1d(profile, sigma=sigma)
    # 除以二维局部起伏：随机散斑在逐行平均后应被压低，连续直条纹则保留。
    local = region - ndimage.gaussian_filter(region, sigma=(3.0, 3.0))
    noise_scale = max(float(np.std(local)), 1e-6)
    return signal, noise_scale


def _projection_score(gray: np.ndarray, tilt_deg: float) -> float:
    """条纹按给定倾角校正后，计算法向一维剖面的周期调制度。"""
    profile, noise_scale = _projection_signal(gray, tilt_deg)
    return float(np.std(profile) / noise_scale) if len(profile) else 0.0


def estimate_global_fringe_orientation(
    bgr: np.ndarray,
    *,
    max_dimension: int = 360,
) -> dict:
    """估计任意方向激光直条纹相对竖直方向的倾角。

    返回值可 JSON 化。``confidence`` 是方向峰突出程度形成的算法质量分数，
    不是概率或测量不确定度；无足够亮度起伏时 ``tilt_deg`` 为 ``None``。
    """
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("无有效画面")
    if bgr.ndim not in (2, 3):
        raise ValueError(f"不支持的图像维度: {bgr.ndim}")
    if min(bgr.shape[:2]) < 16:
        raise ValueError("图像过小，无法估计条纹方向")

    cleaned, overlay_fraction = suppress_cyan_overlay(bgr)
    gray = _gray_float(cleaned)
    dynamic = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    empty = {
        "tilt_deg": None,
        "correction_deg": None,
        "confidence": 0.0,
        "method": "projection_search",
        "contrast_range": round(dynamic, 3),
        "overlay_fraction": round(overlay_fraction, 6),
    }
    if dynamic < 1.0:
        return empty

    scale = min(1.0, float(max_dimension) / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = ndimage.gaussian_filter(gray.astype(np.float32), sigma=1.0)

    coarse_angles = np.arange(-88.0, 88.01, 4.0)
    coarse_scores = np.asarray(
        [_projection_score(gray, angle) for angle in coarse_angles],
        dtype=np.float64,
    )
    coarse_index = int(np.argmax(coarse_scores))
    coarse_best = float(coarse_angles[coarse_index])
    fine_angles = np.arange(coarse_best - 4.0, coarse_best + 4.001, 0.25)
    fine_scores = np.asarray(
        [_projection_score(gray, angle) for angle in fine_angles],
        dtype=np.float64,
    )
    best_index = int(np.argmax(fine_scores))
    tilt = float(fine_angles[best_index])
    while tilt >= 90.0:
        tilt -= 180.0
    while tilt < -90.0:
        tilt += 180.0

    peak = float(fine_scores[best_index])
    baseline = float(np.percentile(coarse_scores, 55))
    prominence = max(0.0, (peak - baseline) / max(peak, 1e-9))
    contrast_factor = float(np.clip(dynamic / 18.0, 0.0, 1.0))
    profile, _ = _projection_signal(gray, tilt)
    profile_range = float(
        np.percentile(profile, 95) - np.percentile(profile, 5)
        if len(profile) else 0.0)
    peak_distance = max(3, len(profile) // 40)
    profile_peaks, _ = signal.find_peaks(
        profile, distance=peak_distance,
        prominence=max(0.25, 0.12 * profile_range),
    )
    periodicity = 0.0
    period_px = None
    if len(profile) >= 16:
        centered = profile - np.mean(profile)
        autocorrelation = signal.correlate(
            centered, centered, mode="full", method="fft")[len(profile) - 1:]
        if autocorrelation[0] > 1e-9:
            autocorrelation /= autocorrelation[0]
            min_lag = max(3, len(profile) // 80)
            max_lag = max(min_lag + 1, len(profile) // 2)
            corr_peaks, properties = signal.find_peaks(
                autocorrelation[min_lag:max_lag], prominence=0.04)
            if len(corr_peaks):
                best_corr = int(np.argmax(properties["prominences"]))
                lag = int(corr_peaks[best_corr] + min_lag)
                periodicity = max(0.0, float(autocorrelation[lag]))
                period_px = float(lag / scale)

    repeat_factor = float(np.clip((len(profile_peaks) - 1) / 2.0, 0.0, 1.0))
    periodicity_factor = float(np.clip(periodicity / 0.35, 0.0, 1.0))
    confidence = float(np.clip(
        1.8 * prominence * contrast_factor
        * repeat_factor * periodicity_factor,
        0.0, 1.0,
    ))
    if confidence < 0.12:
        return empty
    return {
        **empty,
        "tilt_deg": round(tilt, 3),
        "correction_deg": round(-tilt, 3),
        "confidence": round(confidence, 3),
        "projection_score": round(peak, 4),
        "period_px": round(period_px, 2) if period_px is not None else None,
        "periodicity": round(periodicity, 3),
        "num_profile_peaks": int(len(profile_peaks)),
    }
