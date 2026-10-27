'视觉检测模块：YOLO 检测、画面校正、条纹识别与中心定位。'
from importlib import import_module

_EXPORTS = {
    'YOLODetector': 'app.engine.vision.detector',
    'rotate_expand': 'app.engine.vision.angle',
    'FrameCorrector': 'app.engine.vision.correct',
    'CenterTracker': 'app.engine.vision.fringe_center',
    'find_center_by_band': 'app.engine.vision.fringe_center',
    'find_center_in_region': 'app.engine.vision.fringe_center',
    'estimate_fringe_angle_2d': 'app.engine.vision.fringe_angle',
    'estimate_global_fringe_orientation': 'app.engine.vision.fringe_orientation',
    'FringeMotionTracker': 'app.engine.vision.fringe_motion',
    'FringeRecognitionTracker': 'app.engine.vision.fringe_recognition',
    'analyse_fringe_texture': 'app.engine.vision.fringe_recognition',
    'analyse_guidance_geometry': 'app.engine.vision.fringe_guidance',
    'build_fringe_guidance': 'app.engine.vision.fringe_guidance',
    'laser_guidance_signature': 'app.engine.vision.fringe_guidance',
    'render_laser_alignment_instruction': 'app.engine.vision.fringe_guidance',
    'validate_laser_ai_guidance': 'app.engine.vision.fringe_guidance',
    'MicrometerOCR': 'app.engine.vision.micrometer_ocr',
    'MotionFrameEnhancer': 'app.engine.vision.motion_enhancement',
    'analyze_thickness_distribution': 'app.engine.vision.thickness_distribution',
    'sample_colour': 'app.engine.vision.thickness_distribution',
    'sample_colour_band': 'app.engine.vision.thickness_distribution',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
