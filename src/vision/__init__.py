"""视觉检测模块：YOLO 检测、画面校正、条纹识别与中心定位。"""
from src.vision.detector import YOLODetector
from src.vision.angle import rotate_expand
from src.vision.correct import FrameCorrector
from src.vision.fringe_center import (
    CenterTracker,
    find_center_by_band,
    find_center_in_region,
)
from src.vision.fringe_motion import FringeMotionTracker
from src.vision.fringe_recognition import (
    FringeRecognitionTracker,
    analyse_fringe_texture,
)
from src.vision.micrometer_ocr import MicrometerOCR
from src.vision.motion_enhancement import MotionFrameEnhancer
from src.vision.thickness_distribution import (
    analyze_thickness_distribution,
    sample_colour,
    sample_colour_band,
)
