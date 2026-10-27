import cv2
import numpy as np

from app.engine.camera.manager import CameraManager


class Capture:
    def __init__(self, reject=False):
        self.props = {cv2.CAP_PROP_EXPOSURE: -6., cv2.CAP_PROP_GAIN: 0.}
        self.reject = reject
        self.calls = []

    def set(self, key, value):
        self.calls.append((key, value))
        if self.reject and key == cv2.CAP_PROP_EXPOSURE:
            return False
        self.props[key] = value
        return True

    def get(self, key):
        return self.props.get(key, 0.)


def test_rejected_exposure_is_not_reported_as_verified():
    camera = CameraManager(clarity_config={"enabled": True})
    cap = Capture(reject=True)
    camera._apply_clarity_profile(cap, motion=True)
    status = camera.clarity_status()
    assert status["software_enabled"]
    assert not status["hardware_verified"]
    assert status["warning"]
    assert status["exposure"] == -6


def test_roi_blur_shortens_exposure_before_increasing_gain_and_clear_frame_recovers():
    camera = CameraManager(clarity_config={"enabled": True, "check_frames": 1,
        "trigger_checks": 1, "motion_exposure": -7, "motion_gain": 0})
    cap = Capture()
    camera.set_clarity_assist(True)
    camera._apply_clarity_profile(cap, motion=True)
    camera.set_clarity_region((40, 40, 40, 40))
    frame = np.full((120, 160, 3), 120, dtype=np.uint8)
    # Background stays sharp; only the selected stripe patch changes.
    frame[:, ::2] = 220
    camera._update_clarity_assist(cap, frame)
    baseline = camera.clarity_status()["baseline"]
    frame[40:80, 40:80] = 120
    cap.calls.clear()
    camera._update_clarity_assist(cap, frame)
    assert camera.clarity_status()["blurred"]
    assert cap.calls[0] == (cv2.CAP_PROP_EXPOSURE, -8)
    assert not any(key == cv2.CAP_PROP_GAIN for key, _ in cap.calls)
    frame[:, ::2] = 220
    camera._update_clarity_assist(cap, frame)
    assert not camera.clarity_status()["blurred"]
    assert camera.clarity_status()["baseline"] >= baseline
    camera.set_clarity_region((20, 20, 20, 20), 15)
    assert camera.clarity_status()["baseline"] is None
