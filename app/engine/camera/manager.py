"""相机采集模块 — 基于 OpenCV 的摄像头管理"""
from __future__ import annotations
from collections import deque
import sys
import threading
import time
import cv2
import numpy as np
from app.engine.logging import logger
from app.engine.vision.motion_enhancement import MotionFrameEnhancer
from app.engine.camera.registry import CAMERA_REGISTRY, CameraRegistry


class CameraManager:
    """管理 USB 摄像头的打开、帧采集、参数配置和释放"""

    def __init__(
        self,
        index: int = 1,
        resolution: tuple[int, int] = (1280, 1024),
        fps: int = 60,
        clarity_config: dict | None = None,
        owner: str = "main-camera",
        registry: CameraRegistry = CAMERA_REGISTRY,
        auto_mode: bool = False,
    ):
        self.index = index
        self.resolution = resolution
        self.fps = fps
        self.auto_mode = auto_mode
        self.mode_note = ""
        self.actual_resolution = None
        self.actual_fps = 0.0
        self.last_error = ""
        self.clarity_config = dict(clarity_config or {})
        self.owner = str(owner)
        self.registry = registry
        self._motion_enhancer = MotionFrameEnhancer(
            self.clarity_config.get("software_enhancement", {}))
        self._cap: cv2.VideoCapture | None = None
        self._opened = False
        self._lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_at = 0.0
        self._capture_stop = threading.Event()
        self._open_done = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._clarity_requested = False
        self._clarity_applied = False
        self._clarity_reapply_requested = False
        self._clarity_frame_count = 0
        self._clarity_blur_checks = 0
        self._clarity_samples: deque[float] = deque(maxlen=20)
        self._clarity_baseline: float | None = None
        self._clarity_score = 0.0
        self._clarity_brightness = 0.0
        self._clarity_exposure: float | None = None
        self._clarity_gain: float | None = None
        self._clarity_roi = None
        self._clarity_angle = 0.0
        self._clarity_hardware_ok = False
        self._clarity_warning = ""
        self.backend_name = ""

    @staticmethod
    def _backend_candidates() -> list[tuple[int, str]]:
        """Windows 优先 DirectShow，降低 MSMF 多摄像头同时打开失败率。"""
        candidates: list[tuple[int, str]] = []
        if sys.platform.startswith("win"):
            candidates.extend([
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
            ])
        candidates.append((cv2.CAP_ANY, "系统默认"))
        return candidates

    @classmethod
    def _open_device(cls, index: int) -> tuple[cv2.VideoCapture | None, str]:
        for backend, name in cls._backend_candidates():
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap, name
            cap.release()
        return None, ""

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """在专用采集线程中打开摄像头，并等待设备初始化完成。"""
        with self._lock:
            if self._opened:
                return True
            self._latest_frame = None
            self.last_error = ""
            self._capture_stop.clear()
            self._open_done.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"camera-{self.index}-capture",
                daemon=True,
            )
            self._capture_thread.start()
        # DirectShow 对创建线程敏感；打开与 read 必须由同一线程完成。
        if not self._open_done.wait(timeout=15.0):
            self.last_error = "设备初始化或等待首帧超时"
            logger.error("摄像头 %s 初始化超时", self.index)
            self.stop()
            return False
        with self._lock:
            return self._opened

    def _capture_loop(self) -> None:
        """在同一线程完成打开、持续采集和释放，兼容 Windows DirectShow。"""
        cap: cv2.VideoCapture | None = None
        leased = self.registry.acquire(self.index, self.owner)
        try:
            if not leased:
                self.last_error = f"已由 {self.registry.owner_of(self.index)} 占用"
                logger.error(
                    "摄像头 %s 已由 %s 占用，%s 无法打开",
                    self.index, self.registry.owner_of(self.index), self.owner)
                return
            mode = None
            if self.auto_mode:
                from app.engine.camera.modes import preferred_mode, supported_modes

                try:
                    logger.info("相机 %s 正在读取设备能力", self.index)
                    # A 5 MP/30 fps capability entry delivered only 14–15 fps
                    # in this setup. Stay within the tested 1080p pixel budget
                    # for automatic preview; larger modes remain manual options.
                    mode = preferred_mode(supported_modes(self.index), self.fps,
                                          max_pixels=1920 * 1080)
                except Exception as exc:
                    logger.warning("相机能力读取失败，保留手动参数: %s", exc)
                if mode is None:
                    self.mode_note = "无法读取设备能力，使用手动参数"
            cap, backend_name = self._open_device(self.index)
            if cap is None:
                self.last_error = "驱动无法打开设备，请检查连接或其他程序占用"
                logger.error("无法打开摄像头 (index=%s)", self.index)
                return
            # 在设置分辨率前请求 MJPEG，降低两路 USB 视频的总带宽。
            if mode is not None and backend_name != "DirectShow":
                mode = None
                self.mode_note = "当前后端无法核对设备能力，使用手动参数"
            resolution = (mode.width, mode.height) if mode else self.resolution
            fps = mode.fps if mode else self.fps
            logger.info("相机 %s 单次连接，配置 %s×%s @%.1f fps", self.index,
                        resolution[0], resolution[1], fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(mode.fourcc if mode else "MJPG")))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, fps)
            # DirectShow can revert to YUY2 while renegotiating size/rate.
            # Reapply the selected transport last to avoid 1–5 fps at 5 MP.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*(mode.fourcc if mode else "MJPG")))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.actual_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if mode is not None:
                self.mode_note = f"自动选择 {mode.width}×{mode.height} @ {mode.fps:.1f} fps"
            self._apply_clarity_profile(cap, motion=False)
            with self._lock:
                if self._capture_stop.is_set():
                    return
                self._cap = cap
                self.backend_name = backend_name
            logger.info("相机 %s 等待首帧", self.index)
            first_frame = self._read_first_frame(cap)
            if first_frame is None:
                if not self._capture_stop.is_set():
                    self.last_error = "驱动已连接，但没有收到画面；请检查设备连接及采集参数"
                    logger.error("相机 %s: %s", self.index, self.last_error)
                return
            with self._frame_lock:
                self._latest_frame = first_frame
                self._latest_frame_at = time.monotonic()
            with self._lock:
                if self._capture_stop.is_set():
                    return
                self._opened = True
            self._open_done.set()
            logger.info(
                "摄像头 %s 已打开（%s）: %sx%s @%.1ffps",
                self.index, backend_name,
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv2.CAP_PROP_FPS),
            )
            while not self._capture_stop.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                captured_at = time.monotonic()
                self._update_clarity_assist(cap, frame)
                with self._lock:
                    enhance = self._clarity_applied
                    score = self._clarity_score
                    baseline = self._clarity_baseline
                if enhance:
                    try:
                        frame = self._motion_enhancer.apply(
                            frame,
                            clarity_score=score,
                            clarity_baseline=baseline,
                            blur_ratio=float(self.clarity_config.get(
                                "blur_ratio", 0.55)),
                        )
                    except (cv2.error, ValueError) as exc:
                        self._motion_enhancer.enabled = False
                        logger.error(
                            "相机 %s 软件清晰度增强失败，已降级为原始画面: %s",
                            self.index, exc,
                        )
                else:
                    self._motion_enhancer.last_strength = 0.0
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_at = captured_at
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("摄像头 %s 采集失败: %s", self.index, exc)
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                if self._cap is cap:
                    self._cap = None
                self._opened = False
            if leased:
                self.registry.release(self.index, self.owner)
            self._open_done.set()

    def _read_first_frame(self, cap):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not self._capture_stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return frame
            time.sleep(.01)
        return None

    def stop(self):
        """关闭摄像头"""
        self._capture_stop.set()
        thread = self._capture_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is not None and thread.is_alive():
            # 某些驱动会永久阻塞 read；跨线程 release 仅作为超时兜底。
            with self._lock:
                cap = self._cap
            if cap is not None:
                cap.release()
            thread.join(timeout=0.5)
        with self._lock:
            self._cap = None
            self._opened = False
            self._capture_thread = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_at = 0.0
        logger.info("摄像头已关闭")

    def read(self) -> np.ndarray | None:
        """返回采集线程保存的最新 BGR 帧，不阻塞 UI 或推理线程。"""
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def set_clarity_assist(self, enabled: bool) -> None:
        """请求采集线程切换运动清晰度增强，属性仍由采集线程设置。"""
        with self._lock:
            self._clarity_requested = bool(
                enabled and self.clarity_config.get("enabled", True))
            self._clarity_blur_checks = 0
            if self._clarity_requested and self._clarity_samples:
                values = sorted(self._clarity_samples)
                self._clarity_baseline = values[len(values) // 2]

    def latest_sample(self):
        """返回实际采集时间；重复读缓存不能延长闭环输入有效期。"""
        with self._frame_lock:
            return (None if self._latest_frame is None else self._latest_frame.copy(),
                    self._latest_frame_at)

    def configure_motion_clarity(
        self, *, exposure: float, gain: float, enabled: bool,
    ) -> None:
        """更新运动增强参数，并由采集线程安全地应用到相机。"""
        exposure = min(0.0, max(-20.0, float(exposure)))
        gain = min(255.0, max(0.0, float(gain)))
        with self._lock:
            self.clarity_config["motion_exposure"] = exposure
            self.clarity_config["motion_gain"] = gain
            requested = bool(
                enabled and self.clarity_config.get("enabled", True))
            # 参数可能与上次相同；显式“应用”仍应刷新设备属性。
            self._clarity_reapply_requested = requested
            self._clarity_requested = requested
            self._clarity_blur_checks = 0
            if requested and self._clarity_samples:
                values = sorted(self._clarity_samples)
                self._clarity_baseline = values[len(values) // 2]

    def set_clarity_region(self, roi=None, angle=0.0):
        """ROI uses the rotated analysis frame, just like the GUI selection."""
        region = tuple(roi) if roi is not None else None
        with self._lock:
            if (region, float(angle)) != (self._clarity_roi, self._clarity_angle):
                self._clarity_roi, self._clarity_angle = region, float(angle)
                self._clarity_samples.clear()
                self._clarity_baseline = None
                self._clarity_blur_checks = 0

    def clarity_status(self) -> dict:
        with self._lock:
            return {
                "enabled": self._clarity_applied,
                "score": self._clarity_score,
                "baseline": self._clarity_baseline,
                "brightness": self._clarity_brightness,
                "exposure": self._clarity_exposure,
                "gain": self._clarity_gain,
                "hardware_verified": self._clarity_hardware_ok,
                "warning": self._clarity_warning,
                "region": "ROI" if self._clarity_roi is not None else "中央区域",
                "blurred": bool(self._clarity_baseline and self._clarity_score <
                    self._clarity_baseline * float(self.clarity_config.get("blur_ratio", 0.55))),
                "software_enabled": bool(
                    self._clarity_applied and self._motion_enhancer.enabled),
                "software_strength": self._motion_enhancer.last_strength,
                "stripe_strength": self._motion_enhancer.last_stripe_strength,
                "color_gain": self._motion_enhancer.last_color_gain,
            }

    def _apply_clarity_profile(
        self, cap: cv2.VideoCapture, *, motion: bool,
    ) -> None:
        if not self.clarity_config:
            return
        exposure_key = "motion_exposure" if motion else "preview_exposure"
        gain_key = "motion_gain" if motion else "preview_gain"
        exposure = float(self.clarity_config.get(
            exposure_key, -7 if motion else -6))
        gain = float(self.clarity_config.get(gain_key, 80 if motion else 0))
        auto_ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        exposure_ok = cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        self._clarity_exposure = float(cap.get(cv2.CAP_PROP_EXPOSURE))
        self._clarity_gain = float(cap.get(cv2.CAP_PROP_GAIN))
        auto_value = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        self._clarity_hardware_ok = bool(auto_ok and exposure_ok and
            np.isfinite(self._clarity_exposure) and
            abs(self._clarity_exposure - exposure) < 0.25 and
            abs(auto_value - 0.25) < 0.1)
        self._clarity_warning = ("" if self._clarity_hardware_ok or not motion else
            "曝光未核验：相机可能拒绝手动曝光或参数回读不一致；软件锐化仍可运行")
        if self._clarity_warning:
            logger.warning(self._clarity_warning)
        self._clarity_applied = motion
        self._clarity_frame_count = 0
        self._clarity_blur_checks = 0
        logger.info(
            "相机 %s 清晰度配置：%s，曝光 %.1f，增益 %.1f",
            self.index, "运动增强" if motion else "普通预览",
            self._clarity_exposure, self._clarity_gain,
        )

    def _update_clarity_assist(
        self, cap: cv2.VideoCapture, frame: np.ndarray,
    ) -> None:
        if not self.clarity_config:
            return
        with self._lock:
            requested = self._clarity_requested
            applied = self._clarity_applied
            reapply = self._clarity_reapply_requested
            self._clarity_reapply_requested = False
        if requested != applied or reapply:
            self._apply_clarity_profile(cap, motion=requested)

        self._clarity_frame_count += 1
        interval = max(1, int(self.clarity_config.get("check_frames", 8)))
        if self._clarity_frame_count % interval:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._lock:
            roi, angle = self._clarity_roi, self._clarity_angle
        if angle:
            from app.engine.vision.angle import rotate_expand

            gray = rotate_expand(gray, angle, border_value=0)
        height, width = gray.shape
        if roi is not None:
            x, y, w, h = map(int, roi)
            crop = gray[max(0, y):min(height, y+h), max(0, x):min(width, x+w)]
        else:
            crop = gray[height // 5: height * 4 // 5,
                        width // 5: width * 4 // 5]
        if crop.size == 0:
            return
        score = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        brightness = float(crop.mean())
        with self._lock:
            self._clarity_score = score
            self._clarity_brightness = brightness
            active = self._clarity_applied
            baseline = self._clarity_baseline
            if not active:
                self._clarity_samples.append(score)
                values = sorted(self._clarity_samples)
                self._clarity_baseline = values[len(values) // 2]
                return

        if baseline is None or baseline <= 0:
            baseline = max(1.0, score)
            with self._lock:
                self._clarity_baseline = baseline
        elif score > baseline:
            # 停转后清晰帧能更新基线，避免启用时已模糊而永远判为正常。
            baseline = score
            with self._lock:
                self._clarity_baseline = baseline
        blur_ratio = float(self.clarity_config.get("blur_ratio", 0.55))
        if score >= baseline * blur_ratio:
            self._clarity_blur_checks = 0
            return
        self._clarity_blur_checks += 1
        trigger = max(1, int(self.clarity_config.get("trigger_checks", 2)))
        if self._clarity_blur_checks < trigger:
            return
        self._clarity_blur_checks = 0

        max_gain = float(self.clarity_config.get("max_gain", 100))
        min_exposure = float(self.clarity_config.get("min_exposure", -8))
        brightness_limit = float(self.clarity_config.get(
            "min_brightness_for_shorter_exposure", 20))
        current_gain = float(cap.get(cv2.CAP_PROP_GAIN))
        current_exposure = float(cap.get(cv2.CAP_PROP_EXPOSURE))
        changed = False
        # 亮度足够时先缩短曝光；增益只能补亮，不能消除运动积分拖影。
        if brightness >= brightness_limit and current_exposure > min_exposure:
            target = max(min_exposure, current_exposure - 1)
            accepted = cap.set(cv2.CAP_PROP_EXPOSURE, target)
            actual = float(cap.get(cv2.CAP_PROP_EXPOSURE))
            changed = bool(accepted and abs(actual-target) < 0.25)
            manual = abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)-0.25) < 0.1
            with self._lock:
                self._clarity_hardware_ok = bool(changed and manual)
                if not self._clarity_hardware_ok:
                    self._clarity_warning = "曝光缩短未核验，请检查相机驱动或降低电机速度"
                else:
                    self._clarity_warning = ""
        elif current_gain < max_gain:
            changed = cap.set(
                cv2.CAP_PROP_GAIN, min(max_gain, current_gain + 10))
        if changed:
            with self._lock:
                self._clarity_exposure = float(
                    cap.get(cv2.CAP_PROP_EXPOSURE))
                self._clarity_gain = float(cap.get(cv2.CAP_PROP_GAIN))
            logger.info(
                "相机 %s 检测到模糊：清晰度 %.1f/基线 %.1f，亮度 %.1f，"
                "调整曝光 %.1f、增益 %.1f",
                self.index, score, baseline, brightness,
                self._clarity_exposure, self._clarity_gain,
            )

    @property
    def is_opened(self) -> bool:
        return self._opened

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def detect_all(
        max_index: int = 9,
        *,
        registry: CameraRegistry = CAMERA_REGISTRY,
        owner: str = "camera-scanner",
    ) -> list[int]:
        """扫描空闲摄像头；绝不重复打开已被工作模块占用的设备。"""
        available = []
        for idx in range(max_index + 1):
            if not registry.acquire(idx, owner):
                continue
            try:
                cap, _backend = CameraManager._open_device(idx)
                if cap is not None:
                    available.append(idx)
                    cap.release()
            finally:
                registry.release(idx, owner)
        return available
