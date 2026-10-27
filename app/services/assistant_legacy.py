"""Read-only legacy assistant features. No device actions or implicit network calls."""
from __future__ import annotations
import json
from pathlib import Path
import time

from app.core.config import ROOT
from app.engine.agent.experiment_guidance import build_guidance_decision
from app.engine.vision.fringe_adjustment import compare_fringe_adjustment

STAGES = (
    ("laser", "① 激光非定域条纹", "激光光源下是否看到非定域干涉条纹？", "确认光源和光路，观察条纹形态后再继续。"),
    ("diffuser", "② 毛玻璃与等厚直条纹", "加毛玻璃后条纹是否接近直线？", "缓慢逼近等光程位置，以条纹接近直线为观察标志。"),
    ("white", "③ 白光彩色条纹", "换白光扩展光源后是否出现彩色条纹？", "保持已调好的光路，观察彩色条纹是否进入画面。"),
    ("zero", "④ 中央黑色零级条纹", "是否看到中央黑色零级条纹？", "核对中心位置与新鲜读数，稳定后记录原始画面。"),
)


def local_guidance(state, stage):
    selected = next((item for item in STAGES if item[0] == stage), None)
    if selected is None:
        raise ValueError("未知实验阶段")
    _, title, question, action = selected
    at = state.get("reading_at")
    context = {
        "camera": {"interferometer_running": state.get("camera_open"), "fps": state.get("fps")},
        "motor": {"connected": state.get("motor_connected"), "auto_enabled": state.get("search_running")},
        "micrometer": {"connected": at is not None, "reading_age_seconds": None if at is None else max(0, time.time()-at)},
        "vision": {"prediction_running": state.get("detecting"), "fringe_present": state.get("center_x") is not None},
        "experiment_progress": {"stage": title, "next_action": action, "completion_criterion": question},
    }
    decision = build_guidance_decision(context)
    return {**decision.as_dict(), "observation_question": question}


def adjustment_metrics(state):
    # Only real state fields; missing curvature/CV/quality remain missing.
    return {"angle_deg": state.get("tilt_deg")}


def compare_adjustment(before, after):
    return compare_fringe_adjustment(before or {}, after or {})


def export_conversation(path, entries, *, data_source="live"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data_source": data_source, "exported_at": time.time(), "entries": list(entries)}
    if target.suffix.lower() == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = "# 实验助手会话\n\n数据来源：" + data_source + "\n\n"
        text += "\n\n".join(f"## {entry['role']}\n\n{entry['text']}" for entry in entries)
    target.write_text(text, encoding="utf8")
    return target


def review_image(frame, state, config, *, demo=False, provider=None, cancel_event=None):
    """Called only by an explicit review action. Provider injection keeps tests offline."""
    from app.engine.agent.provider import ProviderCancelled
    if demo:
        raise ValueError("演示模式不发送图像复核请求")
    if cancel_event is not None and cancel_event.is_set():
        raise ProviderCancelled("请求已取消")
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("当前没有可复核画面")
    import yaml
    settings_path = ROOT / "config.yaml"
    legacy = yaml.safe_load(settings_path.read_text(encoding="utf8")) if settings_path.is_file() else {}
    llm = ((legacy or {}).get("agent") or {}).get("llm") or {}
    model = str((llm.get("models") or {}).get("vision") or "").strip()
    if not model:
        raise ValueError("旧 config.yaml 未配置 agent.llm.models.vision")
    if provider is None:
        from app.agent.runtime import make_provider
        # Reuse credentials without printing or copying them; endpoint follows legacy vision config.
        class VisionConfig:
            def get(self, key, default=None):
                return llm.get(key.removeprefix("agent.llm."), config.get(key, default))
        provider = make_provider(VisionConfig())
    if not provider.available:
        raise ValueError("未配置 API Key，无法进行图像复核")
    import cv2
    image = frame
    height, width = image.shape[:2]
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        image = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("图像 JPEG 编码失败")
    prompt = ("复核迈克尔逊干涉图，仅做定性判断。说明图像是否足以支持结论，给一个问题、小步操作、预期变化与停止条件。"
              "不计算厚度、不把颜色当标定相位，不猜测旋钮方向，不执行硬件动作。\n程序状态：" + json.dumps(state, ensure_ascii=False))
    answer = provider.chat_with_image(prompt, encoded.tobytes(), model=model,
        detail=str(llm.get("vision_detail", "low")), max_tokens=int(llm.get("vision_max_tokens", 500)),
        cancel_event=cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise ProviderCancelled("请求已取消")
    return answer
