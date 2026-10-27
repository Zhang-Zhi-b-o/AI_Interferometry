"""原生 Qt Markdown 对话；公式由已有 Matplotlib 在后台排版，不访问 CDN。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import html
import re
import threading

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QPalette, QTextDocument
from PySide6.QtWidgets import QTextBrowser

_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat-math")
_MATH_LOCK = threading.Lock()
_TOKENS = re.compile(
    r"(?P<code>```[\s\S]*?```|`[^`\n]*`)|"
    r"\$\$(?P<display>[\s\S]*?)\$\$|"
    r"\\\[(?P<bracket>[\s\S]*?)\\\]|"
    r"\\\((?P<inline>[\s\S]*?)\\\)|"
    r"(?<!\\)\$(?!\$)(?P<dollar>[^\n$]+?)(?<!\\)\$"
)


def extract_math(text):
    formulas = []

    def substitute(match):
        if match.group("code") is not None:
            return match.group()
        formula = next(value for value in match.groups()[1:] if value is not None).strip()
        token = f"CHATMATHTOKEN{len(formulas)}END"
        formulas.append((token, formula, match.group()))
        return token

    return _TOKENS.sub(substitute, text), formulas


@lru_cache(maxsize=128)
def formula_png(formula, color):
    # Matplotlib 的数学解析器不启动外部 TeX，也不执行模型返回的宏命令。
    if len(formula) > 4000:
        raise ValueError("公式过长")
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    from matplotlib import rc_context

    formula = formula.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    if re.search(r"\\begin\{(?:aligned|align\*?|gathered|gather\*?|split|equation\*?)\}", formula):
        from PIL import Image

        cleaned = re.sub(r"\\(?:begin|end)\{(?:aligned|align\*?|gathered|gather\*?|split|equation\*?)\}", "", formula)
        lines = [line.replace("&", "").strip() for line in re.split(r"\\\\", cleaned)]
        images = [Image.open(BytesIO(formula_png(line, color))).convert("RGBA") for line in lines if line]
        if not images:
            raise ValueError("空公式")
        combined = Image.new("RGBA", (max(image.width for image in images),
                                      sum(image.height+8 for image in images)), (0, 0, 0, 0))
        top = 0
        for image in images:
            combined.paste(image, (0, top))
            top += image.height+8
        result = BytesIO()
        combined.save(result, format="PNG")
        return result.getvalue()
    buffer = BytesIO()
    with _MATH_LOCK, rc_context({"savefig.transparent": True}):
        math_to_image("$" + formula + "$", buffer, prop=FontProperties(size=14),
                      dpi=160, format="png", color=color)
    from PIL import Image, ImageOps

    image = Image.open(BytesIO(buffer.getvalue())).convert("RGBA")
    image = ImageOps.expand(image, border=4, fill=(0, 0, 0, 0))
    padded = BytesIO()
    image.save(padded, format="PNG")
    return padded.getvalue()


def render_formulas(formulas, color):
    rendered = {}
    for token, formula, original in formulas:
        try:
            rendered[token] = (formula_png(formula, color), original, None)
        except (ValueError, RuntimeError) as exc:
            # 未闭合/不支持的公式保留原文，不吞掉整条回答。
            rendered[token] = (None, original, str(exc))
    return rendered


class MathChat(QTextBrowser):
    rendering_ready = Signal(int, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self._raw = ""
        self._version = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self.render_now)
        self.rendering_ready.connect(self._apply_render)

    def toPlainText(self):  # noqa: N802
        """会话导出与测试读取原始 Markdown/LaTeX，不读取图片占位字符。"""
        return self._raw

    def clear(self):
        self._raw = ""
        self._version += 1
        self._timer.stop()
        super().clear()

    def setPlainText(self, text):  # noqa: N802
        self._raw = str(text)
        self._changed()

    def appendPlainText(self, text):  # noqa: N802
        self._raw += ("\n" if self._raw else "") + str(text)
        self._changed()

    def append_delta(self, text):
        self._raw += str(text)
        self._changed()

    def _changed(self):
        self._version += 1
        scroll = self.verticalScrollBar()
        at_bottom, previous = scroll.value() >= scroll.maximum()-12, scroll.value()
        # 流式期间立即显示字符，后台排版以节流刷新，避免每个 token 重绘公式。
        super().setPlainText(self._raw)
        scroll.setValue(scroll.maximum() if at_bottom else previous)
        if not self._timer.isActive():
            self._timer.start()

    def render_now(self):
        self._timer.stop()
        version = self._version
        markdown, formulas = extract_math(self._raw)
        if not formulas:
            self._apply_render(version, markdown, {})
            return
        color = self.palette().color(QPalette.ColorRole.Text).name()
        future = _WORKER.submit(render_formulas, formulas, color)

        def complete(task):
            try:
                self.rendering_ready.emit(version, markdown, task.result())
            except Exception:
                # 窗口关闭后丢弃后台排版结果。
                pass

        future.add_done_callback(complete)

    def _apply_render(self, version, markdown, rendered):
        if version != self._version:
            return
        scroll = self.verticalScrollBar()
        at_bottom = scroll.value() >= scroll.maximum() - 12
        previous = scroll.value()
        document = QTextDocument(self)
        document.setDefaultFont(self.font())
        markdown = re.sub(r"(?m)^(你|助手)：", r"\n\n**\1：** ", markdown)
        document.setMarkdown(markdown, QTextDocument.MarkdownFeature.MarkdownNoHTML
                             | QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
        markup = document.toHtml()
        images = []
        for token, (data, source, error) in rendered.items():
            if data is None:
                markup = markup.replace(token, '<code title="' + html.escape(error, quote=True)
                                        + '">' + html.escape(source) + '</code>')
                continue
            image = QImage.fromData(data, "PNG")
            url = QUrl(f"math:{version}/{token}")
            width, height = max(1, round(image.width()*.65)), max(1, round(image.height()*.65))
            markup = markup.replace(token, f'<img src="{url.toString()}" width="{width}" height="{height}" '
                                    f'alt="{html.escape(source, quote=True)}" />')
            images.append((url, image))
        document.setHtml(markup)
        for url, image in images:
            document.addResource(QTextDocument.ResourceType.ImageResource, url, image)
        self.setDocument(document)
        scroll.setValue(scroll.maximum() if at_bottom else previous)

    def loadResource(self, kind, name):  # noqa: N802
        # 模型 Markdown 中的远端图片不触发隐式网络或本地文件读取。
        if name.scheme() != "math":
            return None
        return super().loadResource(kind, name)
