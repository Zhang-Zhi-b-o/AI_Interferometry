from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout

from app.services.assistant_settings import save_assistant_settings


class AssistantSettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("实验助手 · API 设置")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        note = QLabel("填写你自己的 API Key。密钥只保存在本机，不会随程序上传。\n留空可保留已有密钥；DEEPSEEK_API_KEY 环境变量优先。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.api_base = QLineEdit(str(config.get("agent.llm.api_base", "https://api.deepseek.com")))
        self.model = QLineEdit(str(config.get("agent.llm.models.flash", "deepseek-chat")))
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("粘贴你自己的 API Key")
        form = QFormLayout()
        form.addRow("API 地址", self.api_base)
        form.addRow("模型名称", self.model)
        form.addRow("API Key", self.api_key)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        try:
            save_assistant_settings(self.config, api_base=self.api_base.text(),
                                    model=self.model.text(), api_key=self.api_key.text())
        except (ValueError, OSError, TypeError) as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        self.accept()
