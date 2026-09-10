"""杆件属性指派对话框。

当前内核把截面几何和材料分别存放在杆件上，因此一次“属性指派”必须同时
明确两者。只改截面会让几何优先创建的草稿继续缺材料，却在界面上看似完成。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, 
                               QFormLayout, QLabel, QVBoxLayout)

from . import dialog_styles


class AssignmentDialog(QDialog):
    def __init__(self, member_ids: list[int], sections: list[str],
                 materials: list[str], current_section: str = "",
                 current_material: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("属性指派")
        self.resize(360, 190)

        layout = QVBoxLayout(self)
        target = (f"杆件 {member_ids[0]}" if len(member_ids) == 1
                  else f"{len(member_ids)} 根杆件")
        layout.addWidget(QLabel(f"目标：{target}"))
        form = QFormLayout()
        self.cmb_section = QComboBox(self)
        self.cmb_section.addItems(sections)
        if current_section in sections:
            self.cmb_section.setCurrentText(current_section)
        self.cmb_material = QComboBox(self)
        self.cmb_material.addItems(materials)
        if current_material in materials:
            self.cmb_material.setCurrentText(current_material)
        form.addRow("截面", self.cmb_section)
        form.addRow("材料", self.cmb_material)
        layout.addLayout(form)

        hint = QLabel("两项会同时写入所选杆件；之后仍可在属性面板中单独修改。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = dialog_styles.button_box(self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_assignment(self) -> tuple[str, str]:
        return self.cmb_section.currentText(), self.cmb_material.currentText()
