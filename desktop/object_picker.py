"""可搜索、多选的工程对象窗口；选择与模型修改相互独立。"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QVBoxLayout)
from . import dialog_styles


class ObjectPicker(QDialog):
    locate = Signal(str, int)

    def __init__(self, model, references, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择工程对象")
        self.resize(480, 480)
        box = QVBoxLayout(self)
        box.addWidget(QLabel("勾选节点或杆件加入对话；双击对象可在视口定位。"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索类型或编号…")
        box.addWidget(self.search)
        self.items = QListWidget()
        box.addWidget(self.items, 1)
        for kind, field, title in [("node", "nodes", "节点"), ("member", "members", "杆件")]:
            for obj in model.get(field, []):
                detail = (f"({obj.get('x')}, {obj.get('y')}, {obj.get('z')})" if kind == "node"
                          else f"节点 {obj.get('i')} → {obj.get('j')}")
                item = QListWidgetItem(f"{title} {obj['id']}  ·  {detail}", self.items)
                item.setData(Qt.ItemDataRole.UserRole, (kind, obj["id"]))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if (kind, obj["id"]) in references
                                   else Qt.CheckState.Unchecked)
        self.search.textChanged.connect(self._filter)
        self.items.itemDoubleClicked.connect(lambda item: self.locate.emit(
            *item.data(Qt.ItemDataRole.UserRole)))
        buttons = dialog_styles.button_box(self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("加入对话")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)

    def _filter(self, text):
        for i in range(self.items.count()):
            item = self.items.item(i)
            item.setHidden(text.casefold() not in item.text().casefold())

    def references(self):
        return [self.items.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.items.count())
                if self.items.item(i).checkState() == Qt.CheckState.Checked]
