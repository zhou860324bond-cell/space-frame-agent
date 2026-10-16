"""结构图纸识别的「文字证据」：格式、校验与确定性解析。

图纸识别和手绘草图识别的根本区别在于**数值从哪里来**。草图识别从像素位置
量坐标，再用一根杆件标定比例；图纸上本来写着轴线间距、标高和构件规格，
这些文字才是可追溯的数值来源。

所以大模型（或以后的 DXF / 矢量 PDF 读取器）只交出一份证据：每条都是图上
**原样抄下的文字**，附带出处（第几张图、图上哪块区域）。把文字变成数的是本
模块里的确定性解析器；把数组装成模型的是 ``drawing_assemble``。任何一步读不懂
就报出来交给用户，不猜、不按惯例默认。

证据格式（``EVIDENCE_FORMAT``）::

    {
      "format": "space-frame-drawing-evidence/v1",
      "images": [{"id": "P1", "name": "平面图.png", "sha256": "…",
                  "width": 2400, "height": 1600}],
      "units": {"length": {"value": "mm", "text": "尺寸以毫米计", "source": SRC},
                "elevation": {"value": "m", "text": "标高以米计", "source": SRC}},
      "grids": {"x": {"axes": [{"label": "1", "pos": 0.12, "source": SRC}],
                      "spacings": [{"from": "1", "to": "2", "text": "6000",
                                    "source": SRC}],
                      "total": {"text": "12000", "source": SRC},
                      "edges": [{"text": "1'-2\"", "source": SRC}]},
                "y": {…}},
      "levels": [{"name": "2F", "text": "3.600", "source": SRC}],
      "level_heights": [{"from": "1F", "to": "2F", "text": "3600", "source": SRC}],
      "members": [{"label": "KL1", "kind": "beam", "along": "A",
                   "from": "1", "to": "3", "levels": ["2F"], "source": SRC},
                  {"label": "KZ1", "kind": "column", "at": ["1", "A"],
                   "from_level": "1F", "to_level": "3F", "source": SRC}],
      "sections": [{"label": "KL1", "text": "HN400×200×8×13", "source": SRC}],
      "material": {"text": "Q355", "source": SRC},
      "support": {"kind": "fixed", "text": "柱脚刚接", "source": SRC},
      "notes": [{"text": "看不清的内容", "source": SRC}]
    }

``SRC`` 为 ``{"image": "P1", "bbox": [u1, v1, u2, v2], "confidence": 0.9}``
（0~1 归一化图片坐标，原点左上角），用户在核对表里填写的值为 ``{"user": true}``。
x 族轴线是数字轴（竖向轴线，沿水平方向排列），y 族是字母轴；``pos`` 是轴线
在图上的位置（x 族取横坐标 u，y 族取纵坐标 v），**只用于报警，不用于取值**。

矢量文件（DXF）的出处不是图上一块区域，而是文件里的实体：``SRC`` 写作
``{"file": "D1", "handles": ["2A", "2B"], "layer": "AXIS"}``（文件头变量写
``"header": "$INSUNITS"``），文件登记在 ``"files": [{"id": "D1", "name": …,
"sha256": …, "kind": "dxf"}]``。另有三处只有矢量入口会用到：

* 条目可带 ``"value"``：读取器从实体直接取得的数（如 DIMENSION 的实测值），
  ``text`` 仍是原文；两者不符时组装报冲突，用户改写后以文字为准。
* 条目可带 ``"check"``：需要用户确认的原因（如单位取自 $INSUNITS）。
* ``"layers": [{"name": "AXIS", "role": "axis", "summary": "LINE 3、CIRCLE 3",
  "source": SRC}]``：图层用途只由用户或大模型判断（``LAYER_ROLES``），
  ``role`` 为 null 表示尚未判断、暂不读取；判断结果进核对表确认。

间距与层高的 ``text`` 可以是英尺英寸写法和重复写法（「7 at 25'-0"」，from/to 为
首末轴号），由 :func:`parse_dimension` 与 :func:`expand_chain` 确定性展开，原文不改。
``edges`` 是首末轴线到建筑外边缘的尺寸，用来认出量到外边缘的总尺寸。``levels``
按从下到上排列；立面图只给层高时 ``text`` 为 null，标高由 ``level_heights`` 累加。
没有编号的梁柱 ``label`` 为 null，按位置抄写。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from dimension_constraints import length_to_m

EVIDENCE_FORMAT = "space-frame-drawing-evidence/v1"

#: 单位说明能识别的长度单位（换算系数由 dimension_constraints 统一给出）。
LENGTH_UNITS = ("mm", "cm", "m")

#: 置信度低于此值的证据要用户逐条确认；与草图识别面板的「低」档一致。
LOW_CONFIDENCE = 0.75

#: 矢量图纸的图层用途。图层名怎样对应这些用途只由用户或大模型判断，代码不按名字猜。
LAYER_ROLES = {
    "axis": "轴线与轴号", "dimension": "尺寸标注", "level": "标高", "beam": "梁",
    "column": "柱", "label": "杆件编号", "table": "截面表", "notes": "说明",
    "ignore": "不使用",
}

_FULL_WIDTH = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6",
    "７": "7", "８": "8", "９": "9", "．": ".", "，": ",", "＋": "+", "－": "-",
    "−": "-", "—": "-", "Ｘ": "×", "ｘ": "×", "＊": "×", "　": " ",
})
_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")


def length_factor(unit: Any) -> float | None:
    """单位名 → 米/单位；不认识的单位返回 None（不默认）。"""
    return length_to_m(1.0, unit)


def parse_number(text: Any) -> float | None:
    """把图上的一段数字文字变成数；读不懂返回 None。

    接受千分位（6,000）、数字间空格（6 000）、全角字符和标高写法
    （±0.000、+3.600、-0.450、AutoCAD 的 %%p0.000）。带任何其他字符
    （如「6000(估)」「约6m」）一律拒绝——单位和含义要另有证据。
    """
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(float(text)) else None
    if not isinstance(text, str):
        return None
    value = text.translate(_FULL_WIDTH).strip()
    for prefix in ("±", "%%p", "%%P"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", value)
    if not _NUMBER.fullmatch(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ---- 尺寸写法：英尺英寸、重复写法 ------------------------------------------------

#: 1 英寸 = 254/10000 米（1959 年国际码协定的定义值，不是近似）；1 英尺 = 12 英寸。
INCH_M = 0.0254

DIMENSION_HELP = ("可识别的写法：6000、6,000、25'-0\"、13'-4 1/2\"，以及重复写法"
                  "「7 at 25'-0\"」「6@3600」「3×6000」，可带「= 总长」。")

_PRIMES = str.maketrans({"′": "'", "’": "'", "‘": "'", "´": "'",
                         "″": '"', "”": '"', "“": '"'})
_FEET_INCHES = re.compile(
    r"(?:(?P<ft>\d+(?:\.\d+)?)\s*')?"
    r"(?:\s*-?\s*(?P<inch>\d+(?:\.\d+)?)?(?:\s*-?\s*(?P<num>\d+)/(?P<den>\d+))?\s*\")?")
_REPEAT = re.compile(r"(?P<count>\d+)\s*(?:SPA(?:CES|\.)?\s*)?(?:AT\b|@|×)\s*(?P<each>.+)",
                     re.IGNORECASE)


@dataclass(frozen=True)
class Dimension:
    """一条尺寸文字的确定性读法；原文仍保存在证据里。

    ``value`` 是每段的数：写法自带单位（英尺英寸）时为英寸，``inches`` 为真；
    否则是图纸长度单位，换算要靠图上的单位说明。``count`` 是重复写法的段数，
    「7 at 25'-0"」为 7 段、每段 300 英寸。
    """

    value: float
    count: int = 1
    inches: bool = False

    @property
    def total(self) -> float:
        return self.value * self.count

    def show(self, value: float | None = None) -> str:
        """按原写法的单位写出一个数（默认每段的数）。"""
        value = self.value if value is None else value
        return format_feet_inches(value) if self.inches else _plain(value)


def _plain(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value != 0 else "0"


def format_feet_inches(inches: float) -> str:
    """英寸 → 图上习惯的 ft'-in" 写法，如 300 → 25'-0"，160 → 13'-4"。"""
    sign = "-" if inches < 0 else ""
    inches = round(abs(inches), 6)
    feet = int(inches // 12)
    return f"{sign}{feet}'-{_plain(inches - 12 * feet)}\""


def _parse_feet_inches(text: str) -> float | None:
    """ft'-in" 写法 → 英寸；不是这种写法返回 None。"""
    sign = 1.0
    if text[:1] in "+-±":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:].strip()
    match = _FEET_INCHES.fullmatch(text)
    if not match or not any(match.group(key) for key in ("ft", "inch", "num")):
        return None
    inches = 12.0 * float(match.group("ft") or 0) + float(match.group("inch") or 0)
    if match.group("num"):
        denominator = float(match.group("den"))
        if denominator == 0:
            return None
        inches += float(match.group("num")) / denominator
    return sign * inches


def _single_dimension(text: str) -> Dimension | None:
    number = parse_number(text)
    if number is not None:
        return Dimension(number)
    inches = _parse_feet_inches(text)
    return None if inches is None else Dimension(inches, inches=True)


def parse_dimension(text: Any) -> Dimension:
    """尺寸文字 → :class:`Dimension`；读不懂抛 ValueError 并说明支持的写法。

    在 :func:`parse_number` 之外再认两种写法，都是确定性的：

    * 英尺英寸 ``25'-0"``、``13'-4"``、``1'-2"``、``22'-4 1/2"``，自带单位；
    * 重复写法 ``7 at 25'-0"``、``6@3600``、``3×6000``，可带 ``= 175'-0"``，
      等号两边对不上时拒绝（不替图纸决定哪边是对的）。
    """
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        number = parse_number(text)
        if number is None:
            raise ValueError(f"尺寸「{text}」不是有限数。" + DIMENSION_HELP)
        return Dimension(number)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("尺寸为空。" + DIMENSION_HELP)
    value = text.translate(_FULL_WIDTH).translate(_PRIMES).replace("''", '"').strip()
    value = re.sub(r"(?<=\d)\s*[xX*](?=\s*\d)", "×", value)
    left, equals, right = (part.strip() for part in value.partition("="))
    repeat = _REPEAT.fullmatch(left)
    if repeat:
        each = _single_dimension(repeat.group("each").strip())
        count = int(repeat.group("count"))
        if each is None or count < 1:
            raise ValueError(f"读不懂重复写法「{text}」。" + DIMENSION_HELP)
        dimension = Dimension(each.value, count, each.inches)
    else:
        dimension = _single_dimension(left)
        if dimension is None:
            raise ValueError(f"读不懂尺寸「{text}」。" + DIMENSION_HELP)
    if equals:
        total = _single_dimension(right)
        if total is None or total.inches != dimension.inches:
            raise ValueError(f"读不懂「{text}」等号后的总长。" + DIMENSION_HELP)
        if abs(total.value - dimension.total) > 1e-6 * max(1.0, abs(total.value)):
            raise ValueError(f"「{text}」中 {dimension.count} 段之和 "
                             f"{dimension.show(dimension.total)} 与等号后的 "
                             f"{total.show()} 不符；请对照原图改正其中一边。")
    return dimension


def read_dimension(item: Mapping[str, Any]) -> Dimension:
    """一条证据的尺寸读法：矢量入口从实体直接取得的 ``value`` 优先，``text`` 是原文。

    ``value`` 是实体实测值（图纸单位下的总长）。原文读得懂且与实测值不符（如
    DIMENSION 文字被改写成别的数）时抛 ValueError；原文读不懂时用实测值。
    用户改写过的条目以用户文字为准。"""
    value = item.get("value")
    source = item.get("source")
    by_user = isinstance(source, Mapping) and bool(source.get("user"))
    if by_user or isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        return parse_dimension(item.get("text"))
    value = float(value)
    try:
        dimension = parse_dimension(item.get("text"))
    except ValueError:
        return Dimension(value)
    if abs(dimension.total - value) > 1e-6 * max(1.0, abs(value)):
        raise ValueError(f"标注文字「{item.get('text')}」与实体实测值 {_plain(value)} 不符；"
                         "请照图纸意图填写正确的数。")
    return dimension


@dataclass
class ChainLink:
    """相邻两根轴线（或两层）之间的一段，由证据第 ``index`` 条给出。"""

    index: int
    dimension: Dimension
    part: int = 0                   # 重复写法展开后的第几段（从 1 起）；单段写法为 0


@dataclass
class Chain:
    """一组间距证据按轴号顺序展开后的结果，轴网与层高共用。"""

    labels: list[str]
    links: dict[tuple[str, str], ChainLink] = field(default_factory=dict)
    #: 跨越多段的单值尺寸或与已有分段重复的尺寸：(条目序号, 起, 止, 读法)，只用来核对。
    spans: list[tuple[int, str, str, Dimension]] = field(default_factory=list)
    #: 条目序号 → (给用户的说明, 给识别模型重读的中性描述，空串表示不回喂)。
    problems: dict[int, tuple[str, str]] = field(default_factory=dict)
    #: 被有问题的条目占住的相邻段 → 条目序号；这些段不再另报「缺失」。
    blocked: dict[tuple[str, str], int] = field(default_factory=dict)
    #: 自动补的层名 → 由哪一条重复写法补出。
    generated: dict[str, int] = field(default_factory=dict)


def expand_chain(labels: list[str], items: list[Any], *, name_gaps: bool = False,
                 noun: str = "轴号") -> Chain:
    """把间距证据（含重复写法）确定性展开为相邻两轴之间的逐段间距。

    重复写法「7 at 25'-0"」的 from/to 是这串间距的首末轴号，两者之间正好隔 7 段时
    逐段展开；段数对不上就报出来，不猜哪根轴线没读到。立面图常常不给中间各层的
    层名，``name_gaps`` 为真时，若重复写法的首末层在列表里相邻，就按段数补上自动
    层名（「L1 上第 2 层」），数值仍全部来自原文。原文不改写，展开结果只引用条目序号。
    """
    labels = list(labels)
    entries = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        a, b = str(item.get("from") or "").strip(), str(item.get("to") or "").strip()
        try:
            dimension: Dimension | None = read_dimension(item)
        except ValueError as exc:
            dimension, message = None, str(exc)
        else:
            message = ""
        entries.append((index, item, a, b, dimension, message))
    generated: dict[str, int] = {}
    if name_gaps:
        for index, _item, a, b, dimension, _ in entries:
            if dimension is None or dimension.count < 2 or a not in labels \
                    or b not in labels:
                continue
            i, j = sorted((labels.index(a), labels.index(b)))
            if j - i != 1:
                continue
            names = [f"{labels[i]} 上第 {k} 层" for k in range(1, dimension.count)]
            if any(name in labels for name in names):
                continue
            labels[j:j] = names
            generated.update(dict.fromkeys(names, index))
    chain = Chain(labels, generated=generated)
    order = {label: position for position, label in enumerate(labels)}
    for index, item, a, b, dimension, message in entries:
        if a not in order or b not in order:
            missing = "、".join(label for label in (a, b) if label not in order) or "（空）"
            chain.problems[index] = (
                f"引用了不存在的{noun}「{missing}」；请改正或删除这条尺寸。",
                f"尺寸 {a}→{b} 引用的{noun}不在{noun}列表里")
            continue
        i, j = sorted((order[a], order[b]))
        pairs = [(labels[k], labels[k + 1]) for k in range(i, j)]
        if not pairs:
            chain.problems[index] = (f"起止{noun}相同；请改正或删除这条尺寸。",
                                     f"尺寸 {a}→{b} 的起止{noun}相同")
            continue
        if dimension is None or (dimension.count > 1 and dimension.count != len(pairs)):
            if dimension is None:
                chain.problems[index] = (message, "")
            else:
                chain.problems[index] = (
                    f"重复写法写的是 {dimension.count} 段，但{noun} {a}→{b} 之间有 "
                    f"{len(pairs)} 段；请对照原图改正起止{noun}或这条尺寸。",
                    f"尺寸 {a}→{b} 原文「{item.get('text')}」的段数与两{noun}之间的段数对不上")
            for pair in pairs:
                chain.blocked.setdefault(pair, index)
            continue
        if len(pairs) == dimension.count and not any(pair in chain.links for pair in pairs):
            for part, pair in enumerate(pairs, 1):
                chain.links[pair] = ChainLink(index, dimension,
                                              part if dimension.count > 1 else 0)
        else:
            chain.spans.append((index, a, b, dimension))
    return chain


# ---- 构件规格 ----------------------------------------------------------------

_H_PATTERN = re.compile(r"(?:B?H[NWMTP]?)-?(\d+(?:\.\d+)?)×(\d+(?:\.\d+)?)"
                        r"×(\d+(?:\.\d+)?)×(\d+(?:\.\d+)?)")
_CIRCLE_PREFIX = "(?:Φ|φ|ϕ|Ø|ø|D|P)"
_TUBE_PATTERN = re.compile(_CIRCLE_PREFIX + r"(\d+(?:\.\d+)?)×(\d+(?:\.\d+)?)")
_ROD_PATTERN = re.compile(_CIRCLE_PREFIX + r"(\d+(?:\.\d+)?)")
_RECT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)×(\d+(?:\.\d+)?)")
_BOX_PREFIXES = ("□", "B", "箱", "RHS", "SHS")

DESIGNATION_HELP = ("可识别的规格：H 型钢 HN400×200×8×13（高×宽×腹板厚×翼缘厚）、"
                    "圆管 Φ219×8（外径×壁厚）、实心圆 Φ50、矩形 300×600（宽×高）。")


def normalize_designation(text: str) -> str:
    value = text.translate(_FULL_WIDTH).upper()
    value = value.replace("X", "×").replace("*", "×")
    return re.sub(r"\s+", "", value)


def parse_designation(text: Any) -> dict[str, Any]:
    """规格文字 → 截面轮廓（尺寸仍是图纸长度单位，换算由组装负责）。

    返回 ``{"name", "type", "dimensions", "convention"}``；``type`` 与
    ``section_geometry.TYPES`` 一致。``convention`` 非空时表示这条解析依赖
    制图约定（如矩形 b×h），界面要求用户确认。读不懂抛 ValueError 并说明支持
    的写法——不按相近规格猜。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("规格为空。" + DESIGNATION_HELP)
    name = normalize_designation(text)
    match = _H_PATTERN.fullmatch(name)
    if match:
        h, b, tw, tf = (float(v) for v in match.groups())
        return {"name": name, "type": "i", "convention": "",
                "dimensions": {"h": h, "b": b, "tw": tw, "tf": tf}}
    if name.startswith(_BOX_PREFIXES):
        raise ValueError(f"「{text}」是矩形管/箱形规格，本版图纸识别尚不能换算；"
                         "请改写为可识别规格，或导入后用「创建截面」建立矩形管。")
    match = _TUBE_PATTERN.fullmatch(name)
    if match:
        d, t = (float(v) for v in match.groups())
        return {"name": name, "type": "tube", "convention": "",
                "dimensions": {"d": d, "t": t}}
    match = _ROD_PATTERN.fullmatch(name)
    if match:
        return {"name": name, "type": "circle", "convention": "",
                "dimensions": {"d": float(match.group(1))}}
    match = _RECT_PATTERN.fullmatch(name)
    if match:
        b, h = (float(v) for v in match.groups())
        return {"name": name, "type": "rectangle",
                "convention": "按平法 b×h（宽×高）理解",
                "dimensions": {"b": b, "h": h}}
    raise ValueError(f"无法解析规格「{text}」。" + DESIGNATION_HELP)


# ---- 材料 --------------------------------------------------------------------

#: 钢材弹性模量与泊松比，GB 50017-2017 表 4.4.8（各牌号相同）。
STEEL_GRADES = ("Q235", "Q345", "Q355", "Q390", "Q420", "Q460")
STEEL_E = 206.0e9
STEEL_NU = 0.3
STEEL_BASIS = "GB 50017-2017 表 4.4.8"

#: 混凝土弹性模量（Pa），GB 50010-2010（2015 年版）表 4.1.5；泊松比按 4.1.8 取 0.2。
CONCRETE_E = {
    "C15": 2.20e10, "C20": 2.55e10, "C25": 2.80e10, "C30": 3.00e10,
    "C35": 3.15e10, "C40": 3.25e10, "C45": 3.35e10, "C50": 3.45e10,
    "C55": 3.55e10, "C60": 3.60e10, "C65": 3.65e10, "C70": 3.70e10,
    "C75": 3.75e10, "C80": 3.80e10,
}
CONCRETE_NU = 0.2
CONCRETE_BASIS = "GB 50010-2010（2015 年版）表 4.1.5、4.1.8"

_GRADE = re.compile(r"(Q\d{3}|C\d{2})")
_EXPLICIT_E = re.compile(r"E\s*=\s*([0-9.eE+-]+)\s*(GPa|MPa|Pa)", re.IGNORECASE)
_EXPLICIT_NU = re.compile(r"(?:ν|nu|μ)\s*=\s*([0-9.]+)", re.IGNORECASE)
_PRESSURE = {"pa": 1.0, "mpa": 1.0e6, "gpa": 1.0e9}

MATERIAL_HELP = ("请填写牌号（Q235/Q345/Q355/Q390/Q420/Q460 或 C15～C80），"
                 "或直接写 E 与泊松比，如「E=206000MPa, ν=0.3」。")


def parse_material(text: Any) -> dict[str, Any]:
    """材料说明 → ``{"name", "E", "nu", "basis"}``（E 单位 Pa）。

    只认两种证据：规范表里的牌号，或文字里直接写明的 E 与泊松比。
    写了牌号但表里没有、或只写「钢材」，抛 ValueError——弹性模量不估算。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("没有材料说明。" + MATERIAL_HELP)
    value = text.translate(_FULL_WIDTH)
    e_match, nu_match = _EXPLICIT_E.search(value), _EXPLICIT_NU.search(value)
    if e_match and nu_match:
        modulus = float(e_match.group(1)) * _PRESSURE[e_match.group(2).lower()]
        nu = float(nu_match.group(1))
        if not (math.isfinite(modulus) and modulus > 0 and 0 <= nu <= 0.5):
            raise ValueError(f"材料参数不合理：E={modulus:g} Pa、ν={nu:g}。")
        return {"name": "用户材料", "E": modulus, "nu": nu, "basis": "用户填写"}
    grade = _GRADE.search(value.upper())
    if grade:
        name = grade.group(1)
        if name in STEEL_GRADES:
            return {"name": name, "E": STEEL_E, "nu": STEEL_NU, "basis": STEEL_BASIS}
        if name in CONCRETE_E:
            return {"name": name, "E": CONCRETE_E[name], "nu": CONCRETE_NU,
                    "basis": CONCRETE_BASIS}
        raise ValueError(f"牌号 {name} 不在内置规范表中。" + MATERIAL_HELP)
    raise ValueError(f"材料说明「{text}」没有牌号或弹性模量。" + MATERIAL_HELP)


# ---- 结构校验（给大模型的修复回路用） ------------------------------------------

def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source_errors(where: str, source: Any) -> list[str]:
    if source is None:
        return []
    if not isinstance(source, Mapping):
        return [f"{where}.source 必须是对象"]
    if source.get("user"):
        return []
    handles = source.get("handles")
    if handles is not None and not (isinstance(handles, list) and handles
                                    and all(_is_text(h) for h in handles)):
        return [f"{where}.source.handles 必须是非空的实体句柄数组"]
    bbox = source.get("bbox")
    if bbox is not None and not (
            isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and 0 <= float(v) <= 1 for v in bbox)):
        return [f"{where}.source.bbox 必须是 4 个 0~1 之间的数"]
    return []


def validate_evidence(evidence: Any) -> list[str]:
    """检查证据的**形状**：字段齐全、类型对、引用写法对。

    数值读不读得懂、轴网闭不闭合不在这里查（那是组装的事，并会进核对表）。
    这里的错误说明大模型没按格式输出，适合回喂让它重写。
    """
    if not isinstance(evidence, Mapping):
        return ["证据必须是 JSON 对象"]
    errors: list[str] = []
    grids = evidence.get("grids")
    if not isinstance(grids, Mapping):
        errors.append("缺少 grids 对象（x 为数字轴、y 为字母轴）")
        grids = {}
    for family in ("x", "y"):
        grid = grids.get(family) or {}
        if not isinstance(grid, Mapping):
            errors.append(f"grids.{family} 必须是对象")
            continue
        labels = []
        for index, axis in enumerate(grid.get("axes") or []):
            where = f"grids.{family}.axes[{index}]"
            if not isinstance(axis, Mapping) or not _is_text(axis.get("label")):
                errors.append(f"{where} 缺少 label")
                continue
            labels.append(str(axis["label"]).strip())
            errors += _source_errors(where, axis.get("source"))
        if len(labels) != len(set(labels)):
            errors.append(f"grids.{family}.axes 轴号重复")
        for index, item in enumerate(grid.get("spacings") or []):
            where = f"grids.{family}.spacings[{index}]"
            if not isinstance(item, Mapping) or not all(
                    _is_text(item.get(key)) for key in ("from", "to")):
                errors.append(f"{where} 需要 from、to 轴号")
                continue
            if "text" not in item:
                errors.append(f"{where} 缺少 text（照抄图上数字）")
            errors += _source_errors(where, item.get("source"))
        edges = grid.get("edges") or []
        if not isinstance(edges, list) or not all(
                isinstance(item, Mapping) and "text" in item for item in edges):
            errors.append(f"grids.{family}.edges 必须是数组，每项照抄一段外边距 text")
    levels = evidence.get("levels") or []
    if not isinstance(levels, list):
        errors.append("levels 必须是数组")
        levels = []
    level_names = set()
    for index, level in enumerate(levels):
        if not isinstance(level, Mapping) or not _is_text(level.get("name")):
            errors.append(f"levels[{index}] 需要 name（图上的层名或标高原文）")
            continue
        level_names.add(str(level["name"]).strip())
    heights = evidence.get("level_heights") or []
    if not isinstance(heights, list):
        errors.append("level_heights 必须是数组")
        heights = []
    for index, item in enumerate(heights):
        where = f"level_heights[{index}]"
        if not isinstance(item, Mapping) or not all(
                _is_text(item.get(key)) for key in ("from", "to")) or "text" not in item:
            errors.append(f"{where} 需要 from、to 层名与 text（照抄图上层高）")
            continue
        unknown = [str(item[key]).strip() for key in ("from", "to")
                   if str(item[key]).strip() not in level_names]
        if unknown:
            errors.append(f"{where} 引用的层名 {'、'.join(unknown)} 须先在 levels 里列出")
        errors += _source_errors(where, item.get("source"))
    members = evidence.get("members") or []
    if not isinstance(members, list):
        errors.append("members 必须是数组")
        members = []
    for index, member in enumerate(members):
        where = f"members[{index}]"
        if not isinstance(member, Mapping):
            errors.append(f"{where} 必须是对象")
            continue
        kind = member.get("kind")
        if kind == "beam":
            if not (_is_text(member.get("from")) and _is_text(member.get("to"))):
                errors.append(f"{where} 梁需要 from、to 轴号")
            if not isinstance(member.get("levels"), list) or not member.get("levels"):
                errors.append(f"{where} 梁需要 levels（所在标高名称数组）")
        elif kind == "column":
            at = member.get("at")
            if not isinstance(at, list) or not 1 <= len(at) <= 2 \
                    or not all(_is_text(v) for v in at):
                errors.append(f"{where} 柱需要 at（所在轴线交点，如 [\"1\", \"A\"]）")
            if not (_is_text(member.get("from_level")) and _is_text(member.get("to_level"))):
                errors.append(f"{where} 柱需要 from_level 与 to_level")
        else:
            errors.append(f"{where}.kind 只能是 beam 或 column")
        errors += _source_errors(where, member.get("source"))
    sections = evidence.get("sections") or []
    if not isinstance(sections, list):
        errors.append("sections 必须是数组")
        sections = []
    for index, item in enumerate(sections):
        if not isinstance(item, Mapping) or not _is_text(item.get("label")) \
                or "text" not in item:
            errors.append(f"sections[{index}] 需要 label 与 text")
    support = evidence.get("support")
    if support is not None and (not isinstance(support, Mapping) or support.get("kind")
                                not in (None, "fixed", "pinned")):
        errors.append("support.kind 只能是 fixed、pinned 或 null")
    for index, layer in enumerate(evidence.get("layers") or []):
        if not isinstance(layer, Mapping) or not _is_text(layer.get("name")) \
                or layer.get("role") not in (None, *LAYER_ROLES):
            errors.append(f"layers[{index}] 需要 name，role 只能是 "
                          f"{'、'.join(LAYER_ROLES)} 或 null")
    return errors
