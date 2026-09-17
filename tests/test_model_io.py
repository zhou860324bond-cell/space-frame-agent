

# --------------------------------------- 校验信息是给人看的，不是给 jsonschema 看的

def test_schema_errors_are_written_for_a_human():
    """结构层的报错必须是中文，而且说的是工程语言。

    `validate_payload` 的清单最初只写给大模型，后来被原样接到桌面端的
    「3 校验」面板上。于是一个中文界面、面向结构力学学生的软件，在用户
    **最需要帮助的错误路径上**印出来的是：

        [结构] (根): 'materials' is a required property

    实测就是这样：参数化建完几何点开校验，三条全是 jsonschema 原文。
    学生要看到的是"还没有定义材料"。
    """
    from model_io import validate_payload

    only_geometry = {
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "B", "material": "S"}],
        "supports": [],
    }
    errors = validate_payload(only_geometry)
    joined = "\n".join(errors)

    assert "is a required property" not in joined, "不许把 jsonschema 原文甩给用户"
    assert "should be non-empty" not in joined
    assert "材料" in joined and "截面" in joined and "支座" in joined
    assert "(根)" not in joined, "'(根)' 不是给人看的说法"
    assert "模型根层" in joined


def test_unrecognised_schema_errors_keep_the_original_text():
    """认不出的模式要原样保留原文——排障宁可啰嗦，不能少信息。

    把没见过的报错翻成一句笼统的"格式不对"，比不翻更糟：用户和开发者都
    失去了唯一一条线索。
    """
    from model_io import _humanize_schema_error

    class Odd:
        message = "some validator nobody mapped yet"
        validator = "patternProperties"
        validator_value = None
        absolute_path: list = []

    assert _humanize_schema_error(Odd()) == Odd.message


def test_recognised_errors_still_carry_the_original_in_brackets():
    """能翻的也要把原文附在括号里，方便照着搜文档或提 issue。"""
    from model_io import validate_payload

    bad_type = {
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "B", "material": "S"}],
        "materials": [{"name": "S", "E": "很大", "nu": 0.3}],
        "sections": [{"name": "B", "A": 0.01, "Iy": 4e-5,
                      "Iz": 3e-4, "J": 8e-7}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
    }
    hit = [e for e in validate_payload(bad_type) if "类型不对" in e]
    assert hit, "弹性模量填文字应当报类型不对"
    assert "数字" in hit[0], "要说清这里要的是什么"
    assert "原文：" in hit[0], "原文要留着"
