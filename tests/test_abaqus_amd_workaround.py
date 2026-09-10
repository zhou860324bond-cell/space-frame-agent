"""AMD Zen 4 上跑 Abaqus 6.14 需要的环境变量。

实测（AMD Ryzen 9 7940HX / Abaqus 6.14-1）：同一个输入文件，不设
``MKL_DEBUG_CPU_TYPE=5`` 时 standard.exe 以系统错误码 1073741795 中止——
``.dat`` 里没有一条 ***ERROR、``.sta`` 根本不生成；设上就
``THE ANALYSIS HAS COMPLETED SUCCESSFULLY``。

对照实验排除了另外三种解释：

* 不是网格质量——单元全等、畸变为零、规模相当（10125 自由度）的规则网格同样崩
* 不是内存——``memory=4gb`` 无效
* 不是 AVX 本身——``MKL_ENABLE_INSTRUCTIONS=SSE4_2`` 无效

这类"救活了一台机器"的开关最容易在重构里被顺手删掉，所以钉死在这里。
"""

from __future__ import annotations

import abaqus_backend as AB


AMD = {"PROCESSOR_IDENTIFIER": "AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD"}
INTEL = {"PROCESSOR_IDENTIFIER": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel"}


def test_amd_gets_the_mkl_workaround():
    """没有它，这台机器上任何上万自由度的实体作业都会静默崩溃。"""
    assert AB.solver_environment(AMD)["MKL_DEBUG_CPU_TYPE"] == "5"


def test_intel_does_not_get_it():
    """在 Intel 的 AVX-512 机器上强制走 AVX2 只会变慢，没有收益。"""
    assert "MKL_DEBUG_CPU_TYPE" not in AB.solver_environment(INTEL)


def test_a_value_the_user_already_set_is_not_overwritten():
    """用户自己设过就有他的理由，不替他做主。"""
    env = dict(AMD, MKL_DEBUG_CPU_TYPE="4")
    assert AB.solver_environment(env)["MKL_DEBUG_CPU_TYPE"] == "4"


def test_the_host_python_path_is_stripped():
    """Abaqus 自带 Python 2.7，继承宿主的 PYTHONPATH 会让它去 import
    我们的模块，行为不可预料。"""
    env = AB.solver_environment(dict(AMD, PYTHONPATH="/x", PYTHONHOME="/y"))
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env


def test_the_vendor_is_read_from_the_injected_environment():
    assert AB.is_amd_cpu(AMD)
    assert not AB.is_amd_cpu(INTEL)


def test_every_place_that_launches_abaqus_passes_an_environment():
    """三个调用点各写一份环境变量，迟早会漏一个——而漏掉的那个会静默崩溃。

    用 AST 判断而不是搜文本：环境变量常常在前几行先赋值给一个局部名，
    按行数开窗口会误判。这里只问一件事——每个 subprocess.run 是不是都
    显式传了 env。
    """
    import ast
    import inspect

    import run_jobs
    import solid_joint

    checked = 0
    for module in (AB, run_jobs, solid_joint):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (f"{getattr(func.value, 'id', '')}.{func.attr}"
                    if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
            if name != "subprocess.run":
                continue
            checked += 1
            keywords = {kw.arg for kw in node.keywords}
            assert "env" in keywords, (
                f"{module.__name__} 第 {node.lineno} 行的 subprocess.run 没传 env；"
                "AMD 上这会让作业静默崩溃")
    assert checked >= 3, f"只找到 {checked} 个 subprocess.run，调用点数量对不上"


def test_the_environment_actually_carries_the_workaround_on_this_kind_of_box():
    """光传 env 不够——传进去的必须真的带着那个开关。"""
    assert AB.solver_environment(AMD).get("MKL_DEBUG_CPU_TYPE") == "5"


# ------------------------------------------------- 传给求解器进程

def test_the_env_file_is_written_for_amd(tmp_path):
    """光设子进程的环境变量不够。从 CAE 里 job.submit() 提交时，
    standard.exe **不是 CAE 的直接子进程**——驱动会另起一个。

    实测：直接 abaqus job= 带环境变量能跑通；同样的输入换成 CAE 提交
    仍以 1073741795 中止。abaqus_v6.env 是驱动每次调用都会执行的文件。
    """
    path = AB.write_env_file(tmp_path, AMD)
    assert path is not None and path.name == "abaqus_v6.env"
    text = path.read_text(encoding="ascii")
    assert "MKL_DEBUG_CPU_TYPE" in text and "'5'" in text
    compile(text, "abaqus_v6.env", "exec")     # 驱动会执行它，语法必须对


def test_no_env_file_on_intel(tmp_path):
    """Intel 上既不需要，也不该在用户目录里留下多余文件。"""
    assert AB.write_env_file(tmp_path, INTEL) is None
    assert not (tmp_path / "abaqus_v6.env").exists()


def test_no_env_file_when_the_user_already_set_the_variable(tmp_path):
    assert AB.write_env_file(tmp_path, dict(AMD, MKL_DEBUG_CPU_TYPE="4")) is None


def test_the_generated_cae_script_sets_it_too_on_amd():
    """两条机制并存：驱动认哪一条，总有一条能过去。"""
    import solid_joint as sj

    from test_solid_joint import _l_joint

    spec = sj.prepare_joint_spec(_l_joint(), node_id=2)
    on_amd = sj._script_text(spec, [8.0, 6.4, 5.2], "out.json", amd=True)
    on_intel = sj._script_text(spec, [8.0, 6.4, 5.2], "out.json", amd=False)
    assert "MKL_DEBUG_CPU_TYPE" in on_amd
    assert "MKL_DEBUG_CPU_TYPE" not in on_intel, "Intel 上强制 AVX2 只会变慢"
    compile(on_amd, "build_joint.py", "exec")
    compile(on_intel, "build_joint.py", "exec")


def test_the_solid_joint_run_writes_the_env_file_before_launching():
    """顺序错了等于没写：必须在起 Abaqus **之前**落盘。"""
    import ast
    import inspect

    import solid_joint

    import textwrap

    source = textwrap.dedent(
        inspect.getsource(solid_joint.run_joint_analysis))
    tree = ast.parse(source)
    write_line = launch_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", ""))
            if name == "write_env_file":
                write_line = node.lineno
            elif name == "run" and isinstance(func, ast.Attribute):
                launch_line = node.lineno
    assert write_line is not None, "根本没写 abaqus_v6.env"
    assert launch_line is not None and write_line < launch_line
