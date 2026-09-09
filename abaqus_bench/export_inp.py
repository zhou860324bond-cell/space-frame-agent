r"""把模型导出成 Abaqus 输入文件（.inp）。

两个关键决定：

**截面用 SECTION=GENERAL。** 直接给 A、I11、I12、I22、J，绕开任何截面形状的
换算差异——两侧用的是同一组截面常数，剩下的差异才归得到单元格式头上。

**局部坐标系显式给出 n1。** Abaqus 梁的局部 1 轴由数据行的方向余弦指定，
局部 2 轴 = t × n1。本程序的局部 y 就取作 Abaqus 的 n1，于是：

    Abaqus 局部 1 ≡ 本程序局部 y   ->  I11 = Iy
    Abaqus 局部 2 ≡ 本程序局部 z   ->  I22 = Iz

这个对应关系无法在本机验证（这里跑不了 Abaqus），所以 models.py 里专门放了
一个强弱轴悬殊的截面：若 I11/I22 对调，误差会从 1e-6 量级跳到百分级，
一眼可辨。对称截面那一例则是对照组。

用法（Python 3，在本项目环境里跑）：

    set PYTHONPATH=src;abaqus_bench
    python abaqus_bench\export_inp.py            生成全部算例的 .inp
    python abaqus_bench\export_inp.py --element B31
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inp_writer import write_inp
from models import all_models


from console import use_utf8   # 见 src/console.py：别让一个字符打死一次成功的运行

use_utf8()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element", default="both", choices=["B33", "B31", "both"])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "inp"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ["B33", "B31"] if args.element == "both" else [args.element]
    written = []
    for model in all_models():
        for kind in kinds:
            p = write_inp(model, kind, out / ("%s_%s.inp" % (model["name"], kind)))
            written.append(p)
            print("写出 %s  （%d 节点 %d 杆件）"
                  % (p.name, len(model["nodes"]), len(model["members"])))
    print("\n共 %d 个输入文件在 %s" % (len(written), out))
    print("接下来在装了 Abaqus 的机器上跑 run_abaqus.bat")


if __name__ == "__main__":
    main()
