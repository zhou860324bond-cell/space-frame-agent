# 与 Abaqus 对标

验证章节的第二层：解析解证明单个公式对，**对标证明整套流程对**。

## 为什么用 B33，而不是 B31

| 单元 | 格式 | 与本程序的关系 |
|---|---|---|
| **B33** | 三次梁，不计横向剪切 | **同为 Euler-Bernoulli，苹果对苹果** |
| B31 | 一点缩减积分 Timoshenko 梁 | 含剪切变形与细长度补偿，存在格式差异 |

**主对标用 B33**，误差应落在 1e-8 ~ 1e-12 的数值精度量级。B31 也跑一遍，
它会留下系统性偏差——细长构件下约 1e-3，粗短构件下可达 1e-2。

**这个偏差不是误差，是两种梁理论的真实差别。** 报告里把两组结果并排放，
再解释差异来源，比只报一组"误差很小"有说服力得多。答辩时如果被问
"你怎么知道差异是格式引起的而不是你算错了"，这一对比就是答案。

## 五个算例

| 算例 | 考什么 |
|---|---|
| `cantilever_strong_axis` | 端部竖向力，强轴弯曲 `PL³/3EI_z` |
| `cantilever_weak_axis` | 同一根梁改侧向力，弱轴弯曲 `PL³/3EI_y` |
| `torsion_bar` | 纯扭转 `TL/GJ`，与弯曲完全解耦 |
| `portal_frame` | 均布荷载 + 水平力，考等效节点荷载与内力回算 |
| `space_frame` | 两跨一开间两层，考竖直杆件的局部坐标系与三维耦合 |

前两例故意用了**强弱轴悬殊的截面**（`Iy = 2e-5`，`Iz = 4e-4`，相差 20 倍）。
截面主轴方向是空间刚架最容易搞反的地方，用对称截面根本测不出来——一旦
`I11/I22` 对调，这两例的误差会从 1e-6 量级跳到百分级，一眼可辨。
`torsion_bar` 用对称截面作对照组。

## 三个关键的建模决定

**截面用 `SECTION=GENERAL`。** 直接给 A、I11、I12、I22、J，绕开任何截面形状
换算带来的差异。两侧用的是同一组截面常数，剩下的差异才归得到单元格式头上。

**局部坐标系显式写出 n1。** Abaqus 梁的局部 1 轴由数据行的方向余弦指定，
局部 2 轴 = t × n1。本程序的局部 y 取作 Abaqus 的 n1，于是
`I11 = Iy`、`I22 = Iz`。竖直杆件走参考向量退化分支，n1 = 全局 X，与内核一致。

**同一份模型定义。** `models.py` 里的模型既喂给本程序求解，也用来生成 `.inp`，
不存在"两边建的不是同一个结构"的可能。

## 怎么跑

**第一步，生成输入文件**（在本项目环境，Python 3）：

```
set PYTHONPATH=src;abaqus_bench
python abaqus_bench\export_inp.py
```

写出 10 个 `.inp` 到 `abaqus_bench\inp\`（5 个算例 × 2 种单元）。

**第二步，跑 Abaqus。** 双击 `abaqus_bench\run_abaqus.bat`。如果提示找不到
`abaqus` 命令，从开始菜单打开 **Abaqus Command** 窗口，在那里执行该脚本。
它会依次提交所有作业，然后调用 `abaqus python extract_odb.py` 把结果导成 CSV。

**第三步，对比**（回到本项目环境）：

```
python abaqus_bench\compare.py                跑 B33
python abaqus_bench\compare.py --element B31  跑 B31
```

报告写到 `abaqus_bench\benchmark_B33.md` 和 `benchmark_B31.md`，可直接进论文。

## 不需要 Abaqus 也能验的部分

```
python abaqus_bench\compare.py --selftest
```

拿本程序自己的结果冒充参考解，误差必须恒为 0；再把某个分量整体放大 1%，
误差必须精确等于 `0.01/1.01`。**对比逻辑本身先被验过，对标结果才可信。**

`tests/test_benchmark.py` 里另有 20 个用例守着 `.inp` 的内容正确性——
节点与单元齐全、全文保持 ASCII、截面行的 `I11/I22` 映射、竖直杆件的 n1。

## 误差定义

```
e_X = sqrt( Σ (X_本程序,i − X_Abaqus,i)² ) / sqrt( Σ X_Abaqus,i² )
```

逐分量算 U1/U2/U3/UR1/UR2/UR3 与反力 RF1/RF2/RF3，再给位移模长的整体误差。
参考解整体为零的分量记作 `—`，在那里归一化没有意义。

## Abaqus 6.14 的注意事项

- 自带 **Python 2.7**，`extract_odb.py` 因此用 Python 2 语法写，必须用
  `abaqus python` 执行，不能用系统的 Python 3
- 路径全用英文、不带空格
- `.inp` 全程保持 ASCII
- 6.14 打不开更高版本生成的 `.odb`，反之可以
