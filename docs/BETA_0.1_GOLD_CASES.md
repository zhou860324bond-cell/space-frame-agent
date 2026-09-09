# 金标准算例

本清单冻结 **28 个**独立解析真值。运行：

```bat
run_gold.bat
```

判定原则：真值来自经典梁理论、静力平衡和闭合解特征值，**不来自另一份本项目实现**；
不得为通过测试放宽阈值。

覆盖范围已从线性静力扩展到模态、屈曲、P-Δ 二阶弹性与轴向塑性。

## 一、线性静力（12 项）

| # | pytest 算例 | 独立真值 | 相对阈值 |
|---:|---|---|---:|
| 1 | `test_tip_deflection_matches_pl3_over_3ei` | 悬臂端力挠度 `PL³/(3EI)` | `1e-10` |
| 2 | `test_weak_axis_uses_iy` | 弱轴悬臂挠度 `PL³/(3EIy)` | `1e-10` |
| 3 | `test_torsion_angle` | Saint-Venant 扭转角 `TL/(GJ)` | `1e-10` |
| 4 | `test_simply_supported_reaction` | 简支均布荷载反力 `wL/2` | `1e-9` |
| 5 | `test_fixed_fixed_end_moment_is_wl2_over_12` | 两端固接梁固端弯矩 `wL²/12` | `1e-9` |
| 6 | `test_release_reaction_split_is_five_eighths_three_eighths` | 一端固接一端铰支反力 `5wL/8 : 3wL/8` | `1e-9` |
| 7 | `test_timoshenko_tip_deflection_includes_shear_term` | `PL³/(3EI) + PL/(GAy)` | `1e-11` |
| 8 | `test_uniform_load_deflection_matches_the_closed_form[simple]` | 简支梁 `5wL⁴/(384EI)` | `1e-5` |
| 9 | `test_uniform_load_deflection_matches_the_closed_form[clamped]` | 两端固接梁 `wL⁴/(384EI)` | `1e-5` |
| 10 | `test_uniform_load_deflection_matches_the_closed_form[cantilever]` | 悬臂梁 `wL⁴/(8EI)` | `1e-5` |
| 11 | `test_simply_supported_point_load_at_midspan` | 跨中集中力峰值弯矩 `PL/4`，位置 `L/2` | `1e-9` |
| 12 | `test_simply_supported_triangular_load_peak` | 三角荷载峰值 `wL²/(9√3)`，位置 `L/√3` | `1e-5` |

## 二、自振特性（5 项）

| # | pytest 算例 | 独立真值 | 相对阈值 |
|---:|---|---|---:|
| 13 | `test_fundamental_frequency_matches_the_closed_form[cantilever]` | `f₁ = (β₁L)²/(2πL²)·√(EI/ρA)`，β₁L = 1.87510407 | `1e-4` |
| 14 | `test_fundamental_frequency_matches_the_closed_form[simple]` | 同上，β₁L = π | `1e-4` |
| 15 | `test_fundamental_frequency_matches_the_closed_form[clamped]` | 同上，β₁L = 4.73004074 | `1e-4` |
| 16 | `test_torsional_frequency` | 扭转一维解 `(1/4L)·√(GJ/ρIp)` | `2e-3` |
| 17 | `test_axial_frequency` | 轴向一维解 `(1/4L)·√(E/ρ)` | `2e-3` |

一致质量阵**高估**频率，因此第 13~15 项在 16 单元下比对；这是该做法的真实精度。

## 三、屈曲（9 项）

| # | pytest 算例 | 独立真值 | 判据 |
|---:|---|---|---|
| 18~21 | `test_critical_factor_matches_euler[pinned/cantilever/clamped/propped]` | Euler `P_cr = π²EI/(KL)²`，四种支承 | 相对 `1e-4` |
| 22~25 | `test_convergence_is_monotone_from_above[pinned/cantilever/clamped/propped]` | 加密时**从上方**单调逼近 Euler 解 | 单调性 |
| 26 | `test_higher_modes_follow_the_square_of_the_half_wave_number` | 两端铰接各阶之比 `1 : 4 : 9` | 比值 |

第 22~25 项守的是**误差的符号**：一致几何刚度阵必然高估临界荷载，从下方逼近
就说明矩阵写错了。这比任何单点比对都灵敏。

## 四、二阶弹性与材料非线性（2 项）

| # | pytest 算例 | 独立真值 | 判据 |
|---:|---|---|---|
| 27 | `test_pdelta_converges_to_the_exact_second_order_solution` | 悬臂柱顶轴压 + 横力的精确解 `δ = H/(Pk)·(tan kL − kL)`，`k=√(P/EI)` | 8 单元相对 `1e-5`，且加密单调收敛 |
| 28 | `test_bilinear_axial_material_uses_incremental_return_mapping` | 双线性轴向应变 `fy/E + (σ−fy)/(αE)` | 相对 `2e-7` |

第 27 项取 `P/P_cr ≈ 0.43`，二阶效应把挠度放大到 **1.757 倍**——量级足够大，
实现里少算一项躲不过去。实测相对误差：1 单元 2.3e-3、2 单元 1.7e-4、
4 单元 1.1e-5、8 单元 6.8e-7、16 单元 4.3e-8。真值来自二阶微分方程本身，
不是"线性解乘放大系数"的近似。

配套的 `test_pdelta_reduces_to_the_linear_solution_without_axial_force`
（未列入金标准）守的是轴力为零时必须退回线性解。

## 覆盖与边界

这 28 项覆盖强/弱轴弯曲、扭转、剪切变形、均布/三角/集中荷载、杆端释放、
反力、位移、内力及极值位置、自振频率（弯曲/扭转/轴向）、四种支承的屈曲临界荷载
与高阶模态、二阶弹性放大、轴向塑性。

**仍未冻结为金标准**：几何—材料联合非线性、弯曲屈服（需纤维截面）、
剪应力与 von Mises 等效应力（截面契约缺一次矩与壁厚）、动力时程。

自动剖分、Result DB 与编译后重复传力路径检查属于数据链路不变量，
由 `tests/test_model_compiler.py` 与 `tests/test_result_db.py` 单独验证。
