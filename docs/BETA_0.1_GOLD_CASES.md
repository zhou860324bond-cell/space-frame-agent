# Beta 0.1 金标准算例

本清单冻结线性静力主链路的 12 个独立解析真值。运行：

```bat
run_gold.bat
```

判定原则：真值来自经典梁理论和静力平衡，不来自另一份本项目实现；不得为通过测试放宽阈值。

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

这 12 项覆盖强/弱轴弯曲、扭转、剪切变形、均布/三角/集中荷载、杆端释放、反力、位移、内力及极值位置。自动剖分和 Result DB 属于数据链路不变量，由 `tests/test_model_compiler.py` 与 `tests/test_result_db.py` 单独验证。
