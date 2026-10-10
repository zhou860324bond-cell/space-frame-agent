# 公开工程图首次评测（2026-10-09）

用户选择许可明确的公开工程图；原图不修改，独立人工标注在首次接口调用之前固定。
本组是 5 张公开梁图，包含变形示意、反力箭头和剪力/弯矩辅助图；不是 CAD、手绘及手机照片的完整验收集。

## 首次在线结果

DeepSeek `deepseek-flash`，每张一次调用，`max_repairs=0`。共 5 次调用；
草稿通过 0/5，识别阶段中位耗时 11.031 秒，总耗时 63.249 秒。
4 张因审核对象引用冲突被拒收，1 张返回空内容；未保存接口结束原因，空内容的具体原因尚未确认。
失败预测按空对象计分，原始响应、错误及耗时保留，不追加重试、不从分母删除失败。

## 修复后离线回放

只重新解析同一批首次原文，新增接口调用 0 次；4/5 可进入审核。
沿用唯一明确的支座观察名称，消除自动命名冲突；尺寸观察并入已有尺寸证据，保留未知值及未人工确认状态。
这是解析兼容性的改善，不是修复后重新在线识别的成功率。空响应仍然失败，首次在线成绩未覆盖。

| 样本 | 回放 | 节点 F1 | 杆件 F1 | 支座 F1 | 首次耗时（秒） |
|---|---|---:|---:|---:|---:|
| cantilever | PASS | 0.000 | 0.000 | 0.000 | 10.265 |
| end_load | PASS | 1.000 | 1.000 | 1.000 | 8.266 |
| overhang | PASS | 0.000 | 0.000 | 0.000 | 12.000 |
| point_load | FAIL | 0.000 | 0.000 | 0.000 | 21.687 |
| udl | PASS | 0.000 | 0.000 | 0.000 | 11.031 |

全组回放节点 F1=0.182，杆件 F1=0.154，支座 F1=0.143；
待核实问题召回率=0.778，额外问题 20 条。
匹配阈值维持既有 max(5 像素, 图像对角线的 0.5%)，未为本组调宽；杆件/支座还要求节点匹配。
低分说明即使格式有效，轴线像素位置及结构选择仍未达到该标准；不得将进入审核当成识别正确。

## 证据范围与后续

图中的长度及荷载是符号，没有真实数值。因此荷载数值/单位准确率、尺度准确率为 null；没有交叉判读案例，交叉准确率也为 null。
符号荷载不构成已知数值的求解输入，应保持 load_incomplete 阻断。模型返回的不完整荷载条目属于额外预测，荷载 F1 为 0。
支座真值遵循现有提示词：三角形按铰接、墙体按固接；原图未明确区分滑动的情况写入标注备注，不能据此泛化支座判读能力。
首轮及回放均未通过完整发布评分阈值；本组没有覆盖其要求的全部对象，不能替代 30 张合成夹具的确定性回归。
下一轮优先：区分初始梁轴与变形轮廓、识别并排除剪力/弯矩辅助图及支座反力、减少无结构证据的中点、定位空响应原因。

## 来源与许可

| 样本 | 作者 | 许可 | 来源 |
|---|---|---|---|
| overhang | Davius | CC0-1.0 | [Beam cantilever.png](https://commons.wikimedia.org/wiki/File:Beam_cantilever.png) |
| cantilever | Ben pcc | Public domain | [CantileverBeam.png](https://commons.wikimedia.org/wiki/File:CantileverBeam.png) |
| end_load | Daniel De Leon Martinez; modified by Natus37 | Public domain | [Beam Cantilevered Load end.png](https://commons.wikimedia.org/wiki/File:Beam_Cantilevered_Load_end.png) |
| point_load | Tkn20 | Public domain | [Simply supported beam with central point load.jpg](https://commons.wikimedia.org/wiki/File:Simply_supported_beam_with_central_point_load.jpg) |
| udl | Tkn20 | Public domain | [Simply supported beam with universally distributed load.jpg](https://commons.wikimedia.org/wiki/File:Simply_supported_beam_with_universally_distributed_load.jpg) |

原图于 2026-10-09 获取；源页面许可已逐页检查。`sources.json` 记录下载地址、作者及未修改说明。

## 复现

在项目根目录、安装桌面依赖后执行；设置 PYTHONPATH=src（Windows 下 `set PYTHONPATH=src`）。

```text
python -m multimodal_eval.real_world_eval evaluate --root multimodal_eval/public_cases --output results/public-baseline.json
python -m multimodal_eval.real_world_eval replay --root multimodal_eval/public_cases --output results/public-replay.json
```

上述两条仅使用冻结证据，无需凭据或联网。evaluate 未通过完整阈值时返回 1，报告仍保存；不得把这一退出码解释为未执行评分。
原图、输入参数、独立真值和首次响应均用 SHA-256 绑定，修改后须重新建立明确分开的评测轮次。

## 后续轮次

本文件与已保存的首轮 report.json / replay.json 保留当时测量结果。当前 replay 命令使用
当前代码，其输出可能与历史 replay.json 不同；不会覆盖历史文件。后续两轮首响应及当前
代码回放分别记录在 [分轮对照报告](ROUND_REPORT.md)，不把回放提升记为在线准确率。
