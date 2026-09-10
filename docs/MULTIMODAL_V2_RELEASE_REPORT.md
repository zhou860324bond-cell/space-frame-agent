# 多模态 V2 发布证据

日期：2026-09-09

## 结论

MM2-00 至 MM2-08 的确定性链路及 V2 桌面最终接线已完成。`multimodal_eval/manifest.json` 是唯一离线门禁输入，
固定绑定 30 张项目原创 CC0-1.0 图片、ground truth、脱敏响应 fixture，以及 provider、model、
prompt 和 schema 指纹。全量测试收集 1258 项：1257 项通过，1 项按环境条件跳过。

桌面闭环现覆盖：未知尺度参考杆件标定、问题逐项处理、节点拖动、杆件补删、支座与命名节点荷载
编辑、向空/已有模型预演、重合节点显式复用、原子加载和单步撤销。正式提交仍不会自动求解。

## 离线门禁结果

冻结 fixture 回放结果见 `multimodal_eval/release_report.json`：节点、PhysicalMember、清晰子集、
交点决策、支座、荷载、荷载数值/单位、尺度和问题召回均为 1.0，问题误报为 0，全部发布门槛通过。
负向测试会移除所有预测节点并重新绑定响应哈希，门禁随即因节点和杆件 F1 不达标而失败。

这些分数衡量的是固定响应的确定性回放与评分器正确性，不是在线视觉模型的实测准确率。

## 真实图片验收

新增 `multimodal_eval/real_world_eval.py` 和独立的 `real_world_cases/` 私有数据区。流程会为
图片生成标注与授权模板、批量冻结 provider 响应，并且只有“明确授权 + 人工复核真值 +
有效响应”三者齐全时才允许生成 manifest。图片尺寸、image_id 及 provider/model/prompt/schema
指纹均参与检查，冻结后图片、真值和响应均由 SHA-256 防漂移。

当前真实集为 0 张，因此真实模型准确率仍为“未测”，不会用合成图或测试截图补数。真实图片、
标注、响应和 manifest 已默认加入 `.gitignore`，避免误提交私有工程资料。

## 在线 smoke

`multimodal_eval/provider_smoke.json` 明确排除在离线门禁之外。本次环境没有配置
`OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY`，因此三家均记录为
`SKIPPED_NO_CREDENTIALS`；没有发起网络调用，也没有消耗外部额度。

## 复现

```text
python multimodal_eval/generate_seed_assets.py
python -m multimodal_eval.evaluate --output multimodal_eval/release_report.json
python -m multimodal_eval.real_world_eval status
python -m pytest tests/test_multimodal_eval.py tests/test_multimodal_contract.py
```
