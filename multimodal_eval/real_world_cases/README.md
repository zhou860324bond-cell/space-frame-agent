# 真实图片验收区

这里与 `multimodal_eval/images` 的合成夹具严格分开，初始状态不包含任何真实
图片，也不宣称真实准确率。

1. 将已脱敏、获准用于本地评测的 PNG/JPG/WebP 放进 `images/`；文件名主干即
   `image_id`，同一主干不能有多个扩展名。
2. 执行 `python -m multimodal_eval.real_world_eval init`，生成缺失的 metadata 和
   ground-truth 模板。该命令不会覆盖已有标注。
3. 人工独立标注 `ground_truth/*.json`，复核后把 `annotation_status` 改为
   `verified`；确认授权后把 `metadata/*.json` 的 `consent_to_evaluate` 改为 true。
4. 配置提供商环境变量后执行
   `python -m multimodal_eval.real_world_eval recognize --provider openai`。已有响应
   不会被覆盖，避免在看过真值后悄悄重跑挑结果。
5. 执行 `freeze` 绑定全部文件哈希，再执行 `evaluate`。只要任一图片缺少授权、
   已复核真值或响应，`freeze` 就会失败并列出原因。

建议首批 5–10 张，覆盖清晰 CAD 截图、手机拍纸稿、倾斜/阴影、模糊支座、荷载与
无尺寸图。图片可能含项目敏感信息；图片、标注、响应和生成的真实集 manifest 已在
`.gitignore` 中默认排除，只保留空目录骨架。
