# 独立混合荷载图首次试点 — 2026-10-09

本图未用于本轮提示或像素算法调试。源码快照、请求参数和独立目视标注在首次调用前固定；
一次调用后不针对本图调整算法，不覆盖首响应。它只是一张独立试点图，不能代表独立验证集的
覆盖程度或通用准确率，也没有真实数值尺寸/荷载，不能测数值准确率。

本图由编码助手直接查看原图、独立整理对象和坐标，未经过外部人工复核。冻结记录中的
verified 表示本轮标注已逐项检查后固定，不等同于外部人工验收；本图只用于试点评分。

图像：**A beam with various loads**，来源作者按文件页版权声明归于 **Sjhan81**。
[原始文件页](https://commons.wikimedia.org/wiki/File:ABeamWithVariousLoads.jpg)；
[核对的页面版本](https://commons.wikimedia.org/w/index.php?title=File:ABeamWithVariousLoads.jpg&oldid=773915260)。
采用来源页提供的 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) 许可；
JPEG 原图未经修改。本目录的标注也按 CC BY-SA 3.0 提供；代码的项目许可不替代图像许可。

首次在线解析通过，但只识别两个端节点，遗漏斜向集中力与集中矩位置，严格像素阈值下
节点、杆件、支座 F1 均为 0，问题召回率 0.250。失败仍计入分母；不把“进入审核”记作
几何正确。冻结结果见 [report.json](report.json)，源码快照见
[pipeline_snapshot.json](pipeline_snapshot.json)。

本图与原五张开发图分别统计，不能合并后只展示较高成绩。其他公开图片遇到下载限流/拒绝访问，
未获取的图片没有进入试点，不生成或宣称其评测结果。下一步需要带数值尺寸的许可明确 CAD 图、
真实手绘/手机照片，并留出更多独立样本。
