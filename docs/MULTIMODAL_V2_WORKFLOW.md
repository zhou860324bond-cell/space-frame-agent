# 多模态建模 v2：实施工作流

> 状态：实施规范草案（2026-09-08）  
> 受众：项目负责人、后续开发 Agent、测试与验收人员  
> 目标：把结构草图稳定地转换成**经人工确认的物理模型草稿**，而不是让视觉模型直接生成可求解模型。

## 1. 本轮边界

### 必须完成

- 支持导入截图、扫描图或近似正射拍摄的二维梁柱结构草图。明显透视畸变属于阻断问题，
  本轮要求重新拍摄或提供已矫正图片，不能让用户把畸变几何当作精确模型提交。
- 用户必须明确选择 `XY / XZ / YZ` 工作平面；默认可预选 `XZ`，但提交前必须确认。
- 识别节点、物理杆件、支座、节点荷载、杆件荷载和尺寸证据，并保留图片定位与置信度。
- 对未知尺度、尺寸冲突、模糊实体、交叉点连接关系建立统一问题队列。
- 在原图上完成拖点、补点、补杆、删除、交点连接/跨越、支座与荷载修订。
- 提交前展示结构化 diff；只有问题清零且用户明确确认后才写入当前 `Session`。
- 提交后只进入 `draft` 或 `ready` 工作流状态，绝不自动求解。
- 建立固定图片集、缓存识别响应和可重复评分器。

### 本轮不做

- 不从单张二维图片猜完整三维结构、遮挡深度或构件截面方向。
- 不做通用透视重建；只允许用户裁剪、按 90° 旋转，并用已知尺寸标定。
- 不自动补材料、截面、规范参数、未出现的支座或荷载。
- 不修改求解器、Result DB、非线性和云图算法。
- 不把 OCR 文字直接当作可信工程输入；冲突或低置信度内容必须确认。

## 2. 权威状态机

```text
idle
  ↓ 选择图片
image_loaded
  ↓ 选择工作平面并开始识别
recognizing
  ├─用户取消────────→ image_loaded
  ├─解析/修复失败───→ recognition_failed ─→ 重试/更换图片
  └─成功────────────→ review_required
                         ↓ 问题清零、拓扑合法、尺度确认
                     ready_to_commit
                         ↓ 生成 diff + 用户明确确认
                     committing
                         ↓ 原子事务成功
                     committed
                         ↓
                 Agent workflow: draft / ready
```

修改裁剪/旋转会回到 `image_loaded` 并丢弃旧识别草稿；重新识别从
`review_required` 回到 `recognizing`，但在任务成功前保留一份只读旧草稿。任何几何、尺度、工作平面、支座或荷载修改
都会使原确认失效，并从
`ready_to_commit` 返回 `review_required`。识别线程不得直接修改正式 `Session`。

每个状态至少公开以下确定性字段：

```json
{
  "schema": "multimodal-workflow/v2",
  "phase": "review_required",
  "work_plane": "XZ",
  "scale_status": "unknown",
  "counts": {"nodes": 6, "members": 7, "supports": 2, "loads": 1},
  "blocking_issues": 3,
  "warnings": 1,
  "can_commit": false
}
```

状态由 `MultimodalControllerState` 记录与草稿校验器共同计算；controller 在每次载入新图片时
生成 UUID `workflow_instance_id`。`job_status` 只允许
`idle/running`，`commit_status` 只允许 `idle/running`；异步开始/取消/失败、成功提交
分别写入 `job_status/job_id/last_error/commit_status/commit_receipt`，草稿内容仍由校验器判定。界面和大模型
都不能直接改写 `phase` 或 `can_commit`。

每次开始或重试都生成新的 UUID `job_id`。成功、失败和进度回调只有在回调 job_id 等于
controller 当前 job_id 且 `job_status=running` 时才可落状态；取消会先把 job_status 改为 idle、
再清空当前 job_id。任何取消/重试后的迟到响应一律丢弃并只记诊断日志。

识别成功原子替换 draft、清空 last_error，并把 job_status 置 idle；首次识别失败时清空 draft、写
last_error 并进入 `recognition_failed`，重新识别失败时恢复只读旧草稿、记录错误并回到
`review_required`。提交点击采用 compare-and-set：只有 commit_status=idle
才能置 running，重复点击和并发调用直接拒绝。事务成功写 receipt 并回到 idle；失败写提交
错误并回到 idle。

`can_commit=true` 的完整谓词为：`job_status=idle`、`commit_status=idle`、phase 为 `ready_to_commit`、工作平面与尺度
均 confirmed、`model` 已物化、`validation_errors=[]`、阻断 issues 为 0、非空模型碰撞为 0、当前 revision 已生成
merge_plan 与 commit_preview，且 preview 中的 draft/plan/baseline/diff 哈希全部匹配当前数据与
当前 Session 规范哈希。用户点击确认后才生成
`confirmation` 并开始事务；成功回执产生 `committed`，失败回到 `ready_to_commit` 并保留错误。

phase 按以下优先级唯一推导，命中第一条即停止：

1. `commit_status=running` → `committing`。
2. `job_status=running` → `recognizing`。
3. `last_error` 非空且没有可审查草稿 → `recognition_failed`。
4. `commit_receipt` 同时匹配当前 `workflow_instance_id/image_hash/draft_hash/revision`，且其
   `model_hash` 等于当前 Session 哈希 → `committed`。
5. 已载入派生图片但没有草稿 → `image_loaded`。
6. 有草稿，且工作平面/尺度/校验/阻断问题/碰撞任一不满足，或当前 revision 没有新鲜 diff
   → `review_required`。
7. 有草稿且上述提交前条件全部满足 → `ready_to_commit`。
8. 以上均不满足 → `idle`。

取消识别写 `job_status=idle`：首次识别回到 `image_loaded`，重新识别则恢复旧草稿并回到
`review_required`。载入新图片、开始重新识别或编辑草稿都会清空旧 `commit_receipt`；再次编辑
正式模型则使 receipt 的 model hash 与旧 baseline hash 失配，不再显示 `committed`。
`can_commit` 只在第 7 条且基线哈希仍新鲜时为真。

## 3. 用户操作流程

### A. 导入与预处理

1. 拖入、粘贴或选择一张图片。
2. 界面显示原图尺寸、方向和文件类型，不复制覆盖原文件。
3. 用户可旋转 90°、裁剪结构区域；预处理结果保存为派生图并记录来源。
4. 界面**预选** `XZ` 工作平面，用户点击确认后才写成 `work_plane.status=confirmed`。
   图片向右对应第一轴正向，图片向上对应第二轴正向；第三轴取当前工作平面偏移。
5. 点击“开始识别”，界面进入只读等待状态，但允许取消。

### B. 识别与确定性守门

1. 视觉模型只返回 `RecognitionDraft`，不调用建模或求解工具。
2. 每个实体必须携带 `kind / id / confidence / recognition_confidence / image_geometry / source`，
   置信度按 §4 的可空合同校验。
3. 解析后立即执行格式、有限数、重复 ID、悬空引用、零长度和荷载引用校验。
4. 格式错误可把结构化错误回给视觉模型修复，最多两次；仍失败则保留原响应并停止。
5. 校验通过才显示覆盖层；校验不通过的内容不得局部写入正式模型。

### C. 问题队列

问题按阻断优先级排列，用户一次处理一个：

1. **工作平面**：确认预选的 XY、XZ 或 YZ；修改平面会重算全部模型坐标。
2. **尺度与原点**：采用二维各向同性相似变换。用户选择一根识别杆件并输入真实长度，
   或确认一条可靠尺寸；默认锚点为图中“最左、再最低、再最小 ID”的节点并映射到平面
   `(0, 0)`，用户可另选锚点及坐标。单条长度只确定统一比例，不推断透视变形。
3. **尺寸冲突**：各尺寸得到的比例相对中位数相差超过 2% 即阻断。用户必须指定采用的
   尺寸，其余证据标为 `rejected_conflict`，禁止自动平均。
4. **交叉关系**：逐个选择“连接并生成共享物理节点”或“跨越不连接”。连接时保留两根
   PhysicalMember 的原 ID 和端点，不拆物理构件；既有编译器根据共享内节点生成分析单元。
5. **低置信度拓扑**：确认、重画或删除节点/杆件。
6. **边界与荷载**：确认符号类型、作用对象、方向、数值和单位。
7. **缺失工程属性**：只提示后续进入 Property/Load 阶段，不在图片确认页伪造。

警告允许带入草稿；阻断问题必须清零。明确阻断表如下，表外问题默认为警告：

| 问题 | 阻断规则 | 解除方式 |
|---|---|---|
| 工作平面未确认 | 始终阻断 | 用户确认平面 |
| 尺度未知/冲突 | 始终阻断 | 标定或选择可信尺寸 |
| 透视未确认/已拒绝 | `preprocessing.perspective_status != accepted` | 用户确认近似正射，或更换/外部矫正图片 |
| 拓扑非法 | 悬空引用、自连接、零长度、重复杆件 | 编辑至校验通过 |
| 交点未知 | 每个 `unknown` 均阻断 | 选择 connect/cross |
| 低置信度节点/杆件 | `verified=false` 且（`source=vision && confidence < 0.75` 或 `source=migration`） | 确认、重画或删除 |
| 已保留但信息不全的荷载 | 目标、方向、数值或单位任一未知 | 补齐或删除该荷载 |

缺少支座、荷载、材料或截面本身不阻断几何草稿提交，只会使正式 Agent 工作流保持
`draft`。勾选“我已核对”不能替代具体问题的处理记录。

### D. 图上编辑

- 单击实体选中并显示来源、置信度和模型坐标。
- 拖动节点时同步相连杆件覆盖线，重新计算相关尺寸，并使确认失效。
- 补点支持图片坐标和精确模型坐标两种方式。
- 补杆只能连接已有节点；交叉位置是否生成节点必须显式选择。
- 删除被支座或荷载引用的实体前，先展示引用清单并要求处理引用。
- 所有人工修改标记 `source=user`、`verified=true`；保留原
  `recognition_confidence`，当前 `confidence` 置空，后续重新识别不得静默覆盖。

### E. 预演与提交

1. 将已确认识别草稿转换为候选 `FrameDraft`，不触碰当前 Session。
2. 与当前物理模型比较，展示节点、杆件、支座和荷载的新增/删除/修改清单。
3. 如果当前模型非空，默认采用“仅新增”：候选 ID 从当前同类最大 ID 加一开始重映射，
   草稿内部引用同步重映射，既有对象绝不删除或修改。候选节点与既有节点距离小于
   `1e-6 m` 等价值时列为阻断碰撞，用户逐项选择“复用并连接”或“保持独立并平移”。
   “保持独立”不是逐节点移动：用户输入一个平面内平移向量 `[da,db]`，一次作用于**整个候选
   草稿**，应用后 revision 递增、旧碰撞决策清空并重新检查。局部连接只能选择复用既有节点。
   新节点和新杆件 ID 均为正整数：分别按草稿旧 ID 升序，依次映射为当前同类最大 ID+1、+2…；
   所有引用使用同一映射表。复用节点后若形成重复杆件、共线重叠或支座冲突则继续阻断；命名荷载重名时要求改名，
   不静默覆盖。整体替换是另一种高风险模式，必须走 `preview_change/apply_preview`。
   每次复用、平移或改名选择都写入 §4 的 merge_plan；只有由该 plan 确定性生成的 commit_preview
   才能进入确认，界面不得把未入 plan 的临时选择直接传给提交函数。
   预演必须对“应用节点映射与整稿平移后的候选 + baseline”做全量跨集合扫描：候选节点落在既有
   杆件内部、既有节点落在候选杆件内部、以及候选杆件与既有杆件的每个交点都必须进入
   topology_actions。相同关系若已存在于 baseline 不重复提问；所有新增关系必须明确处理并出现在
   diff，禁止依赖编译器扫描后才静默发现连接。
4. 用户点击“确认载入”后才调用正式加载入口；提交时再次核对图片来源、草稿哈希和当前模型哈希。
5. 哈希已变化则拒绝陈旧确认并重新生成 diff。
6. 合并候选必须再次运行 Domain IR 引用校验及空间重合/重叠检查。提交使用深拷贝事务：
   成功时只记录一个 `apply_draft` 历史步骤，并统一清除 `frame/solution/compilation/result_db`；
   任一步失败则四者、model 和 history 全部保持原值。基线哈希直接复用
   `change_preview.model_digest` 的规范 JSON SHA-256。
7. 成功后刷新模型树和视口；随后直接调用现有 `inspect_workflow(Session)` 判定 `draft`
   或 `ready`，多模态模块不得自行判定。系统不自动调用 `solve_model`。

## 4. 数据与追溯合同

v2 顶层必填 `format="space-frame-recognition-draft/v2"`、`image_model`、`model`、`merge_plan`、`source`、
`work_plane`、`scale`、`entities`、`dimensions`、`intersections`、`issues` 和 `revision`。
v1 迁移时保留旧 model 的拓扑/工程数据以及 entities/questions/warnings，但不把旧坐标直接放入
v2 model；缺失的新字段使用未确认状态，并生成对应阻断问题，
不得把旧数据默认成已确认。

v1 的正式坐标不作为已标定结果沿用。若全部节点 entity 都有合法 point，直接据此生成 image_model；
否则对旧 model 的 x/y/z 计算各轴极差，按“极差降序、同值 X/Y/Z”选前两轴，第一轴线性归一化为 u，
第二轴线性归一化后取 `v=1-normalized`，零极差轴统一取 0.5，并对全部节点采用该代理投影。
拓扑、支座和荷载引用按原 ID 复制，v2 model 置 null，平面 proposed、尺度 unknown。缺失 entity
以 `source=migration/confidence=null/recognition_confidence=null/verified=false` 合成，并为代理投影或
合成记录生成 blocking low_confidence issue，用户逐项确认或重画后才可提交。

`work_plane.status` 只能是 `proposed/confirmed`；`plane` 只能是 `XY/XZ/YZ`。v2 草稿及其物化
model 一律使用 `N-m-Pa` 的 SI 内部单位，界面可按项目显示偏好换算，但不得把显示值写入合同。
`scale.status` 只能是 `unknown/confirmed/conflict`；未 confirmed 时 length_per_pixel 为 null，
confirmed 时必须为有限正数；`unit` 固定为 `m/px`，offset 与 anchor_coordinates_xyz 均以米记录。

`image_model` 是提交前始终有效的权威图像拓扑：`nodes[]` 为 `{id,u,v}`，其中 u/v 是有限的派生图
归一化坐标；`members[]` 为 `{id,i,j,material,section}`（后两项允许 null）；`supports[]` 为
`{name,node,fix[6]}`；顶层及 `load_cases[]` 内均使用现有
`nodal_loads/member_loads/member_spans/settlements` 的引用与载荷合同。所有缺省集合必须为空数组，
ID/名称唯一且引用必须命中 image_model。

`image_model` 允许尚未确认的荷载方向、数值或单位为 null，但引用不可悬空，并用
`load_incomplete` 阻断；材料/截面允许 null 且只形成草稿提示。

`model` 在工作平面、尺度未 confirmed 或任一保留荷载不完整时必须为 null；条件满足后由
image_model 一次性确定性物化。该 model 是 `FrameDraft.to_model()` 可直接消费的 Domain IR v1 形状：
`schema_version=1`、`units`、`nodes[{id,x,y,z}]`、
`members[{id,i,j,material,section}]`、supports 和 load_cases。物化后 ID、引用和集合顺序与
image_model 一致，坐标按本节公式计算。image_model 的几何/拓扑/工程字段、尺度、锚点或工作平面
任一变化时都先丢弃 model，再在物化前置条件满足时重新生成整个 model；
在界面输入精确模型坐标时，先用当前相似变换的逆变换更新对应 image_model 节点 u/v，再重新物化。
因此任何时候都不会把像素或无量纲坐标伪装成正式项目长度单位。
材料/截面尚未指定时，对应集合可为空、member 字段写空字符串；这只使 Session 进入既有 `draft`
状态，并不伪造工程属性。补齐后现有严格 Domain IR 校验才允许进入 `ready` 和编译/求解。

| 字段 | 类型与约束 |
|---|---|
| `source` | `{original_path, image_hash, raw_width_px, raw_height_px, derived_path, derived_image_hash, width_px, height_px, preprocessing}`；width/height 专指最终派生图；哈希为 SHA-256 |
| `preprocessing` | `{exif_orientation, rotation_deg, crop_xywh_px, original_to_derived_px, perspective_status}`；`rotation_deg` 只能是顺时针 `0/90/180/270`；crop 是 EXIF 纠正及旋转后坐标系中的整数 `[x,y,w,h]`；矩阵把原始文件像素变为派生图像素；perspective_status 只能是 `unconfirmed/accepted/rejected` |
| `work_plane` | `{status, plane, offset, axis_mapping}`；axis_mapping 结构见下文；offset 固定使用米 |
| `scale` | `{status, length_per_pixel, unit, anchor_node, anchor_coordinates_xyz, evidence_ids}`；类型见下文 |
| `entities[]` | `{kind, id, confidence, recognition_confidence, verified, image_geometry, source, target, payload}`；confidence 字段可空，其他 target/payload 约束见下文 |
| `dimensions[]` | `{id, kind, text, value, unit, image_geometry, target, confidence, recognition_confidence, verified, source, status}`；kind/target 见下文；status 为 `proposed/confirmed/rejected_conflict` |
| `intersections[]` | `{id, members:[a,b], point:[u,v], decision, node_id, member_s}`；详细约束见下文 |
| `issues[]` | `{id, category, severity, entity_refs, message, status, resolution, resolved_by}`；稳定 ID，不因排序变化 |
| `revision` | 非负整数；任何会改变候选模型、映射或确认结论的操作递增 |
| `merge_plan` | `null` 或当前 Session 基线上的完整合并计划，结构见下文 |
| `confirmation` | `null` 或 `{revision, draft_hash, baseline_model_hash, preview_id, diff_hash, confirmed_at}`；编辑后立即清空 |

controller 的 `commit_receipt` 不属于 RecognitionDraft；其结构固定为
`null` 或 `{workflow_instance_id,image_hash,draft_hash,revision,model_hash,committed_at,history_step_id}`，
所有哈希均为 64 位小写十六进制 SHA-256，只有 §2 第 4 条全部相等才代表当前实例已提交。

`merge_plan` 在尚未预演或 baseline 失效时为 null；预演后固定为
`{mode,baseline_model_hash,translation_ab_m,node_actions,topology_actions,member_id_map,support_name_map,load_name_map}`。
mode 只能是 `replace_empty/add_only`；translation 是两个有限米值。node_actions 按草稿节点 ID
升序，元素为 `{draft_id,action,target_id}`，action 只能是 `create/reuse`，target_id 是最终正式节点
ID；member_id_map 是覆盖全部草稿杆件的 `{draft_id_string:final_id}`，其他两个 name_map 分别以
`<draftNode>:<name>` 和 `<case>:<collection>:<name>` 为键，覆盖全部支座/荷载。所有映射值必须命中
既有复用对象或本计划唯一创建的对象；不得遗漏引用，创建 ID 按 §3.E 的排序规则分配。
空 Session 使用 replace_empty，translation 为 `[0,0]`、所有 action 为 create、ID/name 映射为恒等；
非空 Session 只能使用 add_only。

`topology_actions` 覆盖合并相对 baseline 新增的全部跨集合拓扑关系，按
`(kind,candidate_id,existing_id)` 排序。元素为
`{kind,candidate_id,existing_id,decision,node_id}`：kind 是 `member_intersection` 或
`candidate_node_on_existing_member/existing_node_on_candidate_member`。member_intersection 的
decision 可为 `connect/cross`；connect 必须给出最终共享 node_id，cross 必须为 null 且交点处不得
存在正式节点。node_on_member 因现有编译器会自动连接，只允许 `connect/relocate`：connect 记录实际
node_id，relocate 后该几何关系必须已经消失。任何未决项都生成 blocking merge_collision issue。

controller 的 `commit_preview` 固定为
`null` 或 `{preview_id,draft_hash,merge_plan_hash,baseline_model_hash,operations,diff_hash}`；operations
按 `(kind,key,op)` 排序，每项为 `{op,kind,key,before,after}`，op 只能是
`add/reuse/rename/translate/connect/cross`，
before/after 使用 null 或规范对象。`merge_plan_hash` 和 `diff_hash` 分别对 merge_plan、operations
做同一规范 JSON SHA-256。任一草稿编辑、合并选择或 baseline 变化都递增 revision、清空 confirmation
与 commit_preview，并将 merge_plan 置 null 后重新预演。提交只执行 confirmation.preview_id 指向的
不可变 commit_preview；五个 hash/revision 任一不匹配即拒绝，因此同一确认不可能应用到另一合并结果。

`image_geometry` 一律使用派生图像素中心的归一化坐标。派生图必须满足 `W>1,H>1`；像素中心
`x=0..W-1,y=0..H-1` 映射为 `u=x/(W-1),v=y/(H-1)`，所以左上严格是 `[0,0]`、
右下严格是 `[1,1]`。点为 `{point:[u,v]}`，线为 `{line:[[u1,v1],[u2,v2]]}`，区域为
`{bbox:[u0,v0,u1,v1]}`。

`original_to_derived_px` 为 3×3 行优先齐次矩阵，输入/输出均为像素中心坐标，并且是坐标换算的
唯一权威。对原图尺寸 `Wr×Hr`，EXIF orientation 1–8 的像素变换依次为：
`(x,y)`、`(Wr-1-x,y)`、`(Wr-1-x,Hr-1-y)`、`(x,Hr-1-y)`、`(y,x)`、
`(Hr-1-y,x)`、`(Hr-1-y,Wr-1-x)`、`(y,Wr-1-x)`；5–8 会交换输出宽高。
在 EXIF 输出尺寸 `We×He` 上再执行用户指定的顺时针旋转：0° 为 `(x,y)`，90° 为
`(He-1-y,x)`，180° 为 `(We-1-x,He-1-y)`，270° 为 `(y,We-1-x)`；最后减去
crop 的左上角 `(crop_x,crop_y)`。若最终派生图尺寸为 `Wd×Hd`，矩阵输出再分别除以
`Wd-1` 和 `Hd-1` 才得到 `[u,v]`。
实现必须用四角与一个非对称内部点覆盖 8 个 EXIF 方向、4 个旋转及裁剪组合测试。

实体 kind 合同：`node.target={node:id}`；`member.target={member:id}`；
`support.target={support:{node:id,name:name}}` 且 payload 至少含 `fix[6]`；
`load.target={load:{case:case_or___top__,collection,name}}`，
collection 为 `nodal_loads/member_loads/member_spans/settlements` 之一，payload 复用对应
Domain IR 荷载项并显式给出单位。支座 name 在节点内唯一，荷载 name 在 case+collection 内唯一。
image_model 中每个节点、杆件、支座和荷载必须恰有一个对应 entity；反向也必须一一命中；
model 非 null 时同一 target 必须命中相同 ID/名称的正式对象。材料、截面和组合不要求图片实体。
counts 直接统计 image_model 集合，不统计 entities，也不依赖 model 是否已物化。

entity 与 dimension 的 `source` 只能是 `vision/user/migration`。原始视觉记录要求 `confidence` 与
`recognition_confidence` 都是 `[0,1]` 内有限数且初值相等；被人工修改后改为 `source=user`、
`confidence=null`、`verified=true`，并保留原 `recognition_confidence`。用户新建实体同样为
`source=user`、`confidence=null`、`recognition_confidence=null`、`verified=true`；用户新建
dimension 使用同一规则。migration 记录的两个 confidence 均为 null 且 verified=false，只能经人工
确认后转为 user。

dimension 的 `kind` 只能是 `member_length/node_distance/horizontal_distance/vertical_distance/angle`。
`member_length.target={member:id}`；三种 distance 的 target 均为 `{nodes:[i,j]}` 且 ID 升序；
`angle.target={members:[a,b]}` 且 ID 升序。引用必须存在，`value` 必须是有限正数；线性尺寸单位
必须可换算为米，角度单位只能是 `deg/rad`。无法唯一绑定 target 的 OCR 文本不进入
dimensions，只保留为原始响应并生成对应低置信度问题。

intersection 的 `point` 是派生图归一化坐标；`members` 是两个不同的正整数 PhysicalMember ID，
按升序存储且必须引用 image_model.members。`member_s` 是以这两个成员 ID 字符串为键的对象，在所有
decision 下都必须存在，两值均在闭区间 `[0,1]`，分别表示交点在两根构件上的参数坐标。
decision 为 `unknown/cross` 时 `node_id=null`；decision=`connect` 时 `node_id` 必须是正整数并
引用 image_model.nodes 中唯一共享节点。若交点落在任一构件的端点容差内，必须复用该端点节点并把对应
`member_s` 精确写为 0 或 1，禁止创建近重节点；两者均在内部时才创建新节点。共享节点在
image_model 中必须同时落在两条图像线的容差内，model 物化后其模型坐标也必须同时落在两根构件
容差内；PhysicalMember 端点保持不变，编译器凭显式共享内节点完成分析剖分。

提交后的正式拓扑沿用现有编译器语义：**任何正式 model 节点，只要在容差内落于 PhysicalMember
内部，就被视为该杆件的显式连接节点并触发分析剖分**。因此 X 接 connect 创建一个同时落在两根
杆件上的正式节点；T 接复用支杆端点且该点落在主杆内部；cross 不创建交点节点。若 cross 点容差内
已经存在会同时落在相关杆件上的其他正式节点，校验器必须生成 blocking topology issue，要求移动、
删除该节点或改为 connect，禁止静默提交。`intersections` 决策及图片证据随提交原样写入
`Session.multimodal_provenance` 审计侧车（随项目保存、进入同一撤销事务，但不作为求解器输入）；
正式 Domain IR 中节点的存在/缺席才是编译器的唯一拓扑输入。

`draft_hash` 对以下字段使用与 `change_preview.model_digest` 相同的 UTF-8、键排序、无多余空白
规范 JSON SHA-256：`format/image_model/model/merge_plan/source/work_plane/scale/entities/dimensions/intersections/issues/revision`。
明确排除 `confirmation`、controller 运行状态和 UI 展开/选中状态。

issue category 固定为：`work_plane_unconfirmed/scale_unknown/scale_conflict/perspective/
topology/intersection_unknown/low_confidence/load_incomplete/merge_collision`。`entity_refs` 是排序、
去重后的字符串数组，格式只能为 `node:<id>`、`member:<id>`、`support:<node>:<name>`、
`load:<case>:<collection>:<name>`、`dimension:<id>` 或 `intersection:<id>`；`__top__` 表示顶层荷载。
`severity` 只能是 `blocking/warning`，`status` 只能是 `open/resolved`。open 时 `resolution` 与
`resolved_by` 必须为 null；resolved 时 `resolution` 是非空字符串，`resolved_by` 只能是
`user/system`。`blocking_issues` 只统计 `severity=blocking && status=open`；底层条件再次成立时
必须把同一稳定 issue ID 重新打开，不得靠删除记录绕过。v1 迁移的问题默认 `status=open`，
并按上述阻断表决定 severity。同一草稿中 open issue 的 `(category,entity_refs)` 复合键必须唯一；
检测器重复命中时合并证据到同一 issue，不得追加同键记录。

透视门禁不依赖不可复现的自动阈值：导入新图时 `perspective_status=unconfirmed`，用户必须明确
选择“近似正射/已矫正”才写 accepted；选择“存在透视”写 rejected 并要求换图。自动会聚线检测
只能给出建议警告，不能单独改写该状态。不同位置线性尺寸推导的比例差超过 2% 一律归入
`scale_conflict`，不重复归类为 perspective。评测 fixture 直接记录预期 perspective_status 和
用户确认事件。

坐标换算使用派生图像素度量而非归一化欧氏距离。若派生图宽高为 `W/H`，锚点为
`(u0,v0)`、比例为 `s=length_per_pixel`，则平面坐标增量严格为：

```text
da = (u - u0) * (W - 1) * s
db = -(v - v0) * (H - 1) * s
```

尺度只使用 `status=confirmed` 的线性 dimension；angle、proposed 和
rejected_conflict 全部排除。对每条证据，先把标注值换算为米，再除以 target 在派生图上的像素长度
得到 `s_i`；member_length/node_distance 用两端欧氏像素距离，horizontal_distance/vertical_distance
分别用横向/纵向像素距离，零像素长度非法。候选按 evidence ID 排序，中位数 `m` 在偶数项时取中间
两项算术平均。若任一 `abs(s_i-m)/m > 0.02`，scale.status=`conflict` 且 length_per_pixel=null；
否则 status=`confirmed`、length_per_pixel=`m`、evidence_ids 为全部参与 ID。用户指定唯一可信尺寸时，
该项保持 confirmed、其他冲突项改为 rejected_conflict，最终比例精确取该项 s_i。没有合格证据时
status=`unknown`。所有判断使用换算后的双精度米/像素值，恰好 2% 不算冲突。

`axis_mapping` 必须是 `{first_axis,second_axis,offset_axis,image_right_sign,image_up_sign}`，
两个 sign 固定为整数 `1`。三种合法值分别为 `XY: X/Y/Z`、`XZ: X/Z/Y`、`YZ: Y/Z/X`。
换言之：`XY: a=X,b=Y,offset=Z`；`XZ: a=X,b=Z,offset=Y`；`YZ: a=Y,b=Z,offset=X`。
默认锚点按 `(u 最小, v 最大, node_id 最小)` 排序，即图中最左、再最低的节点；其模型坐标
由三个有限数组成的 `anchor_coordinates_xyz` 完整给出，默认两平面轴及偏移轴均为 0；用户修改
工作平面偏移时同步修改对应分量。`anchor_node` 是正整数 image_model node ID，并在 model 物化后
保持同一 ID。`evidence_ids` 是排序、去重字符串数组，只允许 `dimension:<id>`。用户对选定杆件
手工输入真实长度时，创建 `source=user/status=confirmed/kind=member_length` 的 dimension，而不另造
证据类型。长度单位先通过现有
`units.py` 换算，内部比较统一使用米。

`work_plane.offset` 与 `anchor_coordinates_xyz` 的 offset_axis 分量必须按 IEEE-754 数值精确相等；
不一致是 schema validation error，没有字段优先级或自动覆盖。更新偏移只能通过一个原子操作同时
写两处。物化后的所有 node x/y/z 均为米，Domain IR `units` 固定写 `N-m-Pa`。

图片输入限定 PNG/JPEG/WebP、单图不超过 20 MP；读取时先应用 EXIF 方向。派生图保存在项目临时目录，
草稿取消时可清理，提交或显式保存项目时随追溯记录保留。原始视觉响应、自动修复响应和人工修改记录
分开保存；日志和学习轨迹不得包含 API 密钥或图片二进制。

### 几何判定常量

- 图片点匹配：`max(5 px, 派生图对角线的 0.5%)`。
- 交点靠近端点：`max(6 px, 对角线的 0.5%)` 内视为端点候选，不新建近邻节点。
- 平行判定：夹角小于 2°；共线重叠不自动合并，生成阻断问题。
- 模型坐标重合：沿用正式建模入口的 `1e-6 m` 等价容差。
- 孤立节点、多个不连通子结构允许作为几何草稿，但生成警告；自连接、零长度、重复杆件和悬空引用阻断。

这些常量集中定义在一个模块中，测试和界面不得各自复制数值。

## 5. 界面布局

```text
┌──────────────────── 原图与覆盖层 ────────────────────┬──── 问题队列 ────┐
│ 节点/杆件/尺寸/支座/荷载，可分层显示、缩放与编辑       │ 阻断 3  警告 1   │
│                                                        │ 当前问题及选项    │
├──────────────────── 状态与坐标栏 ────────────────────┴─────────────────┤
│ XZ 平面 | 尺度未知 | 鼠标图像坐标 | 模型坐标 | revision 4             │
├────────────────────────────────────────────────────────────────────────┤
│ 重新识别   查看原始响应   预演变更   取消              确认载入（禁用） │
└────────────────────────────────────────────────────────────────────────┘
```

右侧不再同时铺开提供商、JSON、支座、荷载和尺度表单；只展示当前阻断问题。
高级信息放在折叠区，主界面只保留“看图—处理问题—预演—确认”四步。

## 6. 九张实施任务卡

这些卡是有依赖的增量切片，不是可以乱序并行的独立功能。每张卡必须留下可运行状态和
独立目标测试；共享文件只由当前卡修改。

### MM2-00：评测合同、公共常量与种子集

状态：**已完成（2026-09-08）**。实现位于 `src/multimodal_contract.py` 和
`multimodal_eval/`；6 项目标测试通过，相关回归 48 项通过，全量为 1054 通过、1 项环境跳过。

- 修改：新增 `multimodal_eval/` 及 `src/multimodal_contract.py`；冻结 ground truth schema、
  全部公共容差、评分公式和 `multimodal_eval/manifest.json`；先放入至少 8 张种子图片。后续卡只能导入公共常量，不能复制数值。
- 验收：同一预测重复评分完全一致；图片授权/来源、清晰图子集和困难标签可查询。
- 不在范围：追求 30 张最终规模、调用在线模型。

### MM2-01：状态机与 v2 数据迁移

状态：**已完成（2026-09-08）**。实现位于 `src/multimodal_workflow.py`、
`src/recognition_draft.py` 和 `tests/test_multimodal_workflow.py`。

- 修改：`src/recognition_draft.py`，新增独立 `src/multimodal_workflow.py` 及测试；冻结供后续卡使用的 schema/API。
- 验收：七个正常状态和失败状态可由数据确定性推导；v1 草稿可迁移；编辑使确认失效。
- 不在范围：界面重排、调用视觉 API。

### MM2-02：工作平面与图片预处理

状态：**已完成（2026-09-08）**。实现位于 `src/image_preprocess.py`、
`desktop/image_preprocess_widget.py`；现有草图面板只增加派生图和工作平面宿主连接。
6 项目标测试通过，全量收集 1056 项：1055 项通过、1 项按环境条件跳过。

- 修改：新增 `src/image_preprocess.py` 和 `desktop/image_preprocess_widget.py`；`sketch_panel.py` 只增加宿主连接点。
- 验收：XY/XZ/YZ 映射有单元测试；旋转/裁剪后坐标可逆追溯；不覆盖原图。
- 不在范围：通用透视矫正和三维反演。

### MM2-03：视觉 Provider 与 v2 修复循环

状态：**已完成（2026-09-08）**。`SketchParser.parse_v2_with_retry` 为 OpenAI、
Anthropic 和 DeepSeek 使用同一 V2 提示与校验入口；格式或引用错误最多修复两次，API 失败
立即停止，取消及迟到响应由 `MultimodalControllerState` 拒绝。原始文本可通过
`OfflineVisionResponseCache` 按图片、Provider、模型、提示词和 Schema 哈希离线固化。
8 项目标测试通过，全量收集 1064 项：1063 项通过、1 项按环境条件跳过。

- 修改：`src/sketch_parser.py`、各 provider 适配器与离线响应缓存工具。
- 验收：OpenAI/Anthropic/DeepSeek 共享同一 v2 schema；支持取消；格式/引用错误最多修复两次；
  成功、取消、API 失败和两次修复失败均进入规定状态，且全程不修改 Session。
- 不在范围：几何问题自动拍板、在线评测跑分。

### MM2-04：尺寸证据与尺度求解

状态：**已完成（2026-09-08）**。实现位于 `src/dimension_constraints.py`：
线性尺寸统一换算为米，按派生图像素长度生成尺度候选，以稳定证据 ID 排序并取中位数；
超过 2% 时进入冲突门禁，恰好 2% 不误判。用户可指定唯一可信尺寸，其余冲突证据会保留并
标记为 `rejected_conflict`；尺度问题可确定性解决和重开。10 项目标测试通过，全量收集
1074 项：1073 项通过、1 项按环境条件跳过。

- 修改：新增 `src/dimension_constraints.py`；不在本卡改主面板布局。
- 验收：单尺寸标定、冗余一致尺寸、冲突尺寸、单位缺失均有测试；冲突时禁止提交。
- 不在范围：材料、截面或规范文字推断。

### MM2-05：交点与拓扑消歧

状态：**已完成（2026-09-08）**。实现位于 `src/sketch_topology.py`：二维有限线段可检测
T 接、X 接、近端点、重复/重叠和明确跨越；`connect` 复用或创建显式物理节点，`cross`
不创建节点，删除前扫描杆件、支座、荷载、尺寸和交点引用。已有共享端点直接视为连接，
分离的共线构件不误报。端到端测试证明 X 接与 T 接在属性指派后由现有编译器映射到共享
AnalysisNode，PhysicalMember ID 与端点保持不变。10 项目标测试通过，全量收集 1084 项：
1083 项通过、1 项按环境条件跳过。

- 修改：新增 `src/sketch_topology.py`；输出问题和草稿操作 API，不在本卡改主面板布局。
- 验收：T 接、X 接、跨越、近端点、重复线和删除引用六类固定算例通过；另加端到端回归，
  将已确认的 X 接和 T 接草稿提交，用既有属性工具显式赋予测试材料/截面后编译，断言两根 PhysicalMember 共享同一 AnalysisNode，
  且 `physical_to_analysis` 映射到正确的剖分单元。现有编译器已支持显式内节点；若该回归失败，
  本卡可对 `src/model_compiler.py` 做满足该合同的最小修改，不得改变其他编译语义。
- 不在范围：任意三维空间相交自动建节点。

### MM2-06：问题驱动确认界面

状态：**已完成（2026-09-08）**。新增 `desktop/issue_panel.py` 并接入
`desktop/sketch_panel.py`：V2 识别完成后合并派生图追溯和已确认工作平面，运行尺度与拓扑检测，
按阻断优先级逐项展示问题。工作平面、透视和低置信度实体必须留下明确确认记录，交点只能选择
connect/cross；全局“已核对”不能绕过 V2 门禁。当前问题与图中实体联动高亮，任何处理都会清空
旧 confirmation/merge_plan，并与控制器 revision 同步。旧 V1 界面路径保持兼容。6 项目标测试
通过，全量收集 1090 项：1089 项通过、1 项按环境条件跳过。

2026-09-09 端到端补强：V2 草稿现已接通参考杆件尺度标定、节点拖动、杆件补画/删除、支座与
命名节点荷载编辑、结构化预演和原子加载。向已有模型追加时，重合节点会生成 blocking
`merge_collision`，必须由用户点击“复用现有节点”后才可提交；预演明确列出 reuse 与 load 操作。
新增链路测试覆盖确认失效、证据同步、引用重映射和单步撤销。

- 修改：`desktop/sketch_panel.py` 和新增 `desktop/issue_panel.py`，只消费 01～05 已冻结的 API。
- 验收：阻断问题逐项处理；编辑立即撤销确认；低置信度和当前问题在图中联动定位。
- 不在范围：重新设计整个桌面 Ribbon。

### MM2-07：结构化 diff 与原子提交

状态：**已完成（2026-09-08）**。新增 `src/draft_commit.py`：预演阶段确定性生成
replace-empty/add-only 合并计划和实体级 operations，候选模型不进入 preview，也不会提前修改
Session；提交时重新核对 draft/plan/baseline/operations 五组数据与哈希。`Session` 将正式模型与
`multimodal_provenance` 作为一个历史快照提交，异常时连同分析状态和历史一起回滚，成功只增加
一个撤销步骤且不会自动求解。模型保存仍是原 Domain IR JSON，追溯证据写入同目录
`<stem>.multimodal.json`；加载仅在 model hash 匹配时恢复，损坏或失配会忽略并提示，单 JSON
遗留文件保持兼容。5 项目标测试通过，全量收集 1095 项：1094 项通过、1 项按环境条件跳过。

- 修改：新增 `src/draft_commit.py`；本卡修改 `src/agent.py`、`src/history.py`、
  `desktop/main_window.py` 及提交/保存测试，与现有 `change_preview`/Session 历史衔接。
- 验收：提交前 Session 不变；陈旧哈希拒绝；正式 model 与 `Session.multimodal_provenance` 在同一
  撤销事务中成功或回滚；成功只产生一个撤销步骤。保存模型时保持原 Domain IR JSON 兼容，并在
  同目录写 `<stem>.multimodal.json` 侧车；侧车记录 model hash，载入时仅在哈希匹配时恢复，否则
  忽略并警告。遗留的单模型 JSON 仍可直接打开。
- 不在范围：提交后自动求解。

### MM2-08：多模态评测与发布证据

状态：**已完成（2026-09-09）**。`multimodal_eval/manifest.json` 现固定绑定 30 张项目原创、
CC0-1.0 图片及各自 ground truth/脱敏响应 fixture，覆盖清晰图、歪斜与手绘、缺尺寸、T/X 交点、
模糊符号、支座、荷载和拍照噪声。`multimodal_eval/evaluate.py` 只读取 manifest 绑定资产，微平均评分
节点、PhysicalMember、交点决策、支座、荷载、SI 数值/单位、尺度与 open issue，并执行 §7 门槛；
故意移除全部预测节点的负向测试证明门禁会失败。冻结 fixture 回放报告所有指标为 1.0，这只证明
离线回归可复现，不代表在线模型准确率。当前 OpenAI、Anthropic、DeepSeek 环境变量均未配置，
三项 smoke 明确记录 `SKIPPED_NO_CREDENTIALS` 且不进入离线分数。5 项新增目标测试通过；全量收集
1100 项：1099 项通过、1 项按环境条件跳过。

2026-09-09 完整 V2 桌面接线后，全量收集 1109 项：1108 项通过、1 项按环境条件跳过。

- 修改范围仅限 `multimodal_eval/`、测试 fixture、能力矩阵和发布报告。
- 将种子集扩展到至少 30 张经许可的固定图片，覆盖清晰图、歪斜手绘、缺尺寸、交叉歧义、模糊符号和拍照噪声。
- 保存脱敏后的原始识别响应作为离线 fixture，UI/校验测试不依赖网络。
- 评分节点、物理杆件、交点决策、支座、荷载、尺度和问题召回，不评分回复文案。
- OpenAI、Anthropic、DeepSeek 每个已配置 provider 只做一张最小在线冒烟；缺少凭据时明确
  记录为 `SKIPPED_NO_CREDENTIALS`，不得影响离线回归结论。正式回归读取缓存，记录模型、
  提示词和 schema 指纹。

`manifest.json` 是唯一发布门禁输入。每个 gate 图片条目固定
`{image_id,image_hash,ground_truth_path,ground_truth_hash,response_fixture_path,response_hash,
provider,model,prompt_hash,schema_hash,subset_labels,gate}`；`gate=true` 的每张图片恰好绑定一份响应
fixture，不允许运行时挑 provider 或挑最高分缓存。§7 的全部阈值只对所有 gate=true 条目合并后的
单一离线 corpus 应用，不逐 provider 应用。三家 provider 的在线 smoke 只验证连接/schema，不进入
分数。模型、提示词、schema 或响应改变时必须写新 fixture、更新哈希并显式评审 manifest diff；旧
manifest 的发布结论保持可复现，不能被目录中“较新”的缓存自动替换。

实施顺序固定为 MM2-00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08。每张卡单独跑目标测试，
通过后跑全量；前一张状态合同未冻结前，不开始后一张界面实现。

## 7. 最终验收门槛

- 30 张固定图片全部不会绕过确认直接修改 Session。
- 所有空间匹配共用 `T=max(5 px, 图像对角线 0.5%)`。节点匹配先生成距离不超过 T 的
  `(distance_px, prediction_stable_id_as_string, truth_stable_id_as_string)` 候选，按该元组升序遍历；
  仅当预测和真值双方都未匹配时接受，因此结果是确定性一一匹配。杆件按“两个已匹配端点组成的
  无序对”分组，每组内预测和真值分别按稳定 ID 字符串排序后逐一配对，多余项计 FP/FN。
  节点 F1 和物理杆件 F1 分别对全部图片累计 TP/FP/FN 后做微平均，公式均为
  `2TP/(2TP+FP+FN)`；分母为 0 时该项记为 `N/A` 且不参与门槛。清晰规则子集在 ground truth 中预先
  标记，不得跑分后调整。
- 清晰规则子集的节点 F1 和物理杆件 F1 均 ≥ 0.95；全体测试集两项均 ≥ 0.85；交点
  `connect/cross` 决策准确率 ≥ 0.90。支座按“匹配节点 + fix[6]”，荷载按“collection +
  span kind（若适用）+ 名称 + 匹配目标”计算微平均 F1，
  首版门槛均为 0.80，并单独报告数值/单位完全正确率。
- 交点只在“映射后的无序杆件对相同”且像素距离不超过 T 时生成候选，再按与节点相同的
  `(distance, prediction_id, truth_id)` 贪心规则一一匹配。决策完全正确记 1，决策错误、漏检和
  多检各记 0；端到端准确率分母为匹配正确、决策错误、漏检和多检之和。尺寸只在 kind 与
  映射后的 target 相同且标注中点距离不超过 T 时生成候选，同样确定性一一匹配。
- 支座不另设 type 字段：必须匹配节点且 `fix[6]` 完全一致才是 TP。荷载 F1 使用上一条匹配键，
  同键预测和真值分别按稳定 ID 排序逐一配对，重复项计 FP/FN。荷载数值/单位完全正确率以全部
  真值荷载为分母，先换算到 SI 后比较以下规范字段：nodal_loads 为前三项 N、后三项 N·m；
  settlements 为前三项 m、后三项 rad；member_loads 的 w[3] 为 N/m；member_spans 中 point 为
  `a(m)+w1[3](N)`，uniform 为 `w1[3](N/m)`，trapezoid 为 `w1[3]+w2[3](N/m)`。
  预测必须字段齐全且 kind 相同；每个标量须满足
  `abs(pred-truth) <= max(abs_tol, 1e-6*abs(truth))`，其中 m/rad 的 abs_tol=`1e-9`，
  N/N·m/N/m 的 abs_tol=`1e-6`。这一定义覆盖 truth=0；任一字段失败或漏检即整条荷载计错。
- 尺度正确率以有尺度真值的图片为分母，相对误差 ≤2% 才正确；问题召回按
  `category + entity_refs` 匹配：node/member refs 先用上述实体匹配映射，support/load refs 再按
  已匹配目标和名称映射，dimension/intersection refs 使用各自同一匹配器；refs 最后按字典序
  规范化。评分时只纳入“识别完成、任何人工处理之前”的 open issues，ground truth 与预测均要求
  复合键唯一；复合键且 severity 相同记 TP，键相同但 severity 不同记一个 FN 加一个 FP，未匹配
  真值记 FN，未匹配预测即“无依据问题”并记 FP。对全部 gate 图片累计后报告
  `recall=TP/(TP+FN)` 与 FP；分母为 0 时 recall 记 N/A。
- 所有悬空引用、零长度、重复杆件、未知尺度和未决交点都能阻止提交。
- 人工编辑、重新识别和正式模型变化都会使旧确认失效。
- 提交成功只有一个撤销步骤；提交失败时模型、结果和历史均保持不变。
- 图片到物理模型全过程可追溯到图片哈希、实体证据、人工决策和草稿 revision。
- 全量既有测试以本规范建立时的 1049 项为基线，不得删除或放宽；新增上游模型波动只记录
  为识别失败，不能改坏确定性内核。

## 8. 第一个可演示切片

选择一张带尺寸的单跨门架草图：

1. 导入并确认 `XZ` 工作平面。
2. 识别两柱两梁、两个柱脚支座和跨度/高度尺寸。
3. 用户处理屋脊或梁柱交点问题，校准一条已知尺寸。
4. 拖动一个误识别节点并看到相关覆盖线实时更新。
5. 查看结构化 diff，确认载入。
6. 通过既有 Property 流程指定材料和截面，使 Agent 状态从 draft 进入 ready。
7. 模型树显示物理构件；打开分析网格并验证交点剖分，软件不自动求解。

这个切片同时证明视觉证据、人工控制、原子提交和既有物理构件主链路已经接通。
