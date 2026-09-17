# v3 500-scene batch experiment：原生 Cycles 水下数据集说明（草案）

> 状态：batch contract / worker draft。本文件定义现有 v3 生成器的批量实验规则，不把配置、脚本或局部输出误报为 500 scenes 已完成。只有全量生成、校准、验收和归档证据齐全后，才能更新为已交付状态。

## 1. 这是什么，不是什么

这是在已完成的 `seabed_dataset_v3` 原生 Cycles 生成系统上进行的 **v3 batch experiment**。实验只扩展 scene 数量、scene/layout seed、每 scene 的单一光照选择、独立输出根和归档流程；不创建新的水下渲染算法，不把它命名为 v4，也不替换 v3 的几何、水体、相机、标签和资产链路。

生成路径应复用 v3 的 `src/`、已审计资产和原生 Cycles 构建逻辑；批量 driver 可以把当前 v3 的三模式构建入口限制为每个 scene 的一个已选 mode，但不得修改 `src/` 来改变算法语义。`outputs/v3_first_round/` 是已完成的 v3 首轮基线，批次使用配置解析出的新 `outputs/v3_500_scenes_final/`，不得写入、覆盖、混合或重建首轮输出。

本批是合成、可控数据实验，不能表述为真实海域、真实海水光谱标定、绝对辐亮度标定、真实生态统计、活体生态分布或下游模型性能提升。配置中的水体系数、光照范围和质量阈值是生成约定，不是现场测量。

## 2. 目标数量

正式目标是 500 个不同且 accepted 的 scene，使用稳定的 `scene_0001` 至 `scene_0500` 编号（最终格式以批次 config 冻结值为准）。每个 scene 只采样一套可重放的随机光照，生成四个严格配对状态：

| 状态 | 含义 | 目标逻辑 RGB 数量 |
|---|---|---:|
| `clear` | Clear reference，无 seawater volume 和 air-water surface | 500 |
| `mild` | 原生 Cycles 轻度水体 | 500 |
| `medium` | 原生 Cycles 中度水体 | 500 |
| `strong` | 原生 Cycles 强度水体 | 500 |

因此目标为 500 个 Clear reference、1,500 个 degraded RGB 状态和 2,000 个逻辑 RGB 状态。每个状态可以同时保存 EXR 与 PNG；两种编码属于同一个逻辑状态，不能重复计数。

当前批次配置还定义了 train/validation/test 的拆分和 Natural/Artificial/Mixed 的平衡计划。它们是可重放计划值，不是已完成结果；必须以最终保存的 resolved config、plan 和验收报告为准。若采用现有 batch config 的计划，三类单光照 mode 的目标计数为 167/167/166。

## 3. 500 个不同 scene 与单一随机光照

“不同 scene”不能只靠改目录名实现。每个 scene 必须有唯一的 scene/layout seed，并从 scene number 派生独立的布局、几何、相机和鱼姿态随机流；重复运行同一 source/config/seed 时应可重放，改变 scene number 时不能静默复制同一几何状态。

每个 scene 的光照契约是：

1. 只选择一个 `lighting_mode`（例如 `natural`、`artificial` 或 `mixed`）和一套完整 `lighting_plan`。
2. mode 选择可以使用有配额的确定性 schedule；天空、太阳、Spot、方向、能量、色温等参数由记录在 metadata 中的 scene/light seed 可重放采样。确定性 schedule 不等于真实光照统计。
3. 该 lighting plan 在 Clear、mild、medium、strong 四态之间完全复用；不得为同一 scene 建立三套 mode 后再丢弃两套，也不得在水体状态之间重新抽光。
4. `lighting_mode`、lighting plan 摘要、lighting seed 和 plan hash 必须进入 scene manifest、状态 metadata 和签名链。

## 4. Clear / mild / medium / strong 严格配对与共享 GT

同一个 `scene_id` 的四态必须共享：

1. 同一批几何、珊瑚礁、鱼类、岩礁、海草、细节沙床、实例 ID、可见性和静态 LOD。
2. 同一相机、分辨率、裁切、姿态、颜色管理和曝光设置。
3. 同一批鱼的实际变换与姿态；不得为 mild、medium、strong 重新随机摆鱼。
4. 同一套已选 lighting mode、lighting plan、light seed 和可见性关系。
5. 同一 `reference_id`、几何/配对状态摘要和共享 GT。

`clear` 必须是真正移除 seawater volume 与 air-water surface 的 reference，不得由 degraded 图像逆处理得到。`mild`、`medium`、`strong` 只可改变已冻结的原生闭合水体吸收/散射预设及其状态字段；不得使用 Python 对渲染像素做衰减、加雾、颜色覆盖或照片卡片后处理。水体参数不进入几何 GT。

GT 在 scene 范围内只保存一份。四态通过 manifest 引用同一 GT bundle 或同一 reference bundle，不复制三份 degraded GT。归档 staging 推荐把共享 GT 放在 `clear/` 下；也可以放在 scene 根级 `gt/` 目录，但必须记录 `gt_scope=scene_shared` 和 `gt_bundle_sha256`。

沿用 v3 的 GT 定义：`depth_range_m` 是相机原点到首个目标三角形交点的欧氏距离（米）；`depth_camera_z_m` 是同一交点的正 camera-space `-Z` 深度（米）；miss 为 0。Semantic/Instance/Valid Mask 来自实际目标几何求交，不来自照片、纹理或后处理。语义 ID 保持 0 background/invalid、1 sand、2 rock、3 coral、4 grass、5 fish；标签 PNG 不插值，Valid Mask 为 0/255。

## 5. 生成层与输出

v3 batch driver 的内部输出可以保持 v3 的扁平 bundle 结构，典型形式如下。这里的路径是契约示例；实际路径必须以 resolved config 和 `batch_run` 记录为准。

```text
outputs/v3_500_scenes_final/
  plan.json
  states/<scene_id>/accepted.json
  attempts/<scene_id>/a00/
    geometry/                     # depth、标签、geometry plan/quality
    references/<scene_id>_clear/  # clear RGB、共享 GT、metadata、success
    samples/<sample_id>/          # degraded RGB、metadata、success
  rejected/<scene_id>/            # 被移出的旧 attempt 或修复前 bundle（如产生）
  references/<scene_id>_clear/
  samples/<sample_id>/
  scenes/<scene_id>.blend         # 可选 native cache，不是默认发布内容
  batch_manifest.json
  manifest.jsonl
  events.jsonl
  batch_run.json
```

每个 reference 至少包含 `clear_rgb.exr/png`、`depth_range_m.exr`、`depth_camera_z_m.exr`、16-bit Semantic/Instance PNG、Valid Mask 和 metadata；每个 degraded bundle 至少包含 `degraded_rgb.exr/png` 和 metadata。逻辑样本数按 scene/state 计，不按 EXR/PNG 文件数计。

`manifest.jsonl` 若沿用现有 v3 batch driver 的粒度，通常是一行一个 accepted degraded 状态，目标为 1,500 行；`batch_manifest.json` 还应记录 scene/reference/pair 汇总。归档阶段必须另外生成能逐 scene 核对的 500 行 scene index 和 2,000 行四态 state index，不能因为内部 manifest 扁平就丢失配对关系。

## 6. 五类 provenance 签名

每个 scene metadata、batch plan、manifest、验收报告和归档清单都必须回链以下五类签名：

1. **source signature**：未改变的 v3 `src/`、必要的 v3 `run.py` 与本批 additive batch driver 的文件清单及 SHA-256。它证明本批复用了哪个 v3 生成器和哪个批量入口；不能把新算法或 v4 源码混入而仍称 v3 batch。
2. **config signature**：`configs/v3_500_scenes.json` 和 resolved config 的规范化 SHA-256；至少覆盖 500 scene ID/seed schedule、split、每 scene 单光照规则、四态、水体预设、输出编码和质量阈值。config 改变就必须使用新的 output root 并重新校准。
3. **calibration signature**：v3 batch 专属 calibration gate 的路径、状态和 SHA-256，例如计划中的 `reports/calibration_v3_500/gate.json`。现有 `reports/calibration_v3/gate.json` 只能作为 v3 首轮基线，不能未经重新核对直接充当 single-light four-state batch gate；在 batch gate 缺失或未通过前不得生成正式 500 结果。
4. **runtime signature**：实际 Blender 版本/build、Cycles engine/device、Python、OS、OCIO/颜色管理、samples、反弹、颜色编码、安全启动参数、运行锁和实际时间。不得把未运行或未读回的环境写成已验证。
5. **asset signature**：v3 `assets/asset_catalog.json`、源/处理资产相对路径、asset UID、许可证、颜色空间、语义/实例/形态和每个文件 SHA-256。即使 v3 总 source hash 已覆盖资产，也要在 batch manifest 中提供可独立审计的 asset signature；资产路径缺失或 hash 不符时拒绝该 scene。

签名只能使用稳定 JSON 值、规范化配置和文件 hash；禁止临时绝对路径、Python 对象 `repr`、内存地址、隐式网络资源和未锁定系统状态。签名变化不得静默复用旧缓存、旧 GT 或 `v3_first_round` manifest。

## 7. 失败 attempt、恢复与接受规则

一次 attempt 不是一个 accepted scene。每个 scene 最多尝试次数、重试顺序和 seed 派生由 batch config 冻结；每次重试必须有新的 `attempt_index` 和独立目录。

1. 失败 attempt 的 geometry、部分 RGB/GT、metadata、quality、日志和具体失败原因必须保留；不要只保留“failed”字符串。
2. 若运行器需要移走同名旧 attempt，应移动到 `rejected/<scene_id>/` 等可追溯位置，不能覆盖。`events.jsonl` 必须 append-only 记录开始、失败、重试、恢复和接受事件。
3. 失败原因应能定位到 geometry/camera/fish/light clearance、RGB 质量、GT、缺文件、hash、内存、渲染或 I/O gate。
4. 只有四态齐全、严格配对、GT/labels 正确、质量 gate 通过、签名匹配并完成 seal 的 attempt 才能写入 accepted manifest。
5. 失败 attempt 可以留在未压缩 provenance staging 中，但不得计入 500 scenes、500 Clear、各 500 degraded、1,500 degraded 或 2,000 RGB，也不得进入五个正式 ZIP。
6. 合法 resume 必须复用一致的 source/config/calibration/runtime/asset 签名且不重复写入；更换 config、损坏 GT/RGB、错误 job root 或重复 sample ID 时必须明确拒绝。

## 8. 交付前验收门槛

在没有下列证据之前，只能报告 `draft`、`incomplete`、`calibration` 或 `smoke`，不能声称 500-scene v3 batch 已完成：

1. **计数与唯一性**：500 个 accepted scene，scene ID 和 layout seed 唯一；每 scene 正好 Clear + mild + medium + strong；Clear 500、mild 500、medium 500、strong 500；degraded 1,500；逻辑 RGB 2,000；无重复或漏项。
2. **单光照**：每 scene 恰好一个 lighting mode/plan；mode 分布与 config 计划一致；同 scene 四态 lighting plan hash 一致；没有三模式生成后丢弃的隐藏计数。
3. **配对与 GT**：四态几何、相机、鱼姿态、可见性和光照一致；Clear 无水体；三档使用原生水体；GT 共享且 hash 一致；scene 仍保留复杂珊瑚礁、鱼类、岩礁、海草与细节沙床。
4. **文件与标签**：EXR/PNG 可独立读回，尺寸、通道、位深、颜色空间、有限值、深度米制、semantic/instance/mask 和 metadata 正确；没有 NaN、Inf、损坏 bundle、missing asset 或非法外链。
5. **质量与校准**：batch 专属 calibration gate 通过；每个 lighting mode 和 mild/medium/strong 组合通过已冻结的 valid fraction、near/middle/far/background、生态像素、动态范围、clipped fraction、effect/structure noise、实例覆盖、几何/三角预算和内存门。阈值缺失或未记录不得临时放宽。
6. **签名与可复现**：source/config/calibration/runtime/asset 五类签名存在并匹配；独立进程同 seed 的 geometry、GT、metadata 关键摘要一致；resume/损坏/异配置拒绝有证据。
7. **首轮隔离**：`outputs/v3_first_round/`、首轮 manifest、首轮 calibration/report 和旧 source 不被写入；batch 所有 accepted/failed 输出可由新 root 独立追溯。
8. **归档**：五卷各 100 scene；每卷 ZIP 可测试读取；卷清单、scene manifest、state index 和 SHA-256 清单一致；抽样解压与未打包 staging 的文件 hash、GT 引用和签名一致。
9. **人工检查**：每卷至少抽查 Clear/mild/medium/strong、共享 GT、单光照和复杂海底构图；不能只检查漂亮 RGB 或把简单几何占位当作通过。

## 9. 未打包原始数据与上传边界

未打包的 batch output、source、config、batch calibration、runtime、asset catalog、manifest、events、质量报告和所有失败 attempt 是权威审计材料。创建 ZIP 后不得自动删除、移动、覆盖或以 ZIP 替代这些原始目录；EXR/PNG 是否使用格式内部压缩，不改变“未打包原始数据必须保留”的要求。磁盘不足时停止并报告，不静默清理。

本实验默认完全本地化：生成、验收、分卷和 SHA-256 计算不得自动上传到 GitHub、GitHub Release、云盘、对象存储或其它外部服务，不执行 `git push`。即使所有 gate 通过，也只停在本地 staging 和校验清单；任何外部发布、上传或删除原始数据都需要用户当下另行明确授权。

## 10. 已披露限制

1. 这是复用 v3 生成器的合成、可控 batch experiment，不是新的 v4 算法，也不是实测海域数据。
2. 单一随机光照减少了每 scene 的三模式对照；它不能提供真实环境的光照分布、时间变化或方差估计。
3. 水体吸收/散射、IOR、反弹、samples、颜色管理和 device 属于生成配置，不能解释为真实海水光谱或绝对辐亮度标定。
4. GT 是几何交点的欧氏距离和 camera-space 深度，不是水下光程、声学距离或真实传感器测距。
5. 程序化/扫描资产用于受控形态和外观；不能据此推断真实物种比例、生态统计、活体姿态、采集地点或野外行为。已有博物馆标本扫描尤其不能当作活体生态证据。
6. 500 个合成 scene 不保证训练收益、真实域泛化或下游性能提升；本批不包含这样的实验结论。
7. 粒子、海浪、传感器噪声、复杂 caustics 和其它未写入 batch config 的因素不得假定为已模拟。
8. 未来若改变 v3 source、资产、几何、相机、光照定义、水体定义、GT 规则或 config，必须建立新的签名和 output root，不能追加写入本批或覆盖 v3 首轮。
