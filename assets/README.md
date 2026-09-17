# v3 外部资产目录

本目录只服务于 `seabed_dataset_v3`。`asset_catalog.json` 是本批已存在下载文件的机器可读溯源清单；它记录来源页、许可证、下载日期（2026-09-14）、相对路径、实际 SHA-256、分辨率/变体、用途、预览资格和接入风险。

## 目录分层

```text
assets/
├── asset_catalog.json       # 资产 UID、来源、许可证、文件 SHA-256 与接入门槛
├── README.md                # 本说明
├── source/                  # 不可变的原始下载文件；不得覆盖同名源文件
│   ├── polyhaven/           # rock_3、coral_ground_02 的 2K JPG PBR 图（含 height/displacement）
│   └── smithsonian/         # 3 个 CC0 GLB（贴图内嵌）
├── processed/               # 后续导入/烘焙/LOD 结果；本批为空
└── licenses/                # 后续如保存许可页快照或文本，放在此处；本批为空
```

`asset_catalog.json` 中的 `relative_path` 相对于项目根目录 `/Users/xiaoran/Desktop/blender`，不是相对于 `assets/`。本批没有 `processed/` 文件，也没有把源素材直接复制进 `assets/processed/`。目录清单本身不修改 v2；后续 v3 的只读导入审计与预览结果分别记录在 `../reports/external_asset_probe.json`、`../reports/preflight_v3_assets.json` 和 `../previews/`，不能据此宣称正式数据已发布。

## 许可证边界

本批只登记已核验的 CC0/Public Domain 候选：

1. Poly Haven `rock_3` 与 `coral_ground_02`：来源页和 [Poly Haven License](https://polyhaven.com/license) 标为 CC0；作者信息在各自资产页中记录。
2. Smithsonian `Acropora cervicornis` USNM 1171477、`Diploria labyrinthiformis` USNM 74947、`Diodon hystrix` USNM 195928：各自 Smithsonian 3D 对象页标注 Metadata Usage: CC0；通用边界见 [Smithsonian Open Access FAQ](https://www.si.edu/openaccess/faq) 和 [Terms of Use](https://www.si.edu/termsofuse)。

CC0 只说明指定数字资产的版权状态。发布数据集时仍应保留来源 URL、机构/作者和下载日期；不得暗示 Smithsonian 或 Poly Haven 背书，不得把对象页中的采集地点、深度或博物学说明误写成渲染参数或实测海水数据。许可证没有明确允许再分发的素材不得追加到本目录。

## 颜色空间与通道

颜色空间必须按通道记录并在 Blender 导入时显式设置：

1. `*_diff_2k.jpg` 或 GLB 内嵌 `base_color`：颜色数据，使用 sRGB；送入 Principled BSDF 前由 Blender 做颜色管理转换。
2. `*_nor_gl_2k.jpg` 或 GLB 内嵌 `normal_gl`：数据贴图，使用 Non-Color，并通过 Normal Map 节点；`GL` 表示 OpenGL 方向约定，仍需在目标材质上实测法线方向。
3. `*_rough_2k.jpg` 或 GLB 内嵌 `roughness`/`occlusion`：数据贴图，使用 Non-Color；粗糙度和 AO 的通道映射必须写入后续处理记录。
4. `*_disp_2k.jpg`：高度/Displacement 数据，按 Non-Color 作为 bump 输入；若改为真实几何 displacement，必须建立新的 geometry state 并重新生成全部受影响 GT，不能把 bump 输入误报为已改变轮廓。
5. Poly Haven JPG 的嵌入 profile 已在本次清单中记录。Smithsonian GLB 的内嵌 JPEG 在本地检查没有 ICC profile，因此不能依赖文件自动推断；按上面的用途显式设置。

材质贴图只改变 RGB 外观时，标签仍来自实际目标几何。若启用真实 Displacement、重拓扑或替换扫描网格，几何轮廓和遮挡会改变，必须创建新的 geometry state，并重新导出 `depth_range_m`、`depth_camera_z_m`、Semantic、Instance 和 Valid Mask；不得沿用旧 GT。

## 参考照片边界

参考照片只能用于观察沙纹、侵蚀、珊瑚形态、构图和配色统计，不得直接接入渲染节点，不得作为海床平面、背景或带深度的照片卡片，也不得作为 semantic/instance 标签来源。实际渲染必须使用可追溯的 3D 几何和/或 PBR 通道；照片如需随项目保存，应单独标成 `reference_only`，并记录其独立许可证。

## 进入 preview 的含义

清单中的 `preview.eligible: true` 只表示本地文件已存在、来源/许可证已核验、容器或通道基本完整，可以进入隔离的资产级预览。它不等于允许进入正式 v3 场景：`formal_scene_eligible` 当前统一为 `false`，还必须通过单位/轴向/枢轴、法线/UV、静态烘焙、LOD、实例标签、几何预算、纹理内存和配对一致性检查。Smithsonian 的三个模型尤其要注意它们是干燥博物馆标本扫描；Diodon 记录还是干燥鱼类标本，不能直接当作活鱼姿态或颜色证据。

## 自检

在项目根目录执行以下只读检查即可复核本清单：

```sh
python3 -m json.tool seabed_dataset_v3/assets/asset_catalog.json >/dev/null
shasum -a 256 seabed_dataset_v3/assets/source/polyhaven/coral_ground_02/* \
  seabed_dataset_v3/assets/source/polyhaven/rock_3/* \
  seabed_dataset_v3/assets/source/smithsonian/*.glb
file seabed_dataset_v3/assets/source/polyhaven/*/*.jpg \
  seabed_dataset_v3/assets/source/smithsonian/*.glb
```

验收时应逐项比对 `asset_catalog.json` 的 `bytes` 和 `sha256`；路径缺失、哈希不符、许可证状态矛盾、缺图或颜色空间未知时停止接入，不用近似素材替代。不要用本清单中的任何未测值推断三角面数、拓扑质量、世界尺度或渲染性能；这些属于后续导入/性能 gate。
