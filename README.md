# seabed_dataset_v3

基于 Blender 原生 Cycles 的合成水下海底图像数据集与扩散模型训练准备工程。

本仓库包含可控的海底三维场景、Clear 参考图像、三档原生水体退化图像、几何真值标签、批量生成配置、验收脚本和训练就绪的 GitHub Release。核心目标是建立“同一场景、同一物体/相机姿态、不同水体退化程度”的成对数据，并为后续条件扩散模型提供稳定的数据接口。

> 当前发布状态：已完成并发布 400 个 accepted scene，共 400 张 Clear target、1,200 张 degraded input 和 1,200 条 paired samples。原计划中的 scene_0401–scene_0500 没有进入本次发布。

训练数据下载：[v3-400-training-ready Release](https://github.com/weekjvj-dotcom/seabed_dataset_v3/releases/tag/v3-400-training-ready)

## 目录

- [项目定位](#项目定位)
- [当前状态与数据规模](#当前状态与数据规模)
- [数据下载](#数据下载)
- [数据组织与配对规则](#数据组织与配对规则)
- [场景生成与控制变量](#场景生成与控制变量)
- [扩散模型训练流程](#扩散模型训练流程)
- [下一步实施路线](#下一步实施路线)
- [仓库结构](#仓库结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [源码验证与批量生成](#源码验证与批量生成)
- [质量保证与可复现性](#质量保证与可复现性)
- [局限与解释边界](#局限与解释边界)
- [故障排查](#故障排查)
- [许可证与资产来源](#许可证与资产来源)
- [引用](#引用)

## 项目定位

### v3 是什么

v3 是一套由 Blender 原生 Cycles 生成的、可复现的合成水下数据工程，包含：

1. 程序化海床、珊瑚、鱼、岩礁和海草等三维对象；
2. 有来源和许可证记录的外部 PBR/扫描资产；
3. 固定的几何状态、对象姿态、相机和每个 scene 的 lighting plan；
4. 无水体 Clear reference；
5. 原生 Cycles 吸收/散射水体产生的 mild、medium、strong degraded image；
6. depth_range_m、depth_camera_z_m、Semantic、Instance 和 Valid Mask；
7. scene、state、asset、runtime 和配置签名；
8. 纯 Python 合同测试、flat output 检查和归档检查。

RGB 图像由 Cycles path tracing 直接生成。项目没有使用 Python 对渲染后的 RGB 像素做雾化、颜色覆盖、指数衰减或照片后处理。

### v3 与 v4 的关系

本仓库采用 v3 作为算法/生成系统版本。configs/v3_500_scenes.json 是在已验证 v3 生成器上扩展规模的 additive batch 配置，不代表必须重新建设一套 v4 Blender 系统。

当前版本关系如下：

1. v3_first_round：早期小规模基线；
2. v3_500_scenes：批量生成计划；
3. v3-400-training-ready：该计划中已经完成并验收的前 400 个 scene；
4. 如果修改水体模型、几何规则、GT 定义、资产链路或核心生成算法，应建立新的配置、签名和输出根目录。

## 当前状态与数据规模

| 项目 | 当前值 |
| --- | --- |
| 生成系统 | v3 additive batch |
| 已发布 scene | 400 |
| Clear target | 400 |
| degraded image | 1,200（每 scene 3 张） |
| paired samples | 1,200 |
| 状态 | clear、mild、medium、strong |
| 图像尺寸 | 256 × 256 |
| RGB 发布格式 | 16-bit RGB PNG |
| 原始渲染格式 | 32-bit float RGB EXR + 16-bit RGB PNG |
| 水体渲染 | Blender Cycles native volume |
| 训练 ZIP | 4 卷，每卷 100 scene |
| 当前光照策略 | 每个 scene 只选择一个 lighting mode |
| 每 scene .blend | 未作为训练发布物保存 |
| 训练代码 | 尚未实现 PyTorch diffusion trainer |

验收报告中的 scene-level lighting mode 计数为：

| lighting mode | scene 数量 |
| --- | ---: |
| natural | 134 |
| artificial | 133 |
| mixed | 133 |

当前 400 个 scene 在 release manifest 中都标记为生成批次的 train。这不等于已经准备好了神经网络最终的 train/validation/test 划分；训练前仍须按 scene_id 建立场景级 split。

## 数据下载

训练就绪文件已经上传到私有 GitHub Release：[v3-400-training-ready](https://github.com/weekjvj-dotcom/seabed_dataset_v3/releases/tag/v3-400-training-ready)。

如果本地 GitHub CLI 已登录：

    mkdir -p data/v3_400_training_ready
    gh release download v3-400-training-ready --repo weekjvj-dotcom/seabed_dataset_v3 --pattern 'v3_400_training_part*.zip' --dir data/v3_400_training_ready

四个发布卷：

    v3_400_training_part01_scene_0001_0100.zip
    v3_400_training_part02_scene_0101_0200.zip
    v3_400_training_part03_scene_0201_0300.zip
    v3_400_training_part04_scene_0301_0400.zip

每个 ZIP 都包含自己的 README.txt、pairs.jsonl 和 dataset_manifest.json。不要把四个 ZIP 解压到同一个目录，否则同名的卷级索引会覆盖。推荐按卷解压：

    mkdir -p data/v3_400_training_ready/parts
    for archive in data/v3_400_training_ready/*.zip; do
      part_dir="data/v3_400_training_ready/parts/$(basename "$archive" .zip)"
      mkdir -p "$part_dir"
      unzip -q "$archive" -d "$part_dir"
    done

训练 loader 应遍历四个 parts 目录中的 pairs.jsonl，或显式生成合并索引；不要依赖文件名排序猜测配对关系。

## 数据组织与配对规则

每个 scene 在训练 ZIP 中的结构：

    scene_0001/
    ├── scene_0001_clear.png
    ├── scene_0001_mild.png
    ├── scene_0001_medium.png
    ├── scene_0001_strong.png
    ├── depth_range_m.exr
    ├── depth_camera_z_m.exr
    ├── semantic_id.png
    ├── instance_id.png
    ├── valid_mask.png
    └── scene_0001_metadata.json

文件含义：

| 文件 | 含义 | 训练用途 |
| --- | --- | --- |
| scene_####_clear.png | 无 seawater volume 和 air-water surface 的 Clear reference | 监督目标 x0 |
| scene_####_mild.png | 原生 Cycles 轻度水体 | 条件输入 c |
| scene_####_medium.png | 原生 Cycles 中度水体 | 条件输入 c |
| scene_####_strong.png | 原生 Cycles 强度水体 | 条件输入 c |
| depth_range_m.exr | 相机原点到首个目标三角形交点的欧氏距离，单位米 | 几何分析/可选辅助输入 |
| depth_camera_z_m.exr | 首个交点的正 camera-space -Z 深度，单位米 | 几何分析/可选辅助输入 |
| semantic_id.png | 语义类别标签 | 监督/分析 |
| instance_id.png | 实例标签 | 实例分析 |
| valid_mask.png | 有效几何交点，0 或 255 | 深度和标签有效性 |
| scene_####_metadata.json | scene、布局、光照、状态和共享 GT 信息 | 可复现和分层统计 |
| pairs.jsonl | degraded input 到 Clear target 的显式映射 | 训练索引 |

Clear 的命名与三档退化一致：scene_0001_clear、scene_0001_mild、scene_0001_medium、scene_0001_strong。内部原始输出仍保留 reference_id=ref_<digest> 作为唯一追踪标识。

同一个 scene 的四态共享：

1. 几何、珊瑚/鱼/海草/岩礁的实际变换和可见性；
2. 相机、分辨率、裁切、颜色管理和曝光；
3. 该 scene 选定的 lighting plan；
4. reference identity、state digest 和几何 GT。

只有 native water preset 在 clear、mild、medium、strong 之间变化。GT 在 scene 范围内共享一份，不为每个 degraded 状态重复生成几何标签。

Semantic ID 约定为：0 background/invalid、1 sand、2 rock、3 coral、4 grass、5 fish。标签 PNG 不应插值；valid_mask.png 等于 255 的像素必须对应有效语义/实例和正深度，无效像素深度为 0。

## 场景生成与控制变量

当前 400-scene batch 的控制关系：

1. 每个 scene 由唯一的 scene/layout seed 派生几何、相机和对象随机流；
2. 每个 scene 只选择一个 lighting_mode：natural、artificial 或 mixed；
3. 这个 lighting plan 在同一 scene 的 clear、mild、medium、strong 四态之间复用；
4. 三档水体是同一个固定 scene 的不同 native volume preset；
5. scene 之间允许布局、鱼位置、相机和 lighting mode 不同。

因此当前 release 可以做同一 scene 下的退化程度对照，但不是同一 scene 下 natural/artificial/mixed 三种光照的完整控制变量数据集。不能把当前 400 组描述成三光照 paired set。

如果后续补充三种光照，正确方式是先固定一个 scene 的几何、物体姿态、相机和分辨率，再只改变 lighting plan：

    同一个 Blender scene
    固定几何 / 鱼姿态 / 珊瑚姿态 / 相机 / 分辨率
                             │
              ┌──────────────┼──────────────┐
              │              │              │
           natural        artificial       mixed
              │              │              │
        clear+mild+      clear+mild+     clear+mild+
        medium+strong    medium+strong   medium+strong

跨三组记录 light_id、lighting seed 和 lighting plan hash；每一组内部仍保持 Clear 与三档水体的几何/相机/对象一致。补充数据应使用新的 manifest 字段，不能静默混入当前 400 组。

批量计划位于 [configs/v3_500_scenes.json](configs/v3_500_scenes.json)，其中 500 是计划规模，不是本次 release 的实际计数。计划参数包括 256×256、Cycles 512 samples、5 m 海床深度、mild/medium/strong 水体、IOR 1.333，以及 train 400 / validation 50 / test 50 的计划 split。当前只发布 scene_0001–scene_0400，且 release manifest 中均为 train。

## 扩散模型训练流程

本节将你提供的流程图与论文 *Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy* 映射到本仓库。它是下一阶段的实现方案，不是当前仓库已经完成的训练结果。

### 数据到模型的映射

- x0：scene_####_clear.png，无水体 Clear target；
- c：同 scene 的 mild、medium 或 strong PNG，作为水下条件图；
- t：前向 Gaussian diffusion 的随机 timestep；
- ε：采样得到的真实 Gaussian noise；
- xt：由 x0 加噪得到的 timestep 状态；
- εθ(xt, c, t)：条件去噪网络预测的 Gaussian noise。

训练链路：

    固定的 Blender scene
      ├── Clear reference x0
      └── native Cycles degradation c

    x0 + timestep t + Gaussian noise ε
      └── forward diffusion → xt

    concat(xt, c) + t
      ↓
    Transformer denoiser εθ(xt, c, t)
      ↓
    predicted ε
      ↓
    L1(predicted ε, true ε)
      ↓
    更新模型参数

训练时不是把 mild、medium、strong 三张图相互串联，而是从一个 Clear target x0 采样 timestep 和 noise，并把对应的一张 degraded image c 作为条件。

前向扩散可从以下关系开始：

    ε ~ N(0, I)
    xt = √ᾱt · x0 + √(1 − ᾱt) · ε

其中 ᾱt 由 beta schedule 累积得到。loader 应先把 16-bit RGB PNG 转换为 float tensor，再统一归一化到 [-1, 1]，并通过反归一化测试。

### 条件去噪网络 baseline

论文 Figure 1 可作为第一版 baseline：

1. 在 channel 维拼接 xt 与 c，得到 6-channel 输入；
2. 经过卷积投影和 timestep embedding；
3. 使用轻量 Transformer denoiser；
4. 使用 channel-wise attention；
5. 输出与 xt 同尺寸、同 channel 数的噪声预测；
6. 用真实 ε 与预测 εθ(xt,c,t) 计算 L1 loss。

当前仓库没有 training 目录、PyTorch 依赖、Transformer denoiser、训练脚本或 checkpoint。这里的公式和网络结构是后续开发契约，不能解释为已经训练完成。

论文设置可以作为起始参考，例如 T=2000、线性 beta schedule、128×128 crop、batch size 8、Adam、lr=1e-4 和减少 reverse sampling steps 的 DDIM-like 策略。但这些是论文设置，不是本仓库的已验证结果。8 GB Apple M2 环境应先使用较小 crop/batch，必要时 gradient accumulation，先做单 batch overfit，再扩展训练。

## 下一步实施路线

### 1. 先建立 scene-level split

新增 training/splits/v3_400_scene_split.json，按 scene_id 而不是按 PNG 划分：

1. 同一 scene 的 Clear、mild、medium、strong 永远在同一个 split；
2. 同一个 reference_id 不得跨 train/validation/test；
3. 按 lighting_mode 和 layout_family 做可复现分层；
4. 保存 split seed、规则和 release tag；
5. validation/test 不参与采样步数或超参数选择。

当前 400 个 scene 的 release manifest 都是 train。若只使用这 400 个 scene，建议首版使用 320/40/40 的 train/validation/test；这是训练协议建议，不是仓库中已经存在的 split。若 400 个全部用于训练，则必须另行取得独立验证数据。

### 2. 实现训练模块

建议新增：

    training/
    ├── dataset.py       # pairs.jsonl、PNG、共享 GT
    ├── splits.py        # scene-level split
    ├── diffusion.py     # beta schedule、q_sample、reverse sampler
    ├── model.py         # conditional Transformer denoiser
    ├── train.py         # checkpoint、日志和训练入口
    ├── sample.py        # DDPM/DDIM-like 推理
    └── configs/baseline.yaml

loader 最小验收标准：

1. 每条样本读出 c、x0、scene_id、water_id 和 lighting_mode；
2. c 与 x0 尺寸、通道和颜色归一化一致；
3. 三档退化都指向同一个 Clear target；
4. 16-bit PNG 读取、归一化和反归一化通过测试；
5. crop/flip 等增强不改变 pair 语义；
6. 可由 index 追溯到对应 volume 和 scene。

### 3. 单 batch overfit

只取一个 scene 或很小的 batch，固定 seed 和 timestep，确认 xt、c、ε 的 shape/range，训练若干步并确认 loss 下降；保存 c、x0、xt 和预测结果，检查 channel、上下方向和 16-bit 处理。单 batch 无法 overfit 时，不要直接启动长时间全量训练。

### 4. 训练和采样 baseline

第一版记录 Release tag、四卷文件名、split、crop、T、beta schedule、timestep sampling、模型宽度和 block 数、optimizer、learning rate、batch/accumulation、Python/PyTorch/设备、commit、seed、checkpoint 和 validation 样本。

采样先实现容易检查的 uniform reverse sampler，再比较 DDIM-like skip sampling，最后才实现 piecewise 或 search-based timestep schedule。每次只改变一个主要因素。

### 5. 评估和消融

保留直接监督的 degraded → clear baseline，再与 conditional diffusion 比较。至少比较：单一 severity 与三档混合、conditional 与 unconditional、Transformer 与轻量 U-Net、uniform 与 non-uniform sampling、不同 lighting mode，以及未见 scene 上的泛化。指标可包括 normalized RGB L1、PSNR、SSIM 和明确界定的水下增强指标；当前仓库没有任何模型指标。

## 仓库结构

    seabed_dataset_v3/
    ├── README.md
    ├── README_v3_500_scenes.md
    ├── assets/                         # 资产目录、来源、许可证和 SHA-256
    ├── configs/                        # v3 首轮、校准和 500-scene 计划
    ├── src/
    │   ├── assets/                     # 海底、生态和外部资产
    │   ├── labels/                     # depth / semantic / instance / mask
    │   ├── water/                      # native Cycles water volume
    │   ├── contracts.py                # schema、hash、checkpoint contract
    │   ├── lighting.py                # lighting plan
    │   ├── pipeline.py                # scene pipeline
    │   ├── quality.py                 # RGB/几何质量检查
    │   ├── randomization.py           # 可复现随机化
    │   ├── render.py                  # Cycles 渲染与 PNG/EXR 保存
    │   ├── scene.py                   # Blender 场景构建
    │   ├── state.py                   # state digest
    │   └── storage.py                 # metadata 封存
    ├── tools/                         # batch、恢复、归档和验证工具
    ├── tests/                         # contract、标签、物理和输出检查
    ├── release_metadata/v3_400/       # 400 组发布元数据
    ├── reports/                       # 本地校准、验收和运行证据
    └── requirements-validation.txt    # 纯 Python 验证依赖

大体积 outputs/、previews/、Blender scan cache、失败 attempt 和运行日志按 .gitignore 排除在 Git 历史之外。训练 ZIP 通过 GitHub Release 分发，不把大二进制直接提交到 Git。

## 环境要求

纯 Python 检查依赖位于 [requirements-validation.txt](requirements-validation.txt)：numpy 2.3.4、OpenEXR 3.4.15、Pillow 12.3.0。可创建独立环境：

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-validation.txt

重新生成或扩展 batch 需要 Blender 5.2.1 LTS、Blender 自带 bpy、Cycles、Metal 或明确记录的 CPU fallback、完整资产、calibration gate 和足够磁盘。直接执行 python3 tools/run_500_scenes.py 会因为没有 bpy 而失败；该脚本必须由 Blender CLI 调用。

当前仓库没有 requirements-training.txt，也没有锁定 PyTorch 版本。实现训练模块时应根据实际 Python/torch/Metal 环境单独记录训练依赖。

## 快速开始

    git clone https://github.com/weekjvj-dotcom/seabed_dataset_v3.git
    cd seabed_dataset_v3

下载并按卷解压后，检查每个 ZIP：

    for archive in data/v3_400_training_ready/*.zip; do
      python3 -m zipfile -t "$archive"
    done

检查索引的最小脚本：

    python3 - <<'PY'
    import json
    from pathlib import Path
    root = Path("data/v3_400_training_ready/parts")
    rows = []
    for path in sorted(root.glob("*/pairs.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert len(rows) == 1200
    assert len({row["scene_id"] for row in rows}) == 400
    assert {row["water_id"] for row in rows} == {"mild", "medium", "strong"}
    print("pairs:", len(rows), "scenes:", len({row["scene_id"] for row in rows}))
    PY

随后先实现 loader smoke test 和单 batch overfit，再开始全量训练。

## 源码验证与批量生成

纯 Python batch contract 测试：

    python3 -m unittest discover -s tests -p 'test_batch500_contract.py'

如果本机仍保留未打包的 outputs/v3_500_scenes_final/，可执行：

    python3 tests/validate_batch_outputs.py outputs/v3_500_scenes_final --expected-scenes 400

它检查 scene 数、manifest row、Clear/degraded 配对、共享 GT、RGB/PNG header、标签和单光照约束。GitHub clone 默认不包含 raw output，Release ZIP 也不是这个 flat output 的替代品。

若明确要继续生成 scene_0401–scene_0500，使用一次只运行一个 Blender 进程的 supervisor：

    python3 tools/serial_blender_batch.py --config configs/v3_500_scenes.json --template reports/v3_scan_cache.blend --start 401 --end 500

单 scene 的底层形式：

    /Applications/Blender.app/Contents/MacOS/Blender --background reports/v3_scan_cache.blend --disable-autoexec --python-exit-code 1 --python tools/run_500_scenes.py -- --config configs/v3_500_scenes.json --start 1 --end 1 --template-cache --quit

不要并发启动多个 Cycles F12/Blender batch 进程；内存有限时优先使用串行 supervisor，并保留 accepted checkpoint、失败 attempt 和日志。

## 质量保证与可复现性

accepted scene 进入发布清单前须通过：

1. 每 scene 一个 Clear、三个 degraded；
2. 四态 scene identity、state digest 和 reference identity 一致；
3. 当前批次每 scene 只有一个 lighting mode；
4. 几何、相机、对象变换和可见性约束通过；
5. PNG/EXR header、尺寸、通道、有限值和 RGB 质量通过；
6. depth、semantic、instance、valid mask 对齐；
7. source/config/calibration/runtime/asset provenance 一致；
8. 四卷归档成员和 CRC 检查通过。

已发布证据：

- [release_metadata/v3_400/validation.json](release_metadata/v3_400/validation.json)
- [release_metadata/v3_400/release.json](release_metadata/v3_400/release.json)
- [release_metadata/v3_400/manifest.jsonl](release_metadata/v3_400/manifest.jsonl)
- [release_metadata/v3_400/batch_manifest.json](release_metadata/v3_400/batch_manifest.json)
- [reports/v3_400_validation_after_clear_rename.json](reports/v3_400_validation_after_clear_rename.json)

## 局限与解释边界

1. 这是合成、可控的 Blender/Cycles 数据，不是真实海域采集数据；
2. absorption、scatter、IOR、bounce、samples 和颜色管理是生成配置，不等于实测海水光谱或绝对辐亮度标定；
3. 深度是几何交点距离和 camera-space 深度，不是水下光程、声学距离或真实传感器测距；
4. 当前 400 组每个 scene 只有一个 lighting mode，不能支持完整的同 scene 三光照控制变量结论；
5. 程序化和扫描资产不代表真实物种比例、活体姿态、采集地点或野外生态统计；
6. 当前数据不包含粒子、海浪、传感器噪声、复杂 caustics 等未写入配置的因素；
7. 500-scene 配置的计划数量、计划 split 和计划光照计数不能当作当前 400-scene release 的实际结果；
8. 当前仓库没有训练模型、checkpoint、PSNR/SSIM 或下游任务结论；
9. 400 个合成 scene 不保证训练收益、真实域泛化或增强效果；
10. 改变几何、资产、相机、光照定义、水体定义或 GT 规则时，必须重新生成受影响的 GT 和 provenance。

## 故障排查

### 没有 bpy

tools/run_500_scenes.py 是 Blender Python 脚本，不是普通 CPython 脚本。用 Blender CLI 调用它；纯 Python 测试只能运行不依赖 bpy 的模块。

### 解压后索引覆盖

每个 volume 都有同名 pairs.jsonl 和 dataset_manifest.json。按卷解压到 parts/<volume-name>/，再由 loader 遍历或合并。

### clone 后找不到 raw output 或 scan cache

这是预期行为。大体积 outputs/ 和 reports/v3_scan_cache.blend 不进入 Git 历史。只训练时下载 Release ZIP；要重新生成必须提供完整本地工作区、资产、cache、calibration 和磁盘。

### Metal/Cycles 内存压力

不要并发渲染。使用 serial_blender_batch.py、保持 template-cache、缩小范围，并检查 accepted checkpoint 后继续。不要删除 accepted 数据来掩盖失败。

### 训练图像翻转或颜色异常

检查 16-bit 是否保留、RGB channel 顺序、是否重复上下翻转，以及 [-1, 1] 归一化/反归一化是否互为逆操作。

## 许可证与资产来源

本仓库当前没有单独的项目级 LICENSE 文件，因此在公开代码或数据前应补充明确的项目许可声明。第三方资产不自动继承项目代码许可，应以 [assets/asset_catalog.json](assets/asset_catalog.json) 和 [assets/README.md](assets/README.md) 中的来源、许可证、下载日期和 SHA-256 为准。

主要来源包括 Poly Haven 的 rock_3、coral_ground_02 PBR 资产，以及 Smithsonian 3D Open Access 中登记的 CC0/Public Domain 候选扫描资产。请不要把第三方机构描述成项目背书，也不要把扫描对象的博物学说明直接解释为渲染参数或实测海水信息。

## 引用

方法参考论文：

> Yi Tang et al., “Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy,” arXiv:2309.03445, 2023.
> [论文页面](https://arxiv.org/abs/2309.03445)

论文中的 x0、xt、c、timestep conditioning、Transformer denoiser、L1 noise loss 和 skip sampling 是方法参考；本仓库的 Blender 资产、native Cycles 水体、GT 定义和 400-scene release 是独立工程实现，不能直接宣称复现论文实验结果。

## 贡献方式

提交新的生成规则、资产、训练代码或实验结果时，请提供修改文件、影响范围、可复现命令、运行环境、随机种子、对应测试/报告，以及第三方资产的来源页、许可证、下载日期、颜色空间和 SHA-256。

任何改变几何、相机、光照控制变量、水体模型、标签定义或训练 split 的修改，都应使用新的配置/输出根目录或实验标识，不能静默覆盖已经发布的 v3-400-training-ready。
