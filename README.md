# 原生 Cycles 水下数据 v3

v3 是与 `seabed_dataset_v2/` 隔离的真实化版本：它保留可控的 3D 几何、水体、相机、Depth/Semantic/Instance 真值链路，并接入 CC0 PBR 与少量经审计的扫描几何。当前权威交付为 `outputs/v3_first_round/`；不要覆盖该目录或把新的配置混写进去。

## 查看结果

1. 打开离线接触表：`reports/v3_first_round_contact_sheet.png`。
2. 在浏览器打开 `reports/v3_first_round_preview.html`，可按 geometry、lighting 和水体筛选。
3. 在 Blender 打开 `outputs/v3_first_round/scenes/l101_g000.blend`、`l101_g001.blend` 或 `l101_g002.blend`；它们依赖同项目内的 `assets/source/` 外置 PBR 路径，不需要项目 Python 或 autoexec。

## 已冻结配置和证据

1. 正式配置：`configs/v3_first_round.json`，256×256、512 samples、3 geometry states、Natural/Artificial/Mixed、mild/medium/strong。
2. 校准 gate：`reports/calibration_v3/gate.json`。
3. 最终验收：`reports/ACCEPTANCE.md`、`reports/v3_first_round_validation.json`、`reports/native_reopen_verification.json`。
4. 外部资产溯源：`assets/asset_catalog.json` 与 `assets/README.md`。

## 重要边界

1. 当前 PBR height 图只作 Bump；将其改为真实 displacement、替换扫描网格或修改资产文件会改变 source signature，必须使用新配置与新 output root，并重新做 calibration 和 GT 导出。
2. 参考照片不得直接当作背景、海床照片卡片或标签来源。
3. 新正式批次应复制配置，使用新的 `output_root`；不得把新样本混入 `outputs/v3_first_round/`。
