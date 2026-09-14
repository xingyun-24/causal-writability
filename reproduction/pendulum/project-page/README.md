# Pendulum 论文与项目页交付包

本目录固定 `frequency_color_circle` Large、seed 3407、50K 的论文版 Pendulum 身份，并将训练、strict-bank、PCA 和项目页 3-D PCA 所需的配置、名单、代码与命令收在一处。本修正版接入当前论文的 Top-4、按方向拟合的相位 controller；原包的 Rank-1 Stage 4 结果仅作历史记录，不是当前论文的恢复结果。原始压缩包和论文图表均未修改。

## 1. 最终 Short / Long checkpoint

| history | checkpoint（仓库相对路径） | 配置 |
|---|---|---|
| Short | `runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors` | `code/configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml` |
| Long | `runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/long/ckpt/frequency-color-circle-large-long-50k/step-50000.safetensors` | `code/configs/pendulum/Train-frequency_color_circle-large-long-50k.yaml` |

两个配置共用 Large DiT：`dim=1152`、30 blocks、9 heads、FFN 4608、20 inference steps、17 condition latent frames；batch size 32，lr `2e-4`，weight decay `0.01`。Short 使用 `latents/train_short`，Long 使用 `latents/train_long`。checkpoint 体积较大且默认被 Git 忽略，本交付包记录绝对身份而不复制权重。

这两个最终权重已收到并上传至私有 HF 仓库，下载方式见仓库根目录 `CHECKPOINTS.md`。两个启动脚本已与上表统一，并改用当前目录和当前 Python 环境；可通过 `PENDULUM_VENV_PYTHON` 指定解释器。

## 2. Stage 4 代码与依赖

当前论文的入口为 `code/scripts/sshv2/fit_pendulum_harmonic_coordinates.py` 和 `run_pendulum_top4_controller.py`，从当前论文代码原样收录。`code/src/sshv2/experiments/pendulum/stage4_coordinate_difference.py` 提供其坐标计算和 edit 合成依赖；直接运行该旧模块不会自动执行论文的 Top-4 相位方法。交付包同时保留了直接科学依赖：

- `data.py`：摆方程、渲染、Short history mask 和数据配置；
- `evaluation.py`：摆锤轨迹检测与生成频率拟合；
- `mechanism_pca.py`：Wan 运行时、condition latent、20-call block hook、去噪、视频写出与评估工具；
- `stage3_low_rank_causal_route.py`：matched differences、fit-only uncentered SVD 与低秩改写；
- `strict_bank.py`：strict matched-pair 生成与自然生成筛选。

运行时还需要完整仓库的 `src/sshv2/wan/`、`src/sshv2/common/`、`lib/diffsynth/` 子模块与 `models/Wan2.1_VAE.pth`。`code/requirements.txt` 和 `code/setup.py` 是对应仓库版本的安装入口。

## 3. strict bank、PCA 基底与 residual

- 冻结名单：`data/strict_bank_128.jsonl` 和可读表 `data/strict_bank_128.csv`。
- 名单审计：`data/strict_bank_audit.json`。
- 自然生成频率：`data/natural_metrics.csv`。
- PCA 设置与结果：`data/pca_audit.json`。
- 共 128 对：fit 64（32 low + 32 high），held-out 64（32 low + 32 high）。每行包含 `generation_seed`、`aligned_video`、`conflict_video`、物理参数与 split。
- 生成 seed 规则：`generation_seed = base_seed + 23,000,000`；candidate RNG 为 seed 3407 的固定分支流。
- 论文选定位置为 `l*=13`，即零起点 `after DiT block 12`。Stage 3 将每个样本的 20 个 FM call 上 `1088×1152` condition-token residual 按 call 顺序拼接，定义 `d = h_aligned - h_conflict`，只用 fit64 做 uncentered SVD。
- 远端 PCA 基底：`runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short/stage3_low_rank_causal_route/pca/components.npy`。
- 远端 residual bank：`runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short/stage3_low_rank_causal_route/directions.npy`。

重生成 residual 和 fit-only 基底的规范命令（将 `<GPU>` 替换为可用卡）：

```bash
export PYTHONPATH=src:lib/diffsynth
python -u -m sshv2.experiments.pendulum.stage3_low_rank_causal_route \
  --model-name frequency_color_circle \
  --checkpoint runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors \
  --training-config configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --dataset-root data/pendulum/frequency_color_circle_frequency_scan_11x13/circle \
  --manifest "$HANDOFF/data/strict_bank_128.jsonl" \
  --natural-metrics "$HANDOFF/data/natural_metrics.csv" \
  --out-root runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short/stage3_low_rank_causal_route \
  --history short --device cuda:<GPU> --seed-offset 23000000 \
  --hidden-size 1152 --commitment-block-index 12
```

## 4. 安装与端到端命令

本包是完整仓库的补丁包，不是独立运行时。先将 `HANDOFF` 设为本修正版的绝对路径，进入完整 Pendulum 仓库根目录，再覆盖安装包内代码：

```bash
export HANDOFF=/absolute/path/to/pendulum-handoff-corrected-20260913
cp -a "$HANDOFF/code/." .
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src:lib/diffsynth
```

需要额外放置两个最终 checkpoint 和 `models/Wan2.1_VAE.pth`。此包不包含 `src/sshv2/wan/`、`src/sshv2/common/`、完整扫描模块和 DiffSynth，不能只在 `code/` 目录运行上述命令。第 3 节的 residual 重建和第 4.4 节的最终控制器均使用随包冻结名单，不重新筛选 held-out 样本；名单中指向的输入视频须已按原始生成参数准备好。

### 4.1 训练数据与 latent

```bash
python - configs/pendulum/frequency_color_circle_frequency_scan.yaml <<'PY'
from pathlib import Path
import sys, yaml
from sshv2.experiments.pendulum.data import config_from_mapping, generate_dataset
p = Path(sys.argv[1])
c = config_from_mapping(yaml.safe_load(p.read_text())["data"])
generate_dataset(c, Path("data/pendulum") / c.training_manifest_id)
PY

python scripts/sshv2/pendulum_prepare_history_latents.py \
  --source data/pendulum/frequency_color_circle__train_e273989068c3 \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --batch-size 8 --device cuda
```

### 4.2 Short / Long 训练

```bash
bash scripts/sshv2/run_pendulum_frequency_color_circle_large_short_long_50k_train.sh
```

### 4.3 扫描输入数据、自然生成与评估

```bash
python -m sshv2.experiments.pendulum.frequency_color_frequency_scan_data \
  --config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --shape circle \
  --output-root data/pendulum/frequency_color_circle_frequency_scan_11x13/circle

python -m sshv2.experiments.pendulum.frequency_color_frequency_scan_runner predict \
  --dataset-root data/pendulum/frequency_color_circle_frequency_scan_11x13/circle \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --training-config configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml \
  --checkpoint runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors \
  --history short --output-root runs/pendulum/frequency_color_circle_frequency_scan_11x13/circle/short/prediction \
  --device cuda --steps 20 --seed-offset 23000000

python -m sshv2.experiments.pendulum.frequency_color_frequency_scan_runner evaluate \
  --dataset-root data/pendulum/frequency_color_circle_frequency_scan_11x13/circle \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --history short \
  --prediction-root runs/pendulum/frequency_color_circle_frequency_scan_11x13/circle/short/prediction \
  --calibration-root runs/pendulum/frequency_color_evaluation/low_frequency_11color_64states/detector_calibration \
  --output-root runs/pendulum/frequency_color_circle_frequency_scan_11x13/circle/short/metrics

python -m sshv2.experiments.pendulum.frequency_color_frequency_scan_runner color \
  --dataset-root data/pendulum/frequency_color_circle_frequency_scan_11x13/circle \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --history short \
  --prediction-root runs/pendulum/frequency_color_circle_frequency_scan_11x13/circle/short/prediction \
  --frequency-metrics-root runs/pendulum/frequency_color_circle_frequency_scan_11x13/circle/short/metrics \
  --output-root runs/pendulum/frequency_color_circle_frequency_scan_11x13/circle/short/color_metrics
```

### 4.4 当前论文的 Top-4 相位控制器

strict-bank 的 A/B candidate 输入视频、自然生成与筛选由 `strict_bank.py` 完成。`queue_pendulum_large_seed3407_strict_pipeline.sh` 只用于重新构建 bank 和探索性 Stage 2/3；已移除其自动运行旧 Stage 4 的结尾。重新筛选的 bank 和重新选出的层不能直接替代论文冻结名单与 block 12。

完成第 3 节的冻结名单 residual/PCA 重建后，在完整仓库根目录执行以下命令。`directions.npy` 的行顺序必须与随包 `strict_bank_128.jsonl/.csv` 一致。先在各方向的 32 个 fit 样本上拟合 `[1, cos(phi*), sin(phi*)]` 到四个 PCA 坐标的线性规律，其中 `phi*` 是由边界角度与角速度确定的振荡相位；再对各 32 个 held-out 样本生成预测 edit。

```bash
stage3=runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short/stage3_low_rank_causal_route
output=runs/pendulum/paper_top4_reproduction

python scripts/sshv2/fit_pendulum_harmonic_coordinates.py \
  --strict-bank "$HANDOFF/data/strict_bank_128.csv" \
  --directions "$stage3/directions.npy" \
  --components "$stage3/pca/components.npy" \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --out-dir "$output/coordinates"

for direction in A B; do
  python scripts/sshv2/run_pendulum_top4_controller.py \
    --checkpoint runs/pendulum/frequency_color_circle_large_short_long_50k_seed3407/short/ckpt/frequency-color-circle-large-short-50k/step-50000.safetensors \
    --training-config configs/pendulum/Train-frequency_color_circle-large-short-50k.yaml \
    --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
    --manifest "$HANDOFF/data/strict_bank_128.jsonl" \
    --natural-metrics "$HANDOFF/data/natural_metrics.csv" \
    --components "$stage3/pca/components.npy" \
    --coordinate-predictions "$output/coordinates/coordinate_predictions.csv" \
    --strict-bank-root runs/pendulum/unified_mechanism_v2/mechanism/large_seed3407_short/strict_bank \
    --out-root "$output/decoded" \
    --split heldout --direction "$direction" --scales 1.0 \
    --device cuda:0 --hidden-size 1152
done
```

注入位置在解码脚本内固定为零起点 block 12，覆盖 20 次 FM call 的 1088 个 condition tokens。比例 `1.0` 来自当前论文冻结的 fit-only scale selection，见 `data/controller_scale_selection.json`，不得在 held-out 上重新挑选。重拟合 PCA 后应重新生成坐标，不能将新基底与旧 `coordinate_predictions.csv` 混用。

当前论文参考结果位于 `data/coordinate_summary.json`、`data/coordinate_predictions.csv`、`data/decoded_recovery.csv` 和 `data/decoded_recovery_summary.json`：Top-4 预测控制器的 held-out median recovery 约 `.917`。这些是已有论文结果，不代表本修正版已完成新一次 GPU 复现。

解码脚本同时输出的 `metrics_heldout_A/B.csv` 使用历史检测口径，只是中间指标。论文采用后来冻结的 appearance-tolerant 评估：颜色距离 150、最小面积 12、检测比例至少 0.50、连续缺失至多 12 帧、相邻/边界跳跃至多 30 px、长度误差至多 0.10、拟合 RMSE 至多 0.12。原样收录的 `reevaluate_pendulum_decoded_recovery.py` 及其依赖 `reevaluate_pendulum_detection.py` 执行这一重评估。

当四组各 64 条视频均已到位后，使用与实际视频路径对应的 256 行清单统一评估。以下命令用于原始保存的视频路径；对新一轮输出，先用各组实际输出路径构建相同列格式的 `release-csv`，不要将参考结果冒充新生成结果。

```bash
python scripts/sshv2/reevaluate_pendulum_decoded_recovery.py \
  --repo-root "$PWD" \
  --release-csv "$HANDOFF/data/decoded_recovery.csv" \
  --manifest "$HANDOFF/data/strict_bank_128.csv" \
  --experiment-config configs/pendulum/frequency_color_circle_frequency_scan.yaml \
  --output-csv "$output/decoded_recovery.csv" \
  --output-summary "$output/decoded_recovery_summary.json"
```

四组为 natural conflict、full matched、Top-4 oracle、fit-only predicted。仅运行上面的预测控制器命令不代表四组复现均完成。

`data/stage4_summary.json`、`data/stage4_coordinate_predictions.csv` 仍保留原包的旧 Rank-1 / phi0 实验（R≈.945），仅供追溯；不得用其替换当前论文数据。

## 5. 项目页 3-D PCA

生成后的页面与数据位于：

- `assets/pendulum-project-pca.html`：白底 Plotly 静态页面，无自动旋转，支持拖拽旋转、滚轮/手势缩放与 hover；
- `assets/pendulum-project-pca.json`：128 个 held-out endpoint 点和 64 个同基底 difference 点；
- `assets/pendulum-project-pca-preview.png`：静态预览；
- `scripts/capture_pendulum_project_pca.py`：四 shard GPU 激活提取、严格 split/PCA 审计和投影；
- `scripts/plot_pendulum_project_pca.py`：从 JSON 确定性重建 HTML/PNG；
- `scripts/run_remote_capture.sh`：使用 4 块 GPU 并行重建 128 个 endpoint 点。

端点图有 low/high × aligned/conflict 四组，共 128 点；circle 表示 aligned，diamond 表示 conflict，Viridis 表示对应自然生成的 `omega_hat`。差值图使用同一基底并显示 64 个 `aligned - conflict` 点，Viridis 表示该对自然生成频率差 `|Delta omega|`。两幅图都不使用 held-out 拟合、不中心化 held-out，且 PC3 不由二维相位坐标合成。

## 审计边界

- 项目页展示 fit64 matched-difference 基底的 PC1–PC3；论文控制器使用 Top-4。展示三维不等于因果秩为 3，也不能沿用旧 Stage 3 的 `m*=1` 解释当前论文控制器。
- 项目页的 endpoint 坐标是原始 block residual 的投影；difference 坐标才是 matched causal edit 的投影。
- Short 的论文机制 block 是 after block 12。Long 的 common-strict layer control 给出 after block 21，但本 PCA 页不混合 Long checkpoint。
