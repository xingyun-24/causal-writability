# Free fall：仅拟合集计算 PCA 基底

完成日期：2026-09-09。服务器：`yuanman@10.234.161.2`。

原始数据来自服务器项目
`/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2` 下的
`results/freefall-hist32-step100000-block1-pca128/shard*/deltas.npy`。
PCA 重算阶段复用已提取的残差。后续已完成 64 对 held-out 的恢复视频生成及结果重算，见 [恢复实验报告](RECOVERY.md)。

## 划分与方法

- 128 对样本，每对使用 aligned − conflict 的完整 condition-token residual。
- 64 对用于拟合，64 对用于 held-out；每组均为 low 32 对、high 32 对。
- 以原回归的 pair_id 排序、分重力区间交替划分为起点，修复一处共享 base_seed 跨组问题：
  `eval_50208_high` 从 held-out 移至拟合，`eval_50101_high` 从拟合移至 held-out。
  交换对象由同区间、独立 seed、重力值最接近确定，不依据本次 PCA 或回归表现选择。
- 所有同 base_seed 样本均在同一组。完整名单及调整记录见 `results-grouped/split_manifest.json`。
- 延续原实验不中心化的残差 PCA（严格说是未中心化 SVD），采用 float64 计算。
- 只对训练矩阵 `X_train` 的 Gram 矩阵分解。训练基底确定后，才读取 held-out 残差进行投影。
- 后续 `color × [1,g,sqrt(g)]` 回归也只使用相同的 64 对拟合样本。固定报告 2/3/4 个主成分，不用 held-out 选参数或主成分数。

## 结果

| 主成分数 | 拟合集保留能量 | Held-out 保留能量 |
|---:|---:|---:|
| 1 | 91.9834% | 92.0482% |
| 2 | 99.5452% | 99.5544% |
| 3 | 99.7110% | 99.6880% |
| 4 | 99.7263% | 99.6935% |
| 8 | 99.7584% | 99.7004% |
| 16 | 99.8043% | 99.7048% |
| 32 | 99.8774% | 99.7071% |
| 64 | 100.0000% | 99.7088% |

能量比例为该组投影分数的平方和 / 原始残差的平方和，不是中心化解释方差比例。
前两个主成分的 held-out 相对重建误差为 6.6751%（Frobenius 范数比）。

使用前两个主成分做颜色—重力回归，held-out 联合 RMSE 为 0.932759；
PC1/PC2 的 R² 分别为 0.99999574 / 0.99139394。
完整结果见 `results-grouped/summary.json` 和各 `pca*_colour_1_g_sqrtg_fit.json`。

## 工件与复用

`results-grouped/train_only_pca.npz` 保存 `pair_ids`、`train_mask`、`heldout_mask`、
`train_vectors`、`eigenvalues`、全部 128 对的 `scores`、原残差形状与平方范数。
`train_vectors` 的行只对应 `pair_ids[train_mask]`。
为避免旧代码把所有样本当作基底拟合样本，本文件刻意不提供旧版 `vectors` 字段。

特征空间基底以小矩阵和源残差隐式存储，无需另存约 17 GB 的稠密基底：

```python
V = X_train.T @ train_vectors / np.sqrt(eigenvalues)[None, :]
scores_new = (X_new @ X_train.T) @ (train_vectors / np.sqrt(eigenvalues)[None, :])
reconstruction = scores_new[:, :k] @ V[:, :k].T
```

已有 rollout 脚本依赖旧的全样本 PCA 格式，不能直接加载此工件。
PCA、held-out 投影评估和 PC2/3/4 回归已完成。
后续使用独立脚本 `recover_train_only_pca.py` 完成了直接残差恢复及 E3 recovery；
该恢复使用实际 held-out 残差的投影分数，不使用回归预测分数，详见 `RECOVERY.md`。

## 验证与适用范围

验证通过：小矩阵与直接 SVD 等价、基底正交、投影一致、扰动 held-out 不改变训练基底；
实际结果检查了唯一 pair_id、互斥 split、无跨组 base_seed、训练分数恒等式与投影能量上界。
原结果保留。划分交换了两对，且新计算采用 float64，新旧数值比较应考虑这两个差别。

这 128 对原先已按 aligned/conflict 的 E3 表现筛选。本次 held-out 结论适用于这一筛选群体，
并且它们已在旧分析中被查看过；这是修正计算隔离后的评估，不是全新未使用的测试集。

## 保存位置与重跑

服务器 `/data` 已满，未能把脚本或新结果写回原项目目录。
服务器临时副本：`/tmp/yuanman-freefall-pca-trainonly-6oD652iO/`，结果在 `results-grouped/`。
本地 `freefall_pca_redo/` 保存了脚本及全部输出；临时目录不应当作长期存储。

在服务器上执行以下命令可重跑。`--out` 必须是不存在的新目录。

```bash
ROOT=/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2
SCRIPT=/tmp/yuanman-freefall-pca-trainonly-6oD652iO/fit_train_only_pca.py
OPENBLAS_NUM_THREADS=4 python3 "$SCRIPT" --self-test
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 "$SCRIPT" \
  --pca-root "$ROOT/results/freefall-hist32-step100000-block1-pca128" \
  --dataset-dir "$ROOT/data/raw-v3-fixedpos-freefall-eval320/eval" \
  --pair-manifest "$ROOT/results/freefall-hist32-step100000-eval320-strict/recovery_pairs_strict128.json" \
  --out /tmp/freefall-pca-trainonly-new-run --device cuda:3
```
