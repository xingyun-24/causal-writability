# 三个任务的交互 PCA 图：统一格式

## 先统一画的对象

用于比较“四组结构”的主视图统一为：先在 fit 集的 matched differences 上拟合 PCA，再把 aligned/conflict 原始激活投影到同一组 PC1/PC2/PC3。

- 每个 matched pair 对应两个点，不能把原始激活点和差值点混在一张图里。
- 差值本身的三维图可以另设一个视图，明确标为 matched difference。
- Spring/Pendulum 分组为红慢、红快、蓝慢、蓝快；Free Fall 应写红低重力、红高重力、蓝低重力、蓝高重力，不把重力标签改叫频率。
- 明确 block、FM call 是单次还是拼接，以及 fit/held-out 范围。旧全样本 PCA 的图只作历史参考，不能标为新的 train-only 结果。
- 第三维必须来自实际 PC3，不能补零、加抖动或换一个物理量后仍称 PC3。只有二维坐标时先补数据。

## 统一视觉与交互

1. 直接沿用袁满原 HTML 的 Plotly 2.35.2 `plotly_white` 页面：字体、标题、坐标轴和工具栏使用该模板的相同默认样式，不再单独设计一套。
2. 同样的画布比例、图例位置、轴网格和初始视角。默认显示 PC1/PC2/PC3，坐标轴采用相同的缩放规则；不得为了显得四组更分离而分别归一化每个组。
3. 与袁满模板一致，圆点为 aligned、菱形为 conflict；点色用 Viridis 表示生成物理量（Spring/Pendulum 为生成频率，Free Fall 为生成重力）。同一张图所有组共用色标范围。输入红/蓝及真实动力学类别放入 hover；不能把点色误称输入颜色。
4. marker size=6、opacity=.86、边框色 #202124、边框宽 .35；默认不连线，不加装饰箭头、长公式或大段解释。
5. 同样支持拖拽旋转、滚轮缩放、复位和图例筛选。补充着色可选真实物理参数或相位，但三个任务中同一种着色的含义必须一致。
6. Hover 统一显示样本 ID、aligned/conflict、输入类别、真实与生成物理量、PC1/2/3、fit/held-out。解释方法与样本口径放在随附说明，不塞进图中央。

## 统一交付文件

- `interactive.html`：可单独打开的旋转视图；最终交付不要依赖服务器私有路径。
- `data.json`：图中使用的实际点、轴定义和分组。
- `preview.png`：同一默认视角的静态预览。
- `build.py`：从数据生成网页和预览的入口。

`data.json` 的最低字段：

```json
{
  "task": "spring | pendulum | freefall",
  "point_type": "raw_activation_projection | matched_difference",
  "basis_source": "matched_difference",
  "basis_fit_split": "fit | legacy_all | unknown",
  "block": 0,
  "fm_representation": "说明具体 call 或拼接方式",
  "axes": ["PC1", "PC2", "PC3"],
  "points": [
    {
      "pair_id": "实际 ID",
      "condition": "aligned | conflict | difference",
      "split": "fit | heldout | legacy | unknown",
      "input_color": "red | blue",
      "dynamics_band": "slow | fast | low_g | high_g",
      "physical_true": null,
      "physical_generated": null,
      "pc1": 0.0,
      "pc2": 0.0,
      "pc3": 0.0
    }
  ]
}
```

上面只是字段示例，不是可作图的假数据；缺失量用 null，并在说明中标明，不猜测补齐。

## 已找到的参考

- 袁满：`raw_residual_pca_projection_interactive.html`，确有四组柱状点云，来自旧 Block 2、125 matched pairs 的原始激活投影；不是当前 Block 1、64-fit/64-held-out 的新结果。
- Spring：`spring-route-geometry.html`，seed 3408、50K、B6，128 个 held-out 差值点。它用 route-offset + 两个相位平面坐标，不是原始 PC1/2/3 的四组原激活视图。
- Pendulum：当前已检查的分支/材料里没有找到独立的可旋转 3D 页面。二维 phase-plane 数据已有；若做同口径四组图，需要原始激活的三维投影，不能直接把二维图外观改成 3D。

两份现有 HTML 与预览保存在 `../interactive-pca-candidates-20260912/`，原文件未改。本轮确定格式，未把历史图替换为论文中的最新证据，也未生成新的 Pendulum 坐标。
