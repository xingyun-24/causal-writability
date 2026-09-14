# Pendulum 属性绑定冲突：机制假设与关键验证实验

## 1. 文档目标

本文总结 Pendulum 实验中观察到的属性绑定冲突现象，并将后续工作收敛为三个可干预、可证伪的机制假设。

当前核心观察来自 `frequency_color_circle` 的 long-context 条件：训练只包含 red–low-frequency 与 blue–high-frequency 配对；测试固定低频物理轨迹并将颜色从红色连续扫描到蓝色时，输出颜色基本跟随输入，聚合频率仍可保持在低频侧，但生成幅度随配对不一致程度持续下降。

现有逐样本结果还提示两个重要事实：

1. 从完全一致的 low-red 到完全冲突的 low-blue，64/64 个物理状态的生成幅度均下降；端点配对的幅度比中位下降约 0.545。
2. 完全冲突的 low-blue 条件下，输出频率在逐样本层面接近低/高频双模态，而不是所有样本都稳定保持低频；但低频和高频两个输出分支都出现相近的幅度下降。

因此，需要解释的不是一般性的 OOD 失败，而是：

> 当长历史提供的物理证据与训练形成的颜色捷径冲突时，模型为何以及如何把冲突代价系统地转移到幅度等非目标自由度上？

相关现有数据：

- [`geometry_analysis/01_amplitude_and_length_vs_input_alpha_data.csv`](../reports/pendulum-frequency-color-50k/assets/analysis/low/geometry_analysis/01_amplitude_and_length_vs_input_alpha_data.csv)
- [`color_analysis/02_generated_color_vs_frequency_data.csv`](../reports/pendulum-frequency-color-50k/assets/analysis/low/color_analysis/02_generated_color_vs_frequency_data.csv)
- [`frequency_color_shape/per_sample_enriched.csv`](../runs/pendulum/frequency_color_frequency_scan_11x13/analysis/tables/per_sample_enriched.csv)

## 2. 分析原则

后续实验遵守以下原则：

1. 以 `physical_state_id` 为配对统计单元，固定真实轨迹、相位、幅度和 generation noise，再改变属性配对。
2. 不只比较聚合中位数；必须检查每条视频的联合分布 \((\hat\omega,\hat A,\hat\alpha)\)。
3. 机制证据必须包含干预：阻断候选自由度、移除冲突来源，或直接干预模型内部的信息通路。
4. 将“模型机制”与 VAE、检测器、全窗正弦拟合等测量伪影分开。
5. short/long 是分别训练的模型，不能把两者差异直接解释为同一模型上的纯 context 干预。关键 context 实验应尽量使用单模型的 history masking/context dropout 设计。

三个假设位于不同层次，可能同时成立：H1 描述一般的误差分配原则，H2 解释为何当前优先牺牲幅度，H3 描述扩散生成内部如何实现这种分配。

### 2.1 `color_circle` 与 `color_shape`–circle 的初步比较

现有结果允许做一个重要但尚未完全匹配的比较：

- `frequency_color_circle`：训练为 low-red-circle / high-blue-circle；
- `frequency_color_shape`：训练为 low-red-circle / high-blue-square，但测试时取 circle 输入；
- 两者均取 long-context、低频、circle、red→blue 扫描。

在 `frequency_color_shape` 中，circle 本身是低频 cue。因此 low-blue-circle 同时包含：历史与 shape 支持低频、颜色支持高频。它比 `frequency_color_circle` 的 low-blue 输入多一个支持低频的外观证据。

当前可比切片的结果为：

| input color α | `color_circle` amplitude ratio | `color_shape`–circle amplitude ratio | `color_circle` median ω̂ | `color_shape`–circle median ω̂ |
|---:|---:|---:|---:|---:|
| 0.0 | 0.984 | 0.969 | 2.576 | 2.587 |
| 0.5 | 0.781 | 0.760 | 2.572 | 2.589 |
| 1.0 | 0.437 | 0.411 | 2.537 | 2.310 |

11 点 amplitude-ratio 中位曲线的相关系数约为 `0.9996`，平均绝对差约为 `0.022`。但在完全冲突的 blue 端：

- `color_circle` 约 `48%` 的样本进入 high-frequency route；
- `color_shape`–circle 只有约 `12%` 进入 high-frequency route。
- 两者输出颜色 α 的中位数分别约为 `0.995` 和 `0.907`；`color_shape`–circle 的输出并没有同样蓝，但幅度反而略低。

因此，额外的 circle→low cue 明显改变了最终频率选路，却几乎没有改变幅度抑制曲线。这带来三个约束：

1. 幅度下降不要求最终样本进入错误频率 route；
2. 幅度下降也不能主要由低/高频样本混合后的聚合中位数解释；
3. 幅度下降不只是生成颜色蓝度的测量副作用；
4. 连续的幅度抑制与离散的频率选路很可能是可分离的两个过程，前者发生在选路之前或与选路并行。

`frequency_color_shape` 内部的配对 shape 干预进一步支持这一点。在 `ω=2.6, α=1` 时，将 circle 改为 square，使 high-route 比例约从 `0.125` 增至 `0.50`，但 amplitude ratio 只从 `0.411` 变为 `0.374`。相反，在固定 shape 时把颜色从 red 改为 blue，配对 amplitude ratio 的中位变化约为 `−0.52` 到 `−0.55`。即颜色干预对幅度的作用远大于 shape 干预，而 shape 更像是在局部改变 route probability。

完整端点也显示 shape-only 冲突的代价相对较小：`ω=5.8, blue-circle` 中 shape 指向低频、历史和颜色指向高频，模型仍有 `100%` 样本走 high route，amplitude ratio 约为 `0.868`；对应的 all-aligned `blue-square` 约为 `0.960`。因此，现有行为证据更像是颜色 cue 主导幅度抑制、shape cue 较弱或主要影响选路边界。但它尚不能区分“shape 通路确实较晚”与“模型整体几乎没有学会 shape”；这正是后续同输入 cue 干预需要解决的问题。

这仍是描述性证据，不是最终跨模型因果比较，因为两批数据的真实频率、幅度、phase 数量和 diffusion repeats 不完全一致：`color_circle` 汇总 64 个低频随机状态，而 `color_shape` 切片固定 `ω=2.6`、`A=0.135`，包含 8 个 phase × 2 repeats。后续必须用同一输入 bank 和相同 generation seeds 重新评估两个 checkpoint。

## 3. 假设 H1：弱约束自由度承担绑定冲突

### 3.1 机制

训练联合分布只支持 red–low 与 blue–high。low-blue 或 high-red 条件位于训练联合支持之外。面对冲突输入，模型优先保留两个强约束：

- 长历史提供的物理信息；
- 清晰、局部且易生成的颜色信息。

剩余冲突被转移到损失曲率较软、训练中没有被显式绑定的物理自由度，例如幅度、相位、摆动中心或摆长。当前幅度下降是这种 **slack-variable allocation** 的一个实例。

### 3.2 可证伪预测

1. 如果阻止幅度下降，冲突误差不会消失，而会迁移到频率、颜色、相位、阻尼或其他几何量。
2. 在 `amplitude_color_circle` 中制造 color–amplitude 冲突时，某个未绑定自由度也应系统偏离；frequency 是候选，但不要求它必然成为唯一承载者。
3. 在训练中加入少量反事实配对支持后，非目标物理量的异常应显著减弱。

如果约束幅度后所有输出量仍同时正确、且加入反事实训练配对不改变现象，则 H1 不成立。

### 3.3 关键实验：阻断自由度并观察误差迁移

选择 long-context 的四个端点条件：

- low-red：aligned control；
- low-blue：conflict；
- high-blue：aligned control；
- high-red：conflict。

对 conflict 和 aligned 条件分别进行自由生成与 amplitude-constrained 生成。约束可按实现成本依次选择：

1. 在条件端额外提供一小段保持真实幅度和边界速度的未来轨迹，再生成剩余未来；
2. 在采样中加入可微的轨迹幅度/边界连续性 guidance；
3. 训练加入显式的幅度连续性或轨迹状态辅助约束。

主要比较：

\[
\Delta E_\omega,\quad
\Delta E_{color},\quad
\Delta E_{phase},\quad
\Delta E_{length},\quad
\Delta damping
\]

在阻止 \(\hat A\) 下降前后的变化。

支持 H1 的决定性结果是：conflict 条件下幅度被约束后，错误显著、可重复地迁移到另一个输出因子；aligned control 中没有同等迁移。

### 3.4 训练支持干预

只训练 long-context 对照模型，将反事实配对占比设置为：

\[
p_{cf}\in\{0,0.05,0.50\}.
\]

- `0`：原始完全绑定；
- `0.05`：少量 low-blue/high-red 支持；
- `0.50`：颜色与频率去相关。

若幅度异常随 \(p_{cf}\) 增大而快速减弱，说明现象来自联合支持缺失，而不是颜色渲染或测量器本身。

## 4. 假设 H2：幅度下降是在抑制频率信息载体

### 4.1 机制

频率必须通过非零幅度的时间运动才能显现。两个频率候选轨迹的差异近似为：

\[
\Delta\theta(t)
=A\left[
\cos(\omega_Lt+\phi)-\cos(\omega_Ht+\phi)
\right].
\]

因此两个候选未来之间的轨迹或像素冲突能量大致随 \(A^2\) 缩放。模型压低幅度后，低频和高频未来会变得更难区分，同时仍能保留球体颜色。这是一种 **carrier suppression**：不是任意选择一个自由度，而是主动削弱承载冲突信息的运动载体。

这一机制天然不完全对称。幅度下降能够抹除频率证据，但改变频率不能同样直接抹除 large/small amplitude 的最大位移差异。

`color_circle` 与 `color_shape`–circle 的幅度曲线近乎重合、但频率 route probability 明显不同，进一步说明缩幅更可能是对 color–motion 不兼容性的直接响应，而不是错误频率 route 的结果。

### 4.2 可证伪预测

1. 幅度下降强度应随真实频率与颜色暗示频率之间的分离、预测 horizon 和累计相位冲突增加。
2. 同样的 color–amplitude 冲突不一定产生与当前幅度曲线镜像对称的频率曲线。
3. 在 frequency–color 冲突中人为维持较强、连续的运动载体后，模型必须改走其他冲突消解路径。
4. 若缩幅只是一般的 slack-variable 选择，它不应稳定呈现上述与 \(\Delta\omega\)、horizon 和运动可见度有关的特异性交互。

### 4.3 关键实验：因子交换与对称性检验

只使用 long-context，分别测试：

#### A. `frequency_color_circle`

\[
\omega_{true}\in\{\omega_L,\omega_H\},
\qquad
color\in\{red,blue\}.
\]

#### B. `amplitude_color_circle`

\[
A_{true}\in\{A_S,A_L\},
\qquad
color\in\{red,blue\}.
\]

两组都包含两个 aligned 与两个 conflict 端点。第一阶段无需完整 11 点扫描：每个模型先使用端点加中点颜色、16 个配对物理状态、每个输入 4 个固定且跨条件复用的 diffusion seeds。只有在端点结果稳定后再扩展连续扫描。

判别标准：

- 若 frequency–color 冲突稳定表现为幅度下降，而 amplitude–color 冲突不表现为镜像的连续频率偏移，更支持 H2；
- 若两种绑定都把误差近似对称地转移给另一物理因子，更支持 H1 的一般误差分配，而削弱 H2 的特异性解释。

### 4.4 冲突能量干预

在 `frequency_color_circle` 中，对少量关键条件交叉扫描：

- 频率分离 \(|\omega_1-\omega_2|\)：小、中、大；
- 真实输入幅度：small band 内低、中、高三点；
- 预测 horizon：16、32、64 帧。

主要检验交互：

\[
\hat A/A_{true}
\sim
C_{bind}
\times |\Delta\omega|
\times T_{future}.
\]

其中配对不一致度无需假定中间颜色对应精确线性频率：低频条件可定义 \(C_{bind}=\alpha\)，高频条件定义 \(C_{bind}=1-\alpha\)。

如果缩幅随频率分离和未来长度显著增强，将直接支持“通过降低运动载体强度来降低冲突能量”的解释。

## 5. 假设 H3：幅度抑制与频率选路是两阶段、可分离的计算

### 5.1 机制

模型可能先在共享的 color–motion 表示中检测到属性不兼容，并连续降低运动能量或幅度；随后，历史、颜色、shape 和 diffusion noise 才共同决定最终选择 low 还是 high frequency route。

在这一两阶段解释中：

1. **连续阶段**：配对不一致度控制幅度抑制，主要受颜色—运动冲突驱动；
2. **离散阶段**：不同 cue 的相对权重和 diffusion seed 控制 low/high route probability；shape 可以影响此阶段，而不必等比例改变幅度。

这能够解释为什么两个模型的幅度曲线几乎一致，但 `color_shape`–circle 更少跳到 high route。原 H3 中的低/高频模式竞争仍可作为第二阶段的实现方式，但不再被视为幅度下降的必要原因。

当前输出行为只支持“两个效应可以分离”，不能单独证明严格的时间顺序；“先抑制、后选路”必须由层/去噪步干预验证。

### 5.2 可证伪预测

1. 在控制最终 frequency route 后，amplitude ratio 仍应随配对不一致度连续下降。
2. shape 干预可显著改变 route probability，但对 amplitude ratio 的影响明显小于颜色干预。
3. 颜色相关 activation 干预应更早或更直接地改变幅度；历史/shape 相关通路干预应更明显地改变后续 route probability。
4. 改变 diffusion seed 可改变最终 route，但在相同冲突强度下，两个 route 都应保持幅度抑制。

如果幅度差异在控制 frequency route 后消失，或任何能改变 route 的 shape/seed 干预都按同等比例恢复幅度，则连续抑制与离散选路不可分，H3 不成立。

### 5.3 关键实验：完全匹配的跨模型与 cue 干预

建立一个两个 checkpoint 共用的 frozen evaluation bank：

- checkpoint：`frequency_color_circle` 与 `frequency_color_shape`；
- 测试输入首先全部固定为 circle；
- \(\omega_{true}\in\{2.6,5.8\}\)；
- \(A_{true}=0.135\)；
- \(\alpha\in\{0,0.25,0.5,0.75,1\}\)；
- 16 个完全相同的 phase states；
- 每个输入 8 个相同且跨 checkpoint/颜色复用的 diffusion seeds。

然后只对 `frequency_color_shape` checkpoint 增加配对的 square 输入，形成同一模型内的 shape `do`-intervention。所有 circle/square 输入除摆球轮廓外保持逐像素和物理状态一致。

对每条样本保存并联合分析：

\[
(\hat\omega,\hat A,\hat\alpha,\hat\phi,
fitRMSE,boundary\ jump).
\]

不能只比较频率中位数。应报告：

- low/high route probability；
- route 内部的幅度分布；
- 同一 seed 随颜色改变的切换位置；
- 前半窗与后半窗的局部 amplitude envelope。

分析时分别估计：

- color 对 amplitude ratio 与 route log-odds 的配对效应；
- shape 对 amplitude ratio 与 route log-odds 的配对效应；
- checkpoint × color × shape 交互；
- 在 low/high route 内部的 amplitude–α 斜率。

第二步只在上述实验确认的关键条件上做模型内部干预：

1. **历史运动通路消融**：遮挡或阻断历史摆球 motion-tube token 到未来 token 的交互；
2. **颜色 token patching**：在保持物理历史不变时，将 aligned red 条件的颜色相关 activation 替换为 blue，反向也做；
3. **层/去噪步定位**：只在早、中、晚层或早、中、晚 denoising timestep 进行 patch，记录频率分支概率和幅度的变化。

支持 H3 的决定性结果是：颜色 activation patch 可以在 route 尚未确定时制造或解除幅度抑制；历史/shape 通路干预则主要改变后续 route probability。若只能找到一个同时、不可分地控制幅度和 route 的通路，则应拒绝两阶段解释。仅观察 attention weight 不算因果证据。

## 6. 必须先完成的测量排除项

以下项目不是额外机制假设，而是进入机制结论前的必要检查。

### 6.1 幅度测量稳健性

对现有输出同时计算：

1. 全窗自由频率正弦拟合幅度；
2. 固定真实频率或固定输出 route 中心频率的拟合幅度；
3. 轨迹 peak-to-peak/2；
4. 前半窗和后半窗的局部 envelope；
5. 边界位置、速度和幅度连续性。

若只有全窗正弦拟合幅度下降，而局部 envelope 与 peak-to-peak 不下降，应把现象解释为相位漂移、阻尼或非平稳轨迹，而不是稳定缩幅。

### 6.2 颜色无关检测与 VAE control

- 使用与目标颜色无关的几何/摆绳跟踪器重新测量轨迹；
- 对完全相同的真实轨迹做 red→blue VAE encode–decode，不经过 diffusion；
- 检查 VAE 重建后是否已经出现颜色依赖的幅度偏差。

如果 VAE-only control 重现主要幅度曲线，应先处理表示或测量问题，不能把它归因于绑定冲突。

## 7. 最小实验集合与执行顺序

为了避免实验扩张，建议只执行以下三个机制实验。

### E1. 完全匹配的 `color_circle` vs `color_shape` cue-factorial

目的：验证当前跨模型比较，并直接检验 H3。

- checkpoint：`frequency_color_circle`、`frequency_color_shape`；
- history：long；
- 两个 checkpoint 先测试完全相同的 circle input bank；
- `color_shape` 额外测试逐状态、逐 seed 配对的 square 版本；
- 使用相同 \(\omega\)、幅度、phase、颜色和 diffusion noise；
- 输出 color/shape 对 amplitude ratio 与 frequency route 的分离效应。

### E2. 阻断候选 slack variable + 反向绑定

目的：直接验证 H1，并区分 H1 与 H2。

- frequency–color 冲突中约束幅度/边界运动连续性；
- amplitude–color 冲突中对候选 frequency 偏移做对应约束；
- 观察错误是否迁移到颜色、目标物理量、相位、阻尼或几何量；
- aligned 条件作为约束本身副作用的 control。
- 同时比较 `amplitude_color_circle`，检查 color–amplitude 冲突是否产生镜像 frequency 曲线；不对称结果支持 H2。

### E3. 移除冲突来源并定位内部通路

目的：验证现象是否确实由学习到的绑定关系产生，并定位其生成实现。

- 训练支持干预：比较 \(p_{cf}=0,0.05,0.50\)；
- 对固定 checkpoint 和固定 noise 做颜色 token patching、历史运动通路消融；
- 只在 E1/E2 已确认的关键条件上进行，不做全网格内部分析。

## 8. 统一统计与判定

### 8.1 主要结果变量

```text
amplitude_ratio = amplitude_hat / amplitude_true
frequency_error = abs(omega_hat - omega_true)
color_error = abs(alpha_hat - alpha_input)
phase_error
local_damping_or_envelope_slope
length_error
boundary_position_error
boundary_velocity_error
low_high_route
fit_rmse
```

### 8.2 统计单元

- 以 `physical_state_id` 进行 paired bootstrap 或层级建模；
- diffusion repeats 嵌套在 physical state 内；
- 不把视频帧或同一状态的不同颜色当作独立样本；
- 同时报告中位效应、配对置信区间、route probability 和有效率。

建议的基础模型为：

\[
amplitude\_ratio
\sim
C_{bind}\times model\_family
+frequency\_error
+color\_error
+fitRMSE
+(1+C_{bind}\mid physical\_state).
\]

机制结论主要依赖干预前后的配对效应，而不是单个回归系数。

## 9. 结果解释决策表

| 关键结果 | 主要支持 | 主要削弱 |
|---|---|---|
| 约束幅度后，错误迁移到频率/颜色/相位等变量 | H1 | “幅度只是无关副产物” |
| 少量反事实训练配对使全部异常迅速减弱 | H1 | 固定 renderer/VAE 偏差 |
| frequency–color 冲突缩幅，但 amplitude–color 冲突不产生镜像频率曲线 | H2 | 完全对称的自由度交换解释 |
| 缩幅随频率分离、预测 horizon 和累计相位冲突增强 | H2 | 与频率载体无关的一般退化 |
| 两模型幅度曲线匹配，但 route probability 不同 | H2、H3 | “缩幅只是错误 route 的结果” |
| 控制 frequency route 后，幅度仍随配对不一致度下降 | H3 | route mixture 解释全部幅度效应 |
| shape 干预主要改变 route，颜色干预主要改变幅度 | H3 | 单一不可分的 cue-to-output 通路 |
| 颜色 activation patch 先改变幅度，历史/shape patch 后改变 route | H3 | 单阶段、不可分的决策解释 |
| 只有全窗拟合幅度下降，局部 envelope 不下降 | 测量/非平稳解释 | 稳定 carrier suppression |
| VAE-only 重建已经重现主要幅度曲线 | VAE/检测问题 | 绑定冲突的生成机制解释 |

## 10. 最终目标

这组实验不应只回答“模型是否使用 shortcut”，而应回答三个更具体的问题：

1. 冲突是否被系统地分配给弱约束自由度？
2. 当前选择幅度是否因为幅度控制频率信息的可见性？
3. 连续幅度抑制与离散频率选路是否由不同 cue、层或去噪阶段控制？

只有出现以下证据链时，才能提出较强的机制结论：

> 配对冲突引起非目标变量偏离；阻断该变量后误差可预测地迁移；移除训练绑定或因果干预颜色/运动内部通路后，该偏离被解除。
