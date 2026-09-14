# Pendulum frequency–color 实验方案

## 0. 文档范围

本文只讨论 Pendulum 的单一 `frequency_color_circle` 绑定实验：

- 训练绑定：红色对应低频，蓝色对应高频；
- 物理目标：频率；
- 外观变量：球体颜色；
- 球体形状固定为圆形；
- 分别训练并测试 short-history 和 long-history 模型；
- 所有结果变量均从测试生成视频中计算。

本文不会修改
`src/sshv2/experiments/pendulum/config.yaml` 中已经确定的参数。
新增的连续颜色、精确测试频率、重复生成和可视化设定均属于
evaluation-only 补充，不改变训练分布、训练绑定或基础 benchmark。

## 1. 核心科学问题

实验依次回答四个问题：

1. **端点冲突**：真实频率与红/蓝训练绑定冲突时，生成结果跟随物理历史还是颜色 shortcut？
2. **颜色剂量响应**：输入颜色从红色连续变化到蓝色时，输出频率是连续变化、突然切换，还是保持真实频率？
3. **竞争形式**：观察到的“中间频率”是每条视频内部稳定的 compromise，还是不同生成样本在低频、高频路线之间随机混合？
4. **输出外观—运动耦合**：输出球体选择红色、紫色或蓝色时，输出频率是否同步选择低频、中频或高频路线？short 和 long 的耦合强度是否不同？

主要因果问题写为：

\[
\text{固定模型、真实物理轨迹和生成噪声时，改变测试输入颜色，}
\]

\[
\text{输出频率和输出颜色如何变化？}
\]

short–long 比较是两个匹配的训练—测试 pipeline 之间的比较：

- short 模型使用 short-history 训练，并使用 short-history 测试输入；
- long 模型使用 long-history 训练，并使用 long-history 测试输入。

不能把 short/long 解释成同一个固定模型上的纯测试时遮挡干预。

## 2. 不可修改的基础配置

以下值直接来自
`src/sshv2/experiments/pendulum/config.yaml`，在本方案中保持不变。

| 类别 | 固定值 |
|---|---|
| experiment | `pendulum` |
| dataset | `pendulum_v1` |
| target | `frequency` |
| model pairing | `color` |
| test intervention | `color` |
| fixed shape | `circle` |
| OOD color evaluation | enabled |
| low-frequency band | `[2.2, 3.0] rad/s` |
| high-frequency band | `[5.2, 6.4] rad/s` |
| small-amplitude band | `[0.10, 0.17] rad` |
| large-amplitude band | `[0.23, 0.30] rad` |
| short real-history start | frame 57 |
| prediction start | frame 65 |
| future | frames 65–128，共 64 帧 |
| base seed | `3407` |
| training base seeds | `1024` |
| evaluation base seeds | `64` |
| resolution | `128 × 128` |
| fps | `20` |
| total frames | `129` |
| pendulum length | `0.34` |
| bob radius | `7 px` |
| red RGB | `[235, 48, 48]` |
| blue RGB | `[48, 96, 235]` |
| gray RGB | `[150, 154, 164]` |
| green RGB | `[48, 190, 104]` |
| diffusion steps | `20` |
| prediction seed offset | `23000000` |

频率单位始终使用角频率 `rad/s`；图中可增加
\(f=\omega/(2\pi)\) Hz 作为辅助刻度，但不得用 Hz 替换原始记录。

### 2.1 固定训练绑定

`frequency_color_circle` 的训练数据只有：

- 红色圆球 + 低频；
- 蓝色圆球 + 高频。

因此模型训练中学到的颜色相关关系为：

\[
\text{red}\rightarrow[2.2,3.0],
\qquad
\text{blue}\rightarrow[5.2,6.4].
\]

训练颜色只定义频率 **band**，不定义一个精确的唯一频率。
所以对紫色等中间颜色，主分析不能预先假定
“颜色对应一个线性插值得到的精确 shortcut 频率”；
应直接测量输出频率以及它到低、高频 family 的距离。

### 2.2 固定历史构造

- short：帧 0–56 为统一背景，帧 57–64 为真实历史，共 8 个真实条件帧；
- long：帧 0–64 全部为真实历史，共 65 个真实条件帧；
- 两者都生成同样的 64 个未来帧。

20 fps 下，short 的真实历史窗口为 0.40 s，
long 的真实历史窗口为 3.25 s。
以低、高频中心 \(2.6\) 和 \(5.8\) rad/s 计算，
short 中仅分别显示约 0.17 和 0.37 个周期，
而 long 中约显示 1.35 和 3.00 个周期。

## 3. 全部测试共同遵守的配对原则

1. 同一个 `physical_state_id` 下，不同输入颜色必须共享相同真实频率、振幅、相位和真实未来。
2. 同一个颜色扫描内，除球体颜色外，所有渲染像素和物理参数保持一致。
3. short/long 使用相同 `physical_state_id`、输入颜色和基础生成 seed，但使用各自正确的 checkpoint 和历史构造。
4. 用相同 generation seed 比较不同颜色，以减少扩散随机性对颜色效应的干扰。
5. 每条输出都保留 valid 与 invalid 状态；不得只汇总成功生成的视频。
6. 每个聚合结果同时报告样本数、有效率、分布和逐样本点，不只报告均值。
7. 颜色检测不得依赖“预期输出颜色”，否则颜色发生变化时会产生选择偏差。

## 4. 测试子实验总览

基础训练模型保持不变，测试阶段分为四个数据块。

| ID | 子实验 | 测试频率 | 测试颜色 | 主要目的 |
|---|---|---|---|---|
| S0 | configured baseline | 配置中随机采样的低/高频 | red/blue swap、gray、green | 与现有 benchmark 对齐 |
| S1 | ID continuous color sweep | `2.6`, `5.8` | red→blue 11 点 | 主结果：颜色剂量响应 |
| S2 | midpoint competition | `4.2` | red→blue 11 点 | OOD 竞争与中间频率机制 |
| S3 | in-band robustness | 每个训练频段 3 点 | red→blue 5 点 | 检查中心频率结论能否推广 |
| S4 | stochastic repeats | S1/S2 的关键条件 | red、purple、blue | 区分稳定 compromise 与跨 seed 混合 |

S0–S2 是正式结果所需；S3–S4 是稳健性和机制确认。

## 5. S0：保留配置定义的 baseline

### 5.1 测试输入

完全使用基础配置生成的 64 个 evaluation base seeds。
每个 base seed 包含一条随机低频轨迹和一条匹配的随机高频轨迹：

- 低频从 `[2.2, 3.0]` 采样；
- 高频从 `[5.2, 6.4]` 采样；
- 两者共享振幅和初始相位；
- 每条物理轨迹渲染 configured ID color swap、gray 和 green。

按当前生成逻辑，每个 history 的样本数为：

\[
64\ \text{base seeds}
\times 2\ \text{physical frequencies}
\times 3\ \text{appearance variants}
=384.
\]

short 和 long 合计 768 条预测。

### 5.2 回答的问题

1. 在配置原生的 red/blue swap 上，short 和 long 分别有多少 physics、shortcut、compromise 和 invalid？
2. gray/green 没有训练定义的频率绑定时，输出更接近真实物理频率、低频 family 还是高频 family？
3. S1 的精确中心频率结论是否与配置原生的随机频率结果一致？

### 5.3 注意

S0 中的 gray/green 是单独的 OOD color control，
不放在 red→blue 插值轴上，也不人为赋予 color-implied frequency。

## 6. S1：ID 频率上的连续颜色扫描

这是论文或主报告中的核心子实验。

### 6.1 测试频率

固定训练两个频段的中心：

\[
\omega_L=2.6\ \mathrm{rad/s},
\qquad
\omega_H=5.8\ \mathrm{rad/s}.
\]

选择中心频率的理由：

- 两者都位于训练支持内部；
- 不接近频段边界；
- 分别代表训练低、高频条件的均值；
- 可以形成互为镜像的颜色冲突。

对低频 `2.6`：

- red 端与物理频率 aligned；
- blue 端与物理频率 conflict。

对高频 `5.8`：

- red 端与物理频率 conflict；
- blue 端与物理频率 aligned。

不能只使用一个真实频率，否则颜色效应会与冲突方向混淆。

### 6.2 测试颜色

定义：

\[
\alpha_{\mathrm{in}}\in
\{0.0,0.1,\ldots,0.9,1.0\},
\]

\[
C(\alpha)
=\operatorname{round}
\left((1-\alpha)C_R+\alpha C_B\right).
\]

具体 RGB 为：

| \(\alpha\) | RGB |
|---:|---|
| 0.0 | `[235, 48, 48]` |
| 0.1 | `[216, 53, 67]` |
| 0.2 | `[198, 58, 85]` |
| 0.3 | `[179, 62, 104]` |
| 0.4 | `[160, 67, 123]` |
| 0.5 | `[142, 72, 142]` |
| 0.6 | `[123, 77, 160]` |
| 0.7 | `[104, 82, 179]` |
| 0.8 | `[85, 86, 198]` |
| 0.9 | `[67, 91, 216]` |
| 1.0 | `[48, 96, 235]` |

RGB 值作为 uint8 输入保存，同时保留未取整的插值位置
`color_alpha_target`。0 和 1 是训练见过的颜色端点，
其余颜色均标记为 `interpolated_color_ood`。

### 6.3 物理状态

使用全部 64 个 configured evaluation base states。
对每个 state：

- 振幅仍从固定的 small-amplitude 范围采样；
- 相位仍由配置 seed 的确定性随机过程产生；
- 频率被 evaluation-only sweep 固定为 `2.6` 或 `5.8`；
- 同一频率下的 11 种颜色共享完全相同的物理状态；
- short/long 共享原始 129 帧视频，只改变历史构造。

额外审计 64 个状态的：

- amplitude 分布；
- prediction-boundary phase 分布；
- `theta_star` 和 `angular_velocity_star` 分布；
- short 真实窗口中的角位移范围。

如果相位覆盖严重不均，优先增加 evaluation states，
不修改基础 seed、振幅范围或频率范围。

### 6.4 样本规模

primary generation 使用配置中的 seed offset 对应的一次确定性生成：

\[
2\ \text{frequencies}
\times 11\ \text{colors}
\times 64\ \text{physical states}
\times 2\ \text{histories}
=2816.
\]

其中 red/blue endpoint slice 为：

\[
2\times2\times64\times2=512
\]

条预测，可直接形成 aligned/conflict 主比较，
无需再次生成单独的端点数据集。

### 6.5 回答的问题

1. 固定真实低频时，输出频率是否随 red→blue 向高频移动？
2. 固定真实高频时，输出频率是否呈镜像变化？
3. 变化是平滑、阈值式还是在不同样本中离散选路？
4. short 的颜色剂量响应是否显著强于 long？
5. 输出颜色是否连续保留输入颜色，还是被量化到红/蓝端点？

## 7. S2：中间物理频率竞争

### 7.1 测试设定

固定：

\[
\omega_M
=\frac{2.6+5.8}{2}
=4.2\ \mathrm{rad/s}.
\]

颜色使用与 S1 相同的 11 点 red→blue 扫描，
物理状态使用相同的 64 个 amplitude/phase states。

样本规模：

\[
1\ \text{frequency}
\times11\ \text{colors}
\times64\ \text{states}
\times2\ \text{histories}
=1408.
\]

### 7.2 实验地位

`4.2 rad/s` 位于训练频率间隙，因此 S2 同时包含
frequency OOD 和 interpolated-color OOD。
它只用于解释模型的竞争机制，不能替代 S1 的 ID 结论。

### 7.3 回答的问题

1. red 是否把输出拉向低频 family，blue 是否把输出拉向高频 family？
2. purple 附近是否最接近真实的 `4.2 rad/s`？
3. short 的物理保持误差是否呈“两端高、中间低”的 U 型？
4. 中间输出来自单条视频内的稳定中频，还是跨 generation seeds 的低/高路线混合？
5. 输出颜色端点化与输出频率端点化是否同步？

## 8. S3：训练频段内部的稳健性扫描

S3 检查 S1 是否只是两个中心频率的特例。

### 8.1 频率与颜色

测试频率：

\[
\Omega_{\mathrm{robust}}
=\{2.4,2.6,2.8,5.5,5.8,6.1\}.
\]

测试颜色使用五点 pilot：

\[
\alpha_{\mathrm{in}}
\in\{0,0.25,0.5,0.75,1\}.
\]

所有频率都位于配置定义的训练频段内部，
且避开频段边缘。

### 8.2 分阶段样本规模

先用 16 个按 boundary phase 分层选择的 physical states：

\[
6\times5\times16\times2=960.
\]

如果 S3 与 S1 得到相反趋势，或不同频率差异明显，
再扩展到全部 64 states：

\[
6\times5\times64\times2=3840.
\]

### 8.3 回答的问题

1. 颜色效应在各自频段内部是否稳定？
2. 接近频段内侧或外侧时，shortcut 强度是否不同？
3. short–long 差异是否依赖精确真实频率？

## 9. S4：关键条件的生成随机性复测

S4 不改变模型或输入，只增加相同输入下的 diffusion repeats。

### 9.1 条件

\[
\omega_{\mathrm{true}}\in\{2.6,4.2,5.8\},
\]

\[
\alpha_{\mathrm{in}}\in\{0,0.5,1\}.
\]

从 64 个 states 中按 boundary phase 和 amplitude
分层选择 16 个 states。每个输入共生成 5 个 seeds：

- repeat 0 使用配置定义的 `seed_offset=23000000`；
- repeat 1–4 使用独立、确定性的 supplementary seed namespace；
- 每条记录保存最终使用的完整 generation seed；
- 不覆盖或修改基础 `seed_offset`。

总样本数：

\[
3\times3\times16\times2\times5=1440.
\]

其中 repeat 0 已包含在 S1/S2 中，因此新增预测为 1152 条。

### 9.2 回答的问题

1. 紫色或冲突输入是否对 generation seed 敏感？
2. “平均中频”是每个 seed 都产生中频，还是不同 seed 分成低、高两群？
3. 输出颜色选择和输出频率选择是否在 seed 层面共同切换？

## 10. 建议的执行规模

### 10.1 快速 pipeline pilot

在正式生成前，使用：

- 8 个 physical states；
- `omega={2.6,4.2,5.8}`；
- `alpha={0,0.25,0.5,0.75,1}`；
- short/long；
- 每个输入一个 generation seed。

共：

\[
8\times3\times5\times2=240
\]

条预测。pilot 只验证数据、检测、指标和图表 pipeline，
不用于最终统计结论。

### 10.2 正式最小集合

- S0 configured baseline：768；
- S1 ID continuous sweep：2816；
- S2 midpoint competition：1408；
- S4 新增 stochastic repeats：1152。

S3 在主结果完成后运行。

## 11. 每个测试样本必须记录的输入元数据

一条主结果记录对应一条测试生成视频。
字段名必须显式区分训练、测试和输出，不使用含义模糊的 `color`
或 `frequency`。

### 11.1 模型身份

```text
experiment_id
dataset_id
model_name
training_manifest_id
train_history
checkpoint_path
checkpoint_step
train_seed
```

其中 `train_history` 只能是 `short` 或 `long`。

### 11.2 测试干预

```text
sweep_id
subexperiment_id
sample_id
physical_state_id
pair_id
eval_history
test_omega_true
test_frequency_support
test_color_alpha_target
test_color_rgb_target
test_color_support
generation_repeat
generation_seed
```

`test_frequency_support` 取：

- `low_band_id`
- `high_band_id`
- `gap_ood`

`test_color_support` 取：

- `endpoint_id`
- `interpolated_color_ood`
- `gray_ood`
- `green_ood`

正常生成必须满足：

```text
train_history == eval_history
```

### 11.3 物理状态

```text
amplitude_true
phase_initial
phase_boundary
theta_star
angular_velocity_star
pendulum_length
fps
prediction_start
```

另外从实际解码后的测试输入帧重新提取：

```text
test_color_rgb_measured
test_color_alpha_measured
input_color_decode_error
input_detection_rate
short_visible_theta_range
short_visible_motion_px
```

后续颜色响应优先使用 `test_color_alpha_measured`，
目标 RGB 用于数据生成审计。

## 12. 输出轨迹提取

### 12.1 与颜色无关的摆球检测

连续颜色实验不能用“接近输入颜色”作为寻找摆球的必要条件。
否则模型把紫色球生成成蓝色时，评估器可能把真实颜色变化误判为摆球丢失。

主检测器应组合：

- 摆球的圆形边缘；
- 固定 pivot 和 pendulum length 给出的几何可达区域；
- 摆绳末端；
- 相邻帧位置连续性；
- 现有颜色检测结果只作为辅助候选。

每帧输出：

```text
x_px
y_px
detected
ball_area_px
candidate_count
max_local_color_contrast
```

球心转换为从竖直向下方向测量的角度：

\[
\theta_t
=\operatorname{atan2}
\left(
\frac{x_t}{W-1}-p_x,\,
\frac{y_t}{H-1}-p_y
\right).
\]

完整保存：

```text
theta_pred[64]
detection_mask[64]
```

### 12.2 轨迹有效性

保留现有 Pendulum 评估口径：

- detection rate 至少 0.90；
- 最大连续缺失帧不超过 3；
- 最大相邻跳变不超过 20 px；
- median pendulum-length error 不超过 0.05；
- condition–future boundary jump 不超过 20 px；
- 摆球面积位于现有合理区间；
- 自由振荡拟合有效；
- 当前统一的 fit RMSE gate 保持 `0.08 rad`，除非以后以独立校准实验整体修订。

每个失败样本保存唯一的首要 `failure_reason`，并保留所有原始诊断量。

## 13. 输出频率和运动指标

### 13.1 主频率估计

对 64 个未来帧拟合：

\[
\theta(t)
=a\cos(\omega t)+b\sin(\omega t)+c.
\]

搜索范围继续使用配置频段派生的现有范围：

\[
\omega\in[2.2-1.0,\ 6.4+1.0]
=[1.2,7.4]\ \mathrm{rad/s}.
\]

保存：

```text
omega_hat
amplitude_hat
phase_hat
center_hat
free_fit_rmse
free_fit_curve[64]
```

其中：

\[
\hat A=\sqrt{a^2+b^2}.
\]

### 13.2 物理误差

从测试输入元数据中的精确物理参数生成真实 future：

```text
theta_physical_reference[64]
```

计算：

\[
d_{\mathrm{exact\ physics}}
=
\sqrt{
\frac{1}{N}
\sum_t
(\theta_t^{\mathrm{pred}}
-\theta_t^{\mathrm{physical}})^2
}.
\]

同时保存：

```text
omega_absolute_error
amplitude_absolute_error
boundary_angle_error
boundary_velocity_error
d_exact_physics
```

其中：

\[
\text{omega_absolute_error}
=|\hat\omega-\omega_{\mathrm{true}}|.
\]

### 13.3 到低、高频 family 的距离

分别把自由振荡频率限制在两个训练频段内，
但允许系数 `a,b,c` 对输出轨迹重新拟合：

\[
d_L
=
\min_{\omega\in[2.2,3.0],a,b,c}
\operatorname{RMSE}
\left(
\theta_{\mathrm{pred}},
a\cos(\omega t)+b\sin(\omega t)+c
\right),
\]

\[
d_H
=
\min_{\omega\in[5.2,6.4],a,b,c}
\operatorname{RMSE}
\left(
\theta_{\mathrm{pred}},
a\cos(\omega t)+b\sin(\omega t)+c
\right).
\]

保存：

```text
d_low_family
d_high_family
omega_hat_low_family
omega_hat_high_family
```

定义低—高 route preference：

\[
S_{\mathrm{LH}}
=
\frac{d_L-d_H}{d_L+d_H+\epsilon}.
\]

- \(S_{\mathrm{LH}}\approx-1\)：更接近低频 family；
- \(S_{\mathrm{LH}}\approx+1\)：更接近高频 family；
- \(S_{\mathrm{LH}}\approx0\)：两者距离接近。

这个分数可用于所有颜色，包括 purple、gray 和 green，
但它只表示输出运动更接近哪个训练 family，
不自动证明颜色 causally 决定了该路线。

### 13.4 时间稳定性

分别拟合未来前 32 帧和后 32 帧：

```text
omega_hat_front
omega_hat_back
front_fit_rmse
back_fit_rmse
front_back_omega_gap
```

\[
\text{front\_back\_omega\_gap}
=|\hat\omega_{\mathrm{front}}-\hat\omega_{\mathrm{back}}|.
\]

另用固定宽度滑窗计算：

```text
rolling_omega[t]
rolling_fit_rmse[t]
```

滑窗宽度在实现前固定，并写入结果 context；
同一批 short/long 结果必须使用完全相同的宽度。

## 14. 频率路线标签

### 14.1 基础频率类

仅对 valid 且 `free_fit_rmse <= 0.08` 的样本分类：

| 条件 | `omega_class` |
|---|---|
| `2.2 <= omega_hat <= 3.0` | `low_band` |
| `3.0 < omega_hat < 5.2` | `between_bands` |
| `5.2 <= omega_hat <= 6.4` | `high_band` |
| 其余有限值 | `outside_bands` |
| 拟合无效 | `invalid` |

### 14.2 red/blue 端点冲突

只有 `alpha=0` 或 `alpha=1` 才有训练中明确定义的
color-implied band。

对于真实低频 + blue：

- `low_band` → `physics_frequency`
- `high_band` → `shortcut_frequency`
- `between_bands` → `compromise_frequency`

对于真实高频 + red：

- `high_band` → `physics_frequency`
- `low_band` → `shortcut_frequency`
- `between_bands` → `compromise_frequency`

aligned 条件只报告正确 band 和有效性，
不声称模型究竟使用了颜色还是物理，因为两种线索给出相同答案。

### 14.3 中间输入颜色

对于 `0 < alpha < 1`：

- 不人为指定一个精确的 color-implied frequency；
- 不把线性 RGB 插值直接转换成线性频率标签；
- 主结果报告 `omega_hat`、`omega_class`、`d_low_family`、
  `d_high_family` 和 `S_LH`；
- `between_bands` 且轨迹稳定时可标记为
  `coherent_intermediate_frequency`。

### 14.4 coherent intermediate 的判定

必须同时满足：

1. `omega_class == between_bands`；
2. 全段 `free_fit_rmse <= 0.08`；
3. 输出轨迹通过全部 validity gates；
4. `front_back_omega_gap` 低于预先校准的阈值；
5. 前后半段拟合都有效；
6. 无明显轨迹跳变、消失或摆长破坏。

`front_back_omega_gap` 阈值从 aligned、valid 的控制结果中预注册，
例如取其 95% 分位数；不能根据 purple 结果事后选择。

## 15. 输出颜色提取

### 15.1 球体颜色区域

在已经由几何方法确定的球心周围建立：

- `core mask`：球半径的约 60%，作为主颜色统计区域；
- `full-ball mask`：完整球体，用于空间混色分析；
- `boundary ring`：只用于压缩、抗锯齿和 mask 质量诊断。

主颜色结果使用 core mask，避免摆绳、背景和边缘像素污染。

### 15.2 像素、帧和视频三级统计

每帧保存：

```text
rgb_mean_frame
rgb_trimmed_mean_frame
rgb_median_frame
rgb_q05_frame
rgb_q25_frame
rgb_q75_frame
rgb_q95_frame
rgb_min_frame
rgb_max_frame
oklab_median_frame
saturation_median_frame
lightness_median_frame
```

min/max 只用于诊断；正文主统计使用 median、IQR 和 5–95%。

视频级保存：

```text
rgb_median_video
rgb_q05_video
rgb_q95_video
oklab_median_video
saturation_median_video
lightness_median_video
```

### 15.3 红蓝轴位置

在 RGB 空间定义训练端点：

\[
C_R=[235,48,48],
\qquad
C_B=[48,96,235].
\]

对一个输出颜色向量 \(C\)，定义：

\[
\alpha_{\mathrm{out}}
=
\frac{(C-C_R)^\top(C_B-C_R)}
{\|C_B-C_R\|^2}.
\]

不对该值提前截断：

- 0 表示训练红色端点；
- 1 表示训练蓝色端点；
- 0.5 表示红蓝线的中点；
- 小于 0 或大于 1 表示越过训练端点。

计算输出颜色到红蓝参考线的正交距离：

\[
d_\perp
=
\left\|
C-
\left[C_R+\alpha_{\mathrm{out}}(C_B-C_R)\right]
\right\|.
\]

保存：

```text
color_alpha_out_frame[64]
color_alpha_out_median
color_alpha_out_q05
color_alpha_out_q95
color_axis_orthogonal_distance
```

颜色保持误差：

\[
E_{\mathrm{color}}
=
|\alpha_{\mathrm{out}}
-\alpha_{\mathrm{in,measured}}|.
\]

### 15.4 空间混色和时间切换

视频平均紫色可能来自三种不同机制：

1. 每一帧都是均匀紫色；
2. 同一帧内部同时有红、蓝像素；
3. 不同帧在红色和蓝色之间切换。

因此还要计算：

```text
red_pixel_fraction_frame
blue_pixel_fraction_frame
intermediate_pixel_fraction_frame
pixel_color_bimodality_frame
color_temporal_iqr
max_frame_color_jump
color_switch_count
front_back_color_gap
```

对空间双峰和时间切换分别给出独立标签，
不能仅凭视频平均 RGB 判断“输出为紫色”。

## 16. 输出颜色—输出频率联合指标

### 16.1 标准化频率位置

定义：

\[
z_\omega
=
\frac{\hat\omega-2.6}{5.8-2.6}.
\]

它只用于可视化：

- 0 对应低频中心；
- 1 对应高频中心；
- 不对超出 `[0,1]` 的值截断。

正式 route classification 仍使用完整频段边界。

### 16.2 配对颜色和频率位移

对同一个 `physical_state_id`、history 和 generation seed：

\[
\Delta\hat\omega
=
\hat\omega(\alpha=1)
-\hat\omega(\alpha=0),
\]

\[
\Delta\alpha_{\mathrm{out}}
=
\alpha_{\mathrm{out}}(\alpha=1)
-\alpha_{\mathrm{out}}(\alpha=0).
\]

保存：

```text
paired_omega_shift_blue_minus_red
paired_output_color_shift_blue_minus_red
```

分析两者是否共同增大。

### 16.3 联合路线表

输出颜色离散为：

- `red_endpoint`
- `intermediate_color`
- `blue_endpoint`
- `off_axis_color`
- `color_invalid`

输出频率离散为：

- `low_band`
- `between_bands`
- `high_band`
- `outside_bands`
- `frequency_invalid`

计算 5×5 联合计数和比例。
由此区分：

- red + low：完整红低频路线；
- blue + high：完整蓝高频路线；
- intermediate + between：外观和运动都保持中间；
- blue + physical-low 或 red + physical-high：输出外观与运动决策解耦；
- color/frequency invalid：生成质量失败。

联合关系只能表述为“输出颜色—频率耦合”。
输入颜色是实验干预，因此输入颜色对输出频率的 paired shift
具有因果解释；输出颜色与输出频率之间本身不能仅凭相关性声称因果方向。

## 17. short–long 配对结果

对相同 `physical_state_id`、测试颜色和 generation repeat 计算：

```text
long_minus_short_omega_hat
long_minus_short_omega_absolute_error
long_minus_short_d_exact_physics
long_minus_short_S_LH
long_minus_short_color_alpha_out
long_minus_short_color_retention_error
long_minus_short_front_back_omega_gap
short_valid
long_valid
```

主汇总同时报告：

- short physics/shortcut/compromise/invalid rate；
- long physics/shortcut/compromise/invalid rate；
- paired route transition，例如
  `shortcut_frequency -> physics_frequency`；
- paired frequency error 的变化；
- paired output-color endpoint attraction 的变化。

由于 short 和 long 是独立训练模型，
结果文字使用“short pipeline 与 long pipeline 的差异”，
不使用“同一模型增加历史帧后”的表述。

## 18. 主报告图表

主报告采用一套共享的视觉编码：

- short 始终使用同一种线型或面板位置；
- long 始终使用另一种线型或面板位置；
- 低频结果使用暖色边框，高频结果使用冷色边框；
- 输入颜色点使用其真实 RGB；
- 所有频率图共享低频、高频背景区间；
- invalid 样本不出现在频率数值分布中，但必须在相邻面板显示其比例。

## 19. Figure 1：实验设计和代表输入

### 19.1 内容

第一部分画训练绑定：

```text
red circle  -> low-frequency band
blue circle -> high-frequency band
```

第二部分画测试干预：

```text
fixed physical trajectory
        +
red -> purple -> blue input color sweep
        +
short / long matched pipelines
```

第三部分展示 short/long 条件帧：

- short：背景帧 0–56 + 真实帧 57–64；
- long：真实帧 0–64；
- future 都是帧 65–128。

### 19.2 目的

使读者在看结果前明确：

- 训练中颜色与频率完全绑定；
- 测试时真实频率和颜色独立控制；
- short/long 是两个匹配 pipeline；
- 4.2 rad/s 和中间颜色属于补充 OOD 干预。

## 20. Figure 2：red/blue endpoint 的 aligned–conflict 结果

数据来自 S1 的 `alpha=0` 和 `alpha=1` slice。

### 20.1 Panel A：输出频率分布

- x 轴：`low/red`, `low/blue`, `high/red`, `high/blue`；
- y 轴：`omega_hat (rad/s)`；
- short/long 分成两个并排面板；
- 背景阴影标出 `[2.2,3.0]`、`(3.0,5.2)` 和 `[5.2,6.4]`；
- 每个条件显示逐样本点、median、IQR 和 5–95%。

### 20.2 Panel B：route composition

每个条件绘制 100% 堆叠柱：

- physics frequency；
- shortcut frequency；
- compromise frequency；
- outside/off-family；
- invalid。

柱内显示百分比，柱上显示原始计数 `n/N`。

### 20.3 Panel C：short–long route transitions

只使用 short/long 都 valid 的 matched samples，
画 Sankey、alluvial 或转移矩阵：

```text
short route -> long route
```

同时在旁边单独报告：

- short-only valid；
- long-only valid；
- both invalid。

### 20.4 得到的信息

Figure 2 是是否存在 shortcut 的最直接证据：

- conflict 条件下是否进入颜色绑定的错误频段；
- short 的 shortcut rate 是否高于 long；
- long 是否把 short 的 shortcut 或 compromise 转回 physics；
- 颜色冲突是否同时提高 invalid rate。

## 21. Figure 3：ID 频率的颜色剂量响应

数据来自完整 S1。

### 21.1 图布局

使用 2×2 面板：

| | true low `2.6` | true high `5.8` |
|---|---|---|
| short | Panel A | Panel B |
| long | Panel C | Panel D |

每个面板：

- x 轴：`test_color_alpha_measured`；
- y 轴：`omega_hat`；
- 显示每个 physical state 的浅色轨迹线；
- 叠加 condition median 和 trajectory-level bootstrap CI；
- 显示训练低、高频 band；
- invalid rate 作为下方独立窄条，不从图中消失。

### 21.2 不使用的做法

- 不把 purple 预先映射为一个“应有 shortcut 频率”；
- 不只画各颜色的平均频率；
- 不把低频和高频输入混成一条曲线；
- 不在不同 history 中使用不同 y 轴范围。

### 21.3 辅助 Panel

画：

\[
S_{\mathrm{LH}}
\quad\text{vs.}\quad
\alpha_{\mathrm{in}}.
\]

它展示输出从低频 family 向高频 family 的连续偏好变化。

### 21.4 得到的信息

- 颜色响应是否近似单调；
- 转换发生在 purple 附近还是偏向某一端；
- 是平滑插值还是突然发生 band switch；
- short/long 的颜色敏感区间和斜率是否不同；
- 输入真实低频和真实高频是否给出互为镜像的结果。

## 22. Figure 4：输入—输出颜色的三维 RGB 图

这是输出颜色的主图，不由二维均值曲线替代。

### 22.1 坐标和参考线

三维坐标轴：

\[
x=R,\qquad y=G,\qquad z=B,
\]

范围统一固定为 `[0,255]`。

参考线为训练红、蓝端点之间的插值线：

\[
C_{\mathrm{ref}}(\alpha)
=(1-\alpha)C_R+\alpha C_B.
\]

这不是 RGB 立方体中 `R=G=B` 的黑白主对角线。

### 22.2 每个条件的绘制元素

对每个 `alpha`：

1. 解码后实测输入颜色：空心圆；
2. 输出视频颜色 median：实心圆；
3. 从输入点指向输出点的箭头；
4. 输出 RGB 的 5–95% 范围或 covariance ellipsoid；
5. 输出点本身使用实际输出 RGB 着色；
6. 颜色或 marker 区分真实低频与真实高频。

short 和 long 使用两个并排的三维面板，
共享视角、范围和参考线。

### 22.3 配套二维定量图

三维图旁边必须显示：

\[
\alpha_{\mathrm{out}}
\quad\text{vs.}\quad
\alpha_{\mathrm{in,measured}},
\]

并画 `y=x` 参考线。

另画：

\[
d_\perp
\quad\text{vs.}\quad
\alpha_{\mathrm{in}}.
\]

三维图用于展示实际 RGB 偏移方向，
`alpha_out` 用于量化沿红蓝轴的端点化，
`d_perp` 用于识别灰化、亮度变化或其他 off-axis 颜色。

### 22.4 选定样本的像素分布 inset

对 `alpha={0,0.5,1}` 各选相同 `physical_state_id`，
显示球体 core pixels 的颜色分布。
重点区分：

- 均匀 purple；
- 同一帧内 red/blue 双峰；
- 随时间 red/blue 切换；
- 低饱和度或 off-axis 输出。

## 23. Figure 5：输出颜色—输出频率联合图

### 23.1 主散点/密度图

- x 轴：`color_alpha_out_median`；
- y 轴：`omega_hat` 或 `z_omega`；
- 点的填充色：输入颜色；
- marker：真实低频/高频；
- short/long 分面；
- invalid 样本在图边缘用计数条展示。

### 23.2 联合路线矩阵

显示输出颜色类别 × 输出频率类别的 5×5 矩阵。
每格同时显示：

- 样本数；
- 占该测试条件的百分比；
- short–long 的差值。

### 23.3 配对位移图

对同一 physical state：

- x 轴：`paired_output_color_shift_blue_minus_red`；
- y 轴：`paired_omega_shift_blue_minus_red`；
- short/long 使用不同 marker；
- 显示 trajectory-level bootstrap 回归或相关区间。

### 23.4 得到的信息

- 输出颜色端点化是否伴随频率端点化；
- 模型是否生成完整的 red–low 或 blue–high 联合路线；
- 是否存在“输出蓝色但运动保持真实低频”等解耦案例；
- short 和 long 的外观—运动耦合是否不同。

## 24. Figure 6：4.2 rad/s OOD competition

数据来自 S2。

### 24.1 Panel A：输出频率

- x 轴：输入 color alpha；
- y 轴：`omega_hat`；
- 参考线：2.6、4.2、5.8；
- short/long 分面；
- 显示 raw points、median、IQR 和 5–95%。

### 24.2 Panel B：物理保持误差

绘制：

\[
|\hat\omega-4.2|
\quad\text{和}\quad
d_{\mathrm{exact\ physics}}
\]

随颜色变化的曲线。

如果两端颜色 shortcut 较强、purple 处物理证据占优，
曲线可能出现“两端高、中间低”的形状；
这是待检验假设，不预先强制拟合 U 型。

### 24.3 Panel C：purple 分布

对 `alpha=0.5`：

- 画 `omega_hat` violin/KDE 和原始点；
- short/long 使用相同频率范围；
- 根据 S4 的 repeats 判断单峰中频还是低/高双峰；
- 同时显示 coherent intermediate rate、front–back switch rate 和 invalid rate。

### 24.4 Panel D：时间轨迹例子

选择预注册规则得到的代表样本：

- median coherent intermediate；
- red/low endpoint route；
- blue/high endpoint route；
- temporal switching；
- invalid。

每个样本显示：

```text
theta_pred(t)
theta_physical_reference(t)
free_fit_curve(t)
rolling_omega(t)
color_alpha_out(t)
```

不允许只凭视觉手选“最漂亮”的生成视频。

## 25. Figure 7：稳健性和质量控制

### 25.1 S3 frequency × color heatmap

- x 轴：输入 color alpha；
- y 轴：真实输入频率；
- fill：median `omega_hat - omega_true`；
- short/long 两张图共享发散色标；
- 训练 band 和频率间隙清楚标注。

### 25.2 validity heatmap

同样的轴，fill 改为：

- validity rate；
- free-fit failure rate；
- detection failure rate；
- boundary-jump failure rate。

### 25.3 相位和振幅分层

补充图显示颜色效应对以下变量的依赖：

- prediction-boundary phase；
- short-visible motion range；
- amplitude；
- generation repeat。

如果 shortcut 主要出现在 short 可见运动接近静止的状态，
必须在正文中说明，而不能只报告总体平均。

## 26. 最小正文图版与详细 dashboard

### 26.1 正文最小图版

正文可以压缩为六个 panel：

1. 训练绑定和 short/long 设计；
2. endpoint aligned/conflict 的 route composition；
3. ID frequency 的 color-dose response；
4. short/long 的 3D RGB 输出颜色；
5. output color–frequency joint distribution；
6. 4.2 rad/s 的 OOD competition 与 representative traces。

### 26.2 dashboard 下钻

交互式结果页保留：

- subexperiment；
- history；
- checkpoint；
- true frequency；
- input color；
- physical state；
- generation repeat；
- validity；
- route label。

点击任意聚合点后显示：

- 原始输入和输出视频；
- sample metadata；
- `theta(t)`；
- free fit 和 physical reference；
- rolling frequency；
- framewise RGB 和 `alpha_out(t)`；
- ball-pixel color distribution；
- 全部 failure diagnostics。

## 27. 统计汇总规则

### 27.1 分析单位

主要独立分析单位是 `physical_state_id`，
不是帧、像素或单个 diffusion sample。

- 帧和像素用于构造视频级特征；
- 多个 generation repeats 嵌套在同一个 physical state 中；
- short/long、不同颜色均尽可能进行 paired comparison。

### 27.2 汇总量

每个条件至少报告：

```text
N_total
N_valid
validity_rate
median
IQR
q05
q95
trajectory-level bootstrap CI
```

频率、颜色和距离指标保留完整逐样本分布。
均值和标准差可以附加，但不替代 median 和分位数。

### 27.3 bootstrap

bootstrap 以 `physical_state_id` 为 cluster 重采样：

1. 抽取 physical states；
2. 同一 state 的全部颜色、history 和 repeats 一起进入；
3. 重新计算 paired effect；
4. 报告预注册的置信区间。

不能把 64 帧或球体像素作为独立样本扩大有效样本数。

### 27.4 invalid 处理

- validity rate 使用全部生成样本作分母；
- 连续物理和颜色指标只在相应测量有效时汇总；
- 每张只显示 valid 数值的图旁边都必须显示 invalid rate；
- 不把 NaN 或 invalid 静默转换为 0；
- 分别报告 missing prediction、detection failure、fit failure 和 trajectory-invalid。

## 28. 结果解释矩阵

| 输出颜色 | 输出频率 | 可支持的解释 |
|---|---|---|
| 保持输入连续色 | 保持真实频率 | 外观和运动都连续保持 |
| 蓝端点 | 高频 | 选择完整 blue–high 训练路线 |
| 红端点 | 低频 | 选择完整 red–low 训练路线 |
| purple | coherent intermediate | 外观与运动都出现稳定中间解 |
| purple 平均但像素双峰 | 低/高混合 | 空间混色，不能称为均匀中间色 |
| 随时间 red↔blue | 随时间 low↔high | 单视频内部路线切换 |
| 蓝色 | 真实低频 | 外观端点化但运动未跟随颜色 |
| 红色 | 真实高频 | 外观端点化但运动未跟随颜色 |
| 任意 | 高 RMSE/invalid | 生成失败，不能作为竞争证据 |

报告结论时优先使用：

```text
输入颜色干预导致输出频率发生 paired shift
```

以及：

```text
输出颜色选择与输出频率路线发生耦合或解耦
```

避免仅凭输出相关性写成：

```text
输出颜色导致了输出频率
```

## 29. evaluation-only 实现补充

本方案不要求修改基础 `config.yaml`。
建议新增独立的 sweep specification，例如：

```text
configs/pendulum/eval/frequency_color_sweep.yaml
```

该文件只引用基础配置并声明：

```text
subexperiment_id
test_omega_values
test_color_alpha_values
physical_state_indices
generation_repeats
output_root
```

所有基础范围、渲染参数、history 定义、diffusion steps 和 seed offset
继续从原始 `config.yaml` 读取，不在 sweep 文件中复制或覆盖。

### 29.1 数据生成器需要增加的能力

1. 接受 evaluation-only `omega_override`；
2. 接受 `color_alpha` 和任意 RGB；
3. 对同一 `physical_state_id` 复用 amplitude 和 phase；
4. 对所有颜色验证 appearance ROI 外的像素 hash 一致；
5. 在 metadata 中记录 target RGB 和解码实测 RGB；
6. 为 S1–S4 使用独立 `test_manifest_id`；
7. 不改变现有 configured S0 数据。

### 29.2 评估器需要增加的能力

1. 颜色无关的球心检测；
2. 任意 RGB 输入的颜色测量；
3. `between_bands` 和 `coherent_intermediate_frequency`；
4. low/high family restricted fits；
5. front/back 和 rolling frequency；
6. core/full/ring 三种颜色 mask；
7. framewise RGB、`alpha_out` 和颜色切换指标；
8. 输出逐帧 trace 到 NPZ，而不只保存 summary scalars；
9. short/long matched comparison；
10. S4 generation-repeat grouping。

### 29.3 绘图程序需要读取的最小产物

```text
conditions.csv
per_sample.jsonl
per_frame.npz
summary.json
failure_counts.json
```

绘图程序必须从这些产物计算图值，
不得手工转录表格或在代码中写死实验结果。

## 30. 推荐执行顺序

### Stage 0：预检和快照

1. 记录基础 `config.yaml` 的 hash；
2. 记录 short/long checkpoint 路径和 hash；
3. 记录当前代码 commit；
4. 检查两个训练配置的 history 与 checkpoint 匹配；
5. 建立独立 sweep 输出目录；
6. 不覆盖任何已有 dataset、prediction 或 metric 目录。

### Stage 1：实现并审计输入 sweep

先只生成输入视频，不运行模型：

1. 生成 pilot 的 240 个测试输入；
2. 核对每个视频 129 帧、20 fps、128×128；
3. 核对指定频率、振幅和相位；
4. 核对 11 点 RGB；
5. 核对不同颜色只改变球体 appearance ROI；
6. 核对 short mask 只覆盖帧 0–56；
7. 核对 short/long 共享真实帧 57–128；
8. 核对 sample/pair/state ID 唯一且完整。

只有输入审计通过后才进入生成。

### Stage 2：pilot 预测和评估

对 240 个 pilot 输入：

1. 运行 short 预测；
2. 运行 long 预测；
3. 运行轨迹与颜色检测；
4. 检查 per-sample 和 per-frame schema；
5. 检查 3D RGB、frequency-dose 和 trajectory 图能否生成；
6. 人工查看少量检测成功与失败视频，确认失败标签合理；
7. 固定所有 metric thresholds 和 plotting defaults。

pilot 结果不进入正式统计。

### Stage 3：S0 configured baseline

先运行现有 64 base seeds：

1. 验证当前 benchmark 端点冲突和 gray/green 控制；
2. 验证评估扩展没有破坏已有结果；
3. 固定 configured baseline 的 summary。

### Stage 4：S1 主实验

按顺序运行：

1. 低频 `2.6` 的 11 点颜色；
2. 高频 `5.8` 的 11 点颜色；
3. short；
4. long；
5. matched comparison；
6. Figure 2–5。

S1 完成后先冻结主结果产物，再开始 OOD 分析。

### Stage 5：S2 midpoint competition

运行 `4.2 rad/s` 的 11 点颜色，
生成 Figure 6 的初始版本。
此阶段只描述单次 configured generation seed 的分布，
不根据均值判断是否存在 stochastic route mixture。

### Stage 6：S4 stochastic repeats

对 16 个分层 states 的 red/purple/blue 关键条件增加 repeats，
重新完成：

- purple 单峰/双峰判断；
- seed-level route probability；
- color–frequency joint switching；
- representative sample selection。

### Stage 7：S3 robustness

在 S1/S2 主结论稳定后运行 16-state S3。
只有发现明显频率依赖时才扩展到 64 states。

### Stage 8：最终汇总

1. 重新生成全部图；
2. 导出主结果表；
3. 生成 artifact inventory；
4. 验证所有图和表使用同一批 finalized metrics；
5. 再次核对基础 `config.yaml` hash 未改变。

## 31. 建议产物目录

所有新产物与原始训练、数据和现有结果隔离。

```text
data/pendulum/frequency_color_sweeps/
  pilot/
  s1_id_color_sweep/
  s2_midpoint_competition/
  s3_in_band_robustness/
  s4_stochastic_repeats/

runs/pendulum/frequency_color_evaluation/
  configured_baseline/
    short/
    long/
    comparison/
  s1_id_color_sweep/
    short/
    long/
    comparison/
  s2_midpoint_competition/
    short/
    long/
    comparison/
  s3_in_band_robustness/
  s4_stochastic_repeats/
  figures/
  artifact_inventory.json
```

每个 history 结果目录包含：

```text
predictions.json
predictions/
metrics.json
summary.json
per_sample.jsonl
per_frame.npz
conditions.csv
failure_counts.json
context.json
```

`context.json` 至少记录：

```text
base_config_path
base_config_hash
training_config_path
checkpoint_path
checkpoint_hash
code_commit
subexperiment_id
history
generation_seed_rule
metric_thresholds
created_at
```

## 32. 主结果表

### 32.1 condition-level table

每行对应：

```text
subexperiment
history
test_omega_true
test_color_alpha
```

列至少包括：

```text
N_total
N_valid
validity_rate
physics_rate
shortcut_rate
coherent_intermediate_rate
outside_rate
median_omega_hat
q05_omega_hat
q95_omega_hat
median_d_exact_physics
median_S_LH
median_color_alpha_out
q05_color_alpha_out
q95_color_alpha_out
median_color_retention_error
color_endpoint_rate
color_frequency_route_agreement
```

### 32.2 paired-effect table

每行对应一个 `physical_state_id`：

```text
blue_minus_red_omega_shift
blue_minus_red_output_color_shift
long_minus_short_frequency_error
long_minus_short_d_exact_physics
long_minus_short_color_retention_error
short_route
long_route
short_valid
long_valid
```

## 33. 验收清单

### 33.1 配置保护

- [ ] 基础 `config.yaml` hash 与 Stage 0 相同；
- [ ] short/long training configs 未被修改；
- [ ] checkpoint 未被修改；
- [ ] S0 原始 dataset/result 未被覆盖。

### 33.2 数据完整性

- [ ] S1 正式样本数为 2816；
- [ ] S2 正式样本数为 1408；
- [ ] S4 总样本数为 1440，其中新增 1152；
- [ ] 每个 sample ID 唯一；
- [ ] 每个条件都有预期的 64 states；
- [ ] short/long 和颜色间的 state pairing 完整；
- [ ] 11 点 target RGB 与方案表一致；
- [ ] appearance ROI 外的 paired render hash 一致；
- [ ] 没有缺失 prediction 未被标记。

### 33.3 指标完整性

- [ ] 每条视频都有 validity 和 failure reason；
- [ ] valid 视频都有 `omega_hat` 和 fit RMSE；
- [ ] 有完整 `theta_pred[64]`；
- [ ] 有 front/back frequency；
- [ ] 有 low/high family distance；
- [ ] 有 core-mask RGB 和 `alpha_out`；
- [ ] 有 per-frame color trace；
- [ ] NaN/invalid 未被转成 0；
- [ ] intermediate 与 off-family 被分开。

### 33.4 图表完整性

- [ ] 所有频率图共享相同 band 标记；
- [ ] short/long 图共享坐标范围；
- [ ] 每个 valid-only 图旁显示 invalid rate；
- [ ] 3D RGB 图显示输入点、输出点、箭头和参考线；
- [ ] 3D RGB 图 short/long 共享视角；
- [ ] purple 分布保留 raw points；
- [ ] representative videos 按预注册规则选择；
- [ ] 图中百分比可以回溯到 condition table 的原始计数。

### 33.5 解释边界

- [ ] S1 ID 结论与 S2 OOD 结论分开；
- [ ] aligned 条件不声称区分物理和颜色路线；
- [ ] 中间颜色不被强行映射到精确 shortcut 频率；
- [ ] 平均中频被区分为 coherent、seed mixture 或 temporal switch；
- [ ] 输出颜色—频率关系表述为耦合而非输出间因果；
- [ ] short–long 表述为两个匹配 pipeline 的差异。

## 34. 预期最终交付物

1. 冻结的 sweep specification；
2. 输入数据审计报告；
3. S0–S4 的 predictions 和逐样本 metrics；
4. short–long matched comparison；
5. condition-level 和 paired-effect tables；
6. 六面板主图；
7. 完整补充图和 representative videos；
8. 可下钻到逐样本、逐帧和像素颜色分布的 dashboard；
9. artifact inventory 和所有配置/checkpoint/hash；
10. 一份明确区分事实结果、统计不确定性和机制推断的结论摘要。
