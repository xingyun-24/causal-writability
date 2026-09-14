# Paper Release Package Owner（兼 Pendulum Owner）：任务说明

## 任务目标

你是**整篇论文 release package 的唯一总 owner**，负责把最终科学故事整理成一套可以公开发布的产品。职责覆盖 Spring、Pendulum、Projectile 以及论文最终采用的全部任务和结果，包括：

1. **整篇论文的 Project Page**；
2. **整篇论文干净、可复现的 Public GitHub**；
3. **所有任务的 demo、图像、视频及其 provenance**；
4. **发布前的 scientific consistency、reproducibility 和 media QA**；
5. **Pendulum 的科研实验与 release-ready 科学材料**。

“总 owner”意味着你需要主动建立目录、索取材料、发现缺口、追踪补齐、统一格式并完成最终验收，而不是只整理自己做的 Pendulum，也不是等待其他人把所有素材打包好后再拼接。

这不是单纯的协调岗位。除各任务本身的科学实验外，以下 release 工作原则上都由你亲自设计和实现：网页前端与交互、release repo 重构、复现 CLI/脚本、demo 导出与标注工具、asset/provenance registry、文档、测试、部署和最终 release tag。可以向实验负责人确认数值和 claim，但不能把主要实现工作再分散回各实验 owner。

你的核心任务不是把论文全文搬到网页上，也不是公开内部科研目录，而是让陌生研究者能够：

- 30 秒理解论文在问什么；
- 5 分钟看懂最重要的因果证据；
- 看到 intervention 确实改变了最终 decoded video；
- 沿公开代码复现最核心的三条结果。

论文对外故事固定为四步：

1. **Solution selection**：外观 cue 与运动历史会竞争决定生成未来；
2. **State-structured causal route**：模型内部存在由物理状态组织、能够改变 decoded future 的低秩 causal edit family；
3. **Causal writeability / commitment**：错误未来在网络中还能被改写到多深；
4. **Downstream realization**：condition information 通过 self-attention 的 K/V 写入 target stream 后获得生成控制权。

你不负责自行修改这四步故事，也不负责从探索性结果中发明新的 claim。

---

## 一、总体 ownership 与 Pendulum 科研职责

### 1.1 整体 release ownership

你对所有任务的公开材料端到端负责：

- 建立统一的 figure、video、config、manifest 和 provenance 合同；
- 从各实验结果中收集论文最终采用的 audited assets；
- 检查素材是否缺失、过期、与最终 evaluator/claim 不一致；
- 对缺失项建立清单并推动相应实验补齐；
- 将 Spring、Pendulum、Projectile 统一到同一网页叙事和复现入口；
- 对最终网页、GitHub、下载链接和 release tag 负责。

最终应形成一套真正可交付的软件与内容产品，而不只是任务清单或若干整理后的文件夹。

你不需要亲自重做其他人已经完成且通过审计的实验，但不能把其他任务视为“别人提供什么就放什么”。如果素材不完整、无法追溯或不能复现，定位和推动解决仍属于你的责任。

### 1.2 Pendulum：你亲自负责的科研部分

Pendulum 的实验协议由现有实验计划决定；本任务不额外扩展实验轴。你需要把最终通过审计的 Pendulum 结果整理成可发布材料：

- final config、checkpoint ID、seed、数据 manifest 和 evaluator version；
- 64×11 behavior figure 及其统计表；
- matched aligned/conflict 的代表视频；
- canonical causal edit 的 before/after decoded video；
- 若最终机制实验通过，提供 low-rank/state-conditioned edit 与最小 attention/K/V 结果；
- 一份从原始输出到最终图和视频的复现命令。

Pendulum 在论文和网页中的职责是：证明核心 causal abstraction 不只是 Spring renderer 或单一弹簧实现的特例。不要把 Pendulum 扩展成第二套 15-seed、head taxonomy 或完整 FM-window 项目。

Spring、Projectile 等任务的科学结论由最终审定结果决定；你负责主动收集、核对、标准化并发布其图、视频、代码入口和 provenance。你不能自行改变科学结论，但对这些材料是否完整进入 release package 负最终责任。

---

## 二、Project Page

页面按以下顺序组织。顺序是固定叙事，不要把首页做成论文 section 的机械目录：

| Section | 必须展示的内容 | 读者应该理解什么 |
|---|---|---|
| Hero | 标题、问题句和两组紧邻的 matched decoded-video demo | 模型会在多个学到的联合未来中选择；物理与外观可以反向决定对方如何被补全 |
| Solution-selection surface | 64×11 hue-response 主图和少量代表视频 | 连续改变 cue strength，直接观察最终视频采用哪种 continuation |
| Causal writeability | layer-scan 曲线或动画、\(D^\omega\) | 一个已经选中的未来还能在多深的位置被改写 |
| State-conditioned causal editing | \(q\rightarrow z\rightarrow\hat d\rightarrow\) decoded video，并同时展示真实 before/after 视频 | 简单物理状态基可重参数化 causal edit；证据来自最终生成结果，不只是 PCA/probe |
| Commitment mechanism | 极简的 condition K/V \(\rightarrow\) target-stream 示意及最小因果对照 | route 在 self-attention 中写入未来 token 后获得生成控制权 |
| Different seeds, same causal semantics | Large Short 3407 / 3408 / 3409 的并排结果 | 自然行为不同的训练解仍共享 causal semantics，但下游 realization 可以不同 |
| Generalization | Pendulum 和 Projectile 的精简结果 | causal abstraction 是否跨振子和非周期系统成立 |
| Footer | Paper、Code、BibTeX、Authors | 最后才出现阅读、引用和复现入口 |

### 首页首屏：固定文案与双向视频证据

首屏标题固定为：

> # The Physics a World Model Chooses
>
> **When a world model supports multiple futures, what determines which learned dynamics actually controls generation?**

标题下面不要先放作者、摘要、方法图或长段文字，直接放两组同步、可循环播放的视频。桌面端并排，移动端依次排列；两组采用一致的裁切、播放进度、标注位置和物理量显示。

#### Demo A — Physics \(\rightarrow\) Appearance

展示：

```text
purple ball + fast observed dynamics
                 ↓
       dynamics-conditioned completion
                 ↓
blue future + the same fast dynamics
```

也就是：

\[
\text{purple-fast}\rightarrow\text{blue-fast}.
\]

- 左侧让读者看到 ambiguous/purple cue 和 fast 历史；右侧展示 decoded future 变为 blue，同时 fast dynamics 保持。
- 视频旁只标 `Input / Observed history`、`Decoded future` 和必要的 fitted physical metric，不要在首屏写 block、PCA、head 或 tensor 术语。
- 这组首先证明的是：当外观 cue 模糊时，历史动力学能够选择与 fast mode 关联的外观—动力学联合 continuation。

#### Demo B — Appearance \(\rightarrow\) Physics

展示：

\[
\text{red-fast}\rightarrow\text{red-slow}.
\]

- 输入保持 red，但历史真实动态为 fast；decoded future 保持 red，并转向训练中与 red 关联的 slow dynamics。
- 这组证明强 appearance cue 可以反向选择动力学 continuation。

两组视频下方只放一句 headline takeaway：

> **The model contains a joint appearance–dynamics causal structure: depending on cue strength, physical history or appearance can determine which learned continuation controls the decoded future.**

如果最终确有通过审计、能够只改变目标 component 且保持非目标 component 的**内部 activation intervention**，可将第二句升级为：

> **We can causally alter which component controls the decoded future while preserving the matched non-target state.**

否则不要把自然生成或 input-level counterfactual cue manipulation 标成 activation intervention。网页应明确区分：

- `matched input intervention / counterfactual completion`：只改变输入 cue 或物理历史；
- `internal causal intervention`：在指定 layer/site 注入 activation edit。

Hero 的两组 demo 优先使用同一个冻结 matched trajectory family。每组至少满足：

- amplitude、phase、boundary state、trajectory/nuisance 参数相同；
- renderer seed 和 generation seed 相同；
- 除明确写出的目标变量外，其他输入因素不变；
- 两个输出均通过当前 evaluator 的 validity 与物理拟合；
- Demo A 定量验证 fast frequency 保持、future color 向 blue 移动；
- Demo B 定量验证 future color 保持 red、frequency 从 fast 转向 slow；
- 候选从预先冻结的 eligible registry 中选取，并记录筛选口径，不能看完 intervention 结果后无约束挑最漂亮样本。

如果找不到同时通过上述 non-target preservation gate 的样本，就保留“双向 joint-mode completion”的较弱但准确叙述，不得为了首页视觉效果声称 selective factor control。

### 首屏之后的五个核心模块

1. **Solution-selection surface**

   放 64×11 主图，并使用固定一句话：

   > **We directly sweep physical and shortcut evidence and ask which future the decoded video realizes.**

2. **Causal writeability**

   放 layer-scan animation 或简洁曲线，并以问题引导：

   > **How long can an already selected future still be rewritten?**

3. **State-conditioned causal editing**

   明确显示

   \[
   q\rightarrow z\rightarrow\hat d\rightarrow\text{decoded video}.
   \]

   这一节必须有实际 decoded before/after 视频和物理读出；PCA、energy、coordinate fit 只能作为辅助证据，不能替代行为结果。

4. **Commitment mechanism**

   用一张极简图说明 condition-side K/V 如何经 self-attention 写入 target/future stream。只配支持该箭头的最小 intervention 对照，不在首页展开 seed-specific head taxonomy。

5. **Different seeds, same causal semantics**

   并排展示 3407 / 3408 / 3409：自然 physics-follow 明显不同，但均存在可用的 causal route，且对齐后具有共享语义。网页正文只讲 shared causal semantics 与 distinct downstream realization；具体 head 差异放补充页或论文 appendix。

Pendulum / Projectile 的 generalization 证据紧随这五项之后。页面最底部才放 `Paper / Code / BibTeX / Authors`，不要在首屏和主证据流之间插入重复导航、长作者列表或引用模块。

### 视觉与交互质量

页面必须达到正式论文项目页的完成度，而不是默认模板加若干静态图片：

- 建立统一的 typography、spacing、颜色、图例和 motion system；
- Hero 使用真正有信息量的视频对照，首屏不堆公式和内部术语；
- before/after intervention、Short/Long、不同 cue 尽量使用统一的同步播放或切换组件；
- 64×11 response figure 应支持清楚查看 true-low/true-high 与 cue 轴，必要时提供轻量交互，但不能牺牲静态论文图的准确性；
- 动画只用于解释因果顺序和输入输出关系，不做无意义装饰；
- 桌面端和移动端都要经过人工检查；
- 视频应有 poster、静音自动播放、暂停和 fallback；
- 页面应快速加载，首屏不一次下载所有高码率视频；
- 保持可访问性：足够对比度、alt text、键盘可用、避免只靠红蓝颜色传达类别。

视觉风格需要经过反复比较和删改。不能把“内容都放上去了”视为设计完成。

### 64×11 图必须解释准确

- 64 条独立基础物理轨迹；
- 每条轨迹测试 11 个 red→blue hue；
- 每个 checkpoint 共 704 个真实生成 rollout；
- 比较模型时使用相同 manifest 和 generation seeds；
- 统计单位是 64 条 trajectory，不把同一轨迹的 11 个 hue 当成独立样本。

网页应突出最终生成行为，而不是只展示 activation probe、PCA scatter 或 tensor statistic。

### Causal intervention demo 的额外约束

- 后续 state-conditioned editing demo 必须同时包含 natural conflict、明确的 internal intervention 和最终 decoded future；
- 标清 intervention layer/site、fit split 与 evaluator，但这些信息放在可展开详情或 provenance 中，不堆在视频画面上；
- 不得把 joint-mode edit 误写为纯 physics subspace；强 edit 可能同时改变 dynamics 和训练关联的 appearance；
- 只有 non-target preservation 经过冻结标准审计时，才使用 `selective control`；否则使用 `state-conditioned joint continuation edit`；
- 不得用 activation/PCA 变化代替 decoded-video causal effect。

---

## 三、Demo 与 Media Assets

每个进入网页或社交媒体的核心 demo 至少保留：

- 原始高质量 MP4；
- project-page 使用版本；
- clean version；
- annotated version；
- WebM/GIF 仅作为兼容版本。

核心网页稳定后，再按需要制作横版、方形或竖版裁切；多平台裁切不是第一阶段阻塞项。

Annotated version 只标关键语义，例如：

- Natural Conflict；
- Causal Intervention；
- Decoded Future；
- fitted \(\omega\)、gravity 或任务对应物理量。

不要在 demo 上堆叠 block、head、gain 和 tensor shape。

### 每个 demo 必须有 provenance

建议统一保存为 `demo_registry.jsonl`，每条至少包含：

```text
demo_id
task
checkpoint_path_or_release_id
checkpoint_hash
training_seed
trajectory_id / pair_id
split
true physical state
input cue
generation_seed
natural outcome
intervention layer/site
intervention type
fit-only model or basis version
gain, if used
before/after physical metric
validity
source video paths
exported asset paths
paper figure/section correspondence
evaluator version
```

任何公开 demo 都必须能够从 registry 回到原始实验记录，不能只保留剪辑后的视频。

---

## 四、Public GitHub

不要直接公开内部 runtime、服务器绝对路径、历史失败目录或完整 autoresearch 工作树。应从经过批准的 release branch 整理干净入口。

最低交付包括：

```text
README.md
CLAIMS.md
LICENSE / NOTICE
environment 或 requirements
final configs
data / manifest construction
behavior evaluation
matched causal patching
writeability aggregation
low-rank / state-conditioned editing
demo export
checkpoint and dataset instructions
tests / smoke tests
```

README 必须提供三个最短复现路径：

1. **Behavior**：生成或读取固定 64×11 manifest，复现 solution-selection figure；
2. **Commitment**：构造 strict matched bank，运行 condition-residual patching，计算 recovery、layer curve 与 \(D^\omega\)；
3. **Causal editing**：input physical state → simple physical basis → low-rank coordinates → intervention → decoded video → physical evaluator。

每条路径都应说明：

- 所需数据和 checkpoint；
- 一条可复制的命令；
- 预计硬件、时间和磁盘；
- 预期生成的文件；
- 与论文 figure/table 的对应关系。

Checkpoint 不直接塞进 Git。提供稳定下载位置、SHA256、许可证/使用说明和加载示例。

---

## 五、Scientific Claim 边界

建立 `CLAIMS.md`，所有网页、README、caption 和 demo annotation 统一遵守。

可以使用：

- state-structured causal route；
- causal writeability / causal commitment；
- condition K/V→target write；
- shared causal abstraction with distinct downstream realizations；
- activation intervention changes the physical dynamics in the decoded rollout。

不能写成：

- physics neuron；
- pure/disentangled physics subspace；
- 模型没有学会 unsupported middle dynamics；
- 某个 K head 是所有模型的通用物理机制；
- physics 可以在不影响 appearance 的情况下独立控制；
- \(D^\omega\) 表示 physics representation 位于哪里。

必须保持：

- route 可能共同改变 appearance 与 dynamics，因此称为 joint-continuation / shortcut-breaking causal route；
- \(D^\omega\) 衡量 causal revisability/access length，不是 representation depth；
- block、head、gain 和 K/V 分工可能随 seed 与任务改变；
- Pendulum/Projectile 结果未冻结前只能放 placeholder，不能先写成已验证结论。

论文标题、`first/to our knowledge`、最终 causal claim 和所有超出上述范围的文字，必须经过论文负责人确认。

---

## 六、执行顺序

### Phase 1：先完成 Pendulum vertical slice

- 先用 Pendulum 做出一个**完整而非占位的端到端样板**；
- 完成 project page 的 Hero、behavior、causal edit 和 reproduction 页面组件；
- 建立 clean release branch/repo，并让 Pendulum 的最短复现路径真实可运行；
- 建立 `CLAIMS.md`、`demo_registry.jsonl` schema 和 release checklist；
- 整理 Pendulum 的 final config、manifest、checkpoint ID、figure、视频和 provenance；
- 完成 Pendulum clean-clone smoke test；
- 同时为 Spring、Projectile 建立 asset inventory 和 placeholder，但暂不为了填满页面加入未冻结结果。

Pendulum vertical slice 的作用是把设计系统、组件、目录合同和复现流程先做通。最终网页的主 Hero 和叙事中心仍由论文最终故事决定，不因为 Pendulum 最先实现就把它包装成论文主案例。

### Phase 2：逐步接入 Spring 主故事

- 将 Spring 的 solution selection、state-structured route、writeability/commitment 和 downstream realization 依次接入已有组件；
- 用 Spring final audited assets 替换对应 placeholder；
- 为每个 figure/demo 补齐 provenance 和 hash；
- 补齐 behavior、commitment、causal editing 三条论文核心复现路径；
- 确认网页数值、caption、README 和论文一致。

### Phase 3：接入 Projectile 并完成跨任务统一

- 接入 Projectile behavior、causal edit 和最终通过的最小 downstream mechanism；
- 统一三个任务的视觉编码、术语、evaluator 说明和 provenance；
- 删除不能支持主故事的临时 panel，不把 Generalization 做成第二篇论文；
- 冻结完整 asset inventory。

### Phase 4：投稿/公开前

- 在全新环境做 clean-clone smoke test；
- 检查所有链接、视频编码、移动端和无声自动播放；
- 检查 checkpoint/data 下载及 SHA256；
- 请未参与项目的人完成 30 秒理解测试和一次最短复现；
- 修复阻塞问题后冻结 release tag。

---

## 七、Agent 工作时长与迭代合同

如果主要由 coding agent 执行，不接受“一轮生成网页后即交付”。建议为第一阶段 Pendulum vertical slice 提供 **8–12 个有效 agent 工作小时**，并要求 agent 持续工作到阶段验收通过；时间只是防止过早停在 skeleton，最终仍以质量门为准。

至少完成以下四轮：

1. **结构轮**：信息架构、数据合同、组件和复现路径完整；
2. **视觉轮**：重新审视排版、留白、视频对照、图表层级和移动端，不沿用第一版默认样式；
3. **科学与复现轮**：逐项核对 claim、数字、manifest、provenance，并在 clean environment 实跑命令；
4. **陌生读者轮**：根据未参与项目者的 30 秒理解测试再次删改。

每轮必须留下简短的 changelog 和未解决问题清单。Agent 不应为了凑时长制造额外功能；若某项设计不能提高理解、可信度或复现性，应删除而不是继续装饰。

第一阶段只有同时满足以下条件才可宣布完成：

- Pendulum 页面不是 placeholder；
- 至少一个自然 demo 和一个 causal intervention demo 可正常播放且可追溯；
- Pendulum behavior 与 causal-edit 最短复现命令在 clean clone 中通过；
- desktop/mobile、加载性能和基本 accessibility 通过；
- 已经完成至少三次明显的设计/内容 revision，而非只修改文字和颜色。

---

## 八、最终验收标准

任务完成必须同时满足：

1. 陌生读者能在 30 秒回答：论文问题是什么、最重要发现是什么、intervention 是否改变了真实生成视频、commitment 是什么意思；
2. 网页四步故事与论文一致，没有把探索性细节写成主结论；
3. 每个公开 figure/video 都有可追溯 provenance；
4. clean clone 后可按 README 复现至少 behavior、commitment 和 causal editing 三条路径；
5. Pendulum 的 final figure、demo、config、manifest、checkpoint ID 和复现命令齐全；
6. Spring、Pendulum、Projectile 及论文实际保留的所有结果都已进入统一 asset/provenance inventory，缺口均关闭或明确记录；
7. Public GitHub 不包含私有路径、密钥、未授权权重、内部日志或无关历史实验；
8. Paper、网页、README、caption 和数值表述通过最终一致性检查。

一句话边界：

> 你对整篇论文的公开 release 端到端负责：把 solution selection、state-structured causal route、causal commitment 和 downstream realization 做成一个能看懂、能看到真实生成效果、能沿代码复现的公开产品；Pendulum 同时是你亲自负责的科研任务。你不能自行扩大论文 claim，但必须主动推动所有任务的发布材料达到完整、可追溯、可复现的标准。
