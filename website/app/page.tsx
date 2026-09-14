import PendulumGenerality from "./PendulumGenerality";
import FreefallGenerality from "./FreefallGenerality";
import SupplementGallery from "./SupplementGallery";
import { PCAView } from "./ResearchMedia";
import { citation, publication } from "./site-content";

type PanelProps = {
  label: string;
  title: string;
  src: string;
  width: number;
  height: number;
  alt: string;
  children: React.ReactNode;
  reverse?: boolean;
  compact?: boolean;
};


function FigurePanel({ label, title, src, width, height, alt, children, reverse, compact }: PanelProps) {
  return (
    <article className={`figurePanel ${reverse ? "reverse" : ""} ${compact ? "compact" : ""}`}>
      <a className="panelImage" href={src} target="_blank" rel="noreferrer">
        <img src={src} width={width} height={height} alt={alt}/>
        <span>Full resolution ↗</span>
      </a>
      <div className="panelCopy">
        <span>{label}</span>
        <h3>{title}</h3>
        {children}
      </div>
    </article>
  );
}

function ResearchVideo({ label, title, detail, src }: { label: string; title: string; detail: string; src: string }) {
  return (
    <article className="researchVideo">
      <div className="videoTop"><span>{label}</span><i>64 FRAMES · 20 FPS</i></div>
      <video aria-label={`${title}: ${detail}`} src={src} muted loop playsInline controls preload="metadata">
        <a href={src}>Open the MP4 video</a>
      </video>
      <div className="videoMeta"><div><strong>{title}</strong><span>{detail}</span></div><a href={src} target="_blank" rel="noreferrer">Open MP4 ↗</a></div>
    </article>
  );
}

function ChapterIntro({ number, eyebrow, title, children }: { number: string; eyebrow: string; title: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="chapterIntro">
      <div><span className="chapterNo">{number}</span><p>{eyebrow}</p></div>
      <h2>{title}</h2>
      <div className="chapterLead">{children}</div>
    </div>
  );
}

export default function Home() {
  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top"><span className="brandMark">WM</span><span>WORLD MODEL MECHANISMS</span></a>
        <div className="topMeta"><span className="statusDot">Local preview</span><a href="/paper/main.pdf">Paper ↗</a></div>
      </header>
      <nav className="sectionNav" aria-label="Section navigation">
        <a href="#top">00 Abstract</a><a href="#overview">01 Overview</a><a href="#choice">02 Selection</a><a href="#rewrite">03 Control</a><a href="#commit">04 Commitment</a><a href="#realize">05 Realization</a><a href="#pendulum">06 Pendulum</a><a href="#freefall">07 Free Fall</a><a href="#atlas">08 Supplement</a>
      </nav>

      <section className="hero" id="top">
        <div className="heroEyebrow"><span>PHYSICS</span><span>PREDICTIVE SHORTCUTS</span><span>CAUSAL CONTROL</span></div>
        <div className="heroGrid">
          <div className="heroCopy">
            <p className="paperLabel">{publication.subtitle}</p>
            <h1>A Chosen Future<br/>Can Still Be <em>Rewritten.</em></h1>
            <p className="lede">When appearance cues conflict with physical history, which training-supported continuation controls generation, and does the rejected continuation remain causally accessible within the model?</p>
            <div className="heroActions"><a href="#overview" className="primary">View the evidence <b>↓</b></a><span>Decoded-video measurements · causal interventions · cross-run transfer</span></div>
          </div>
          <div className="heroThesis">
            <span>CENTRAL FINDING</span>
            <blockquote>Physical structure can remain causally available without controlling natural generation.</blockquote>
            <p>Natural generation identifies the selected continuation; intervention identifies alternative continuations that remain causally accessible.</p>
            <div className="thesisFlow"><i>selection</i><b>≠</b><i>availability</i><b>→</b><i>causal authority</i></div>
          </div>
        </div>
        {publication.trailer && <div className="trailerStage"><video src={publication.trailer} poster={publication.trailerPoster} controls playsInline preload="metadata" aria-label="Research overview film"/></div>}
        <div className="heroStats">
          <div><span>704</span><p>decoded futures in the 64 × 11 cue sweep</p></div>
          <div><span>23 / 26</span><p>endpoint shortcut failures corrected at the purple cue</p></div>
          <div><span>4-D</span><p>compact route retaining nearly the full causal effect</p></div>
          <div><span>6 / 6</span><p>successful directed transfers across run pairs</p></div>
        </div>
      </section>

      <section className="section" id="overview">
        <ChapterIntro number="01" eyebrow="MECHANISTIC OVERVIEW" title={<>A shortcut can dominate while<br/><em>an alternative future remains writable.</em></>}>
          <p>The analysis separates behavioral selection, causal accessibility, commitment with depth, and downstream realization.</p>
        </ChapterIntro>
        <div className="overviewGrid">
          <FigurePanel label="FIGURE 1A · TRAINING STRUCTURE" title="Natural correlations support predictive shortcuts" src="/paper/vector/fig1a-context.svg" width={530} height={360} alt="Natural contexts in which motion and appearance are correlated">
            <p>Context, motion, and future appearance are correlated in the training distribution, permitting appearance-based prediction without exclusive reliance on physical history.</p>
          </FigurePanel>
          <FigurePanel label="FIGURE 1B · CONTROLLED CONFLICT" title="Cue conflict reveals solution choice" src="/paper/vector/fig1b-conflict.svg" width={580} height={360} alt="Appearance cue varied while physical history is held fixed" reverse>
            <p>Physical history is held fixed while the appearance cue is varied continuously. The decoded continuation can switch between shortcut-consistent and history-consistent modes.</p>
          </FigurePanel>
          <FigurePanel label="FIGURE 1C · CAUSAL INTERVENTION" title="Internal edits recover the rejected continuation" src="/paper/vector/fig1c-rewrite.svg" width={780} height={320} alt="State-conditioned edit changing the decoded future">
            <p>A state-conditioned internal edit changes the final decoded motion, establishing a causal effect on generation rather than a correlational probe readout.</p>
          </FigurePanel>
          <FigurePanel label="FIGURE 1D · COMMITMENT" title="Condition-to-target writes close causal writeability" src="/paper/vector/fig1d-commitment.svg" width={450} height={330} alt="Condition keys and values writing the selected future into target states" reverse>
            <p>A future is writable while condition-side intervention can redirect it. Commitment occurs when condition-to-target writes make the same intervention ineffective at subsequent depths.</p>
          </FigurePanel>
        </div>
      </section>

      <section className="section dark" id="choice">
        <ChapterIntro number="02" eyebrow="NATURAL SOLUTION SELECTION" title={<>Cue strength selects among<br/><em>training-supported continuations.</em></>}>
          <p>Fitted frequency and future RGB are measured directly from decoded rollouts. The trajectory, rather than the individual cue-conditioned rollout, is the statistical unit.</p>
        </ChapterIntro>
        <div className="landscapePair">
          <article><div><span>FIGURE 2A · SLOW HISTORY</span><h3>Selection under slow physical evidence</h3><p>As the input cue changes from red to blue, decoded futures move from the slow mode through an intermediate region toward the fast mode.</p></div><a href="/paper/vector/fig2a-slow.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig2a-slow.svg" width="600" height="404" alt="Decoded cue sweep for slow physical histories"/></a></article>
          <article><div><span>FIGURE 2A · FAST HISTORY</span><h3>Selection under fast physical evidence</h3><p>The corresponding sweep under fast history separates the effect of cue strength from the direction of the physical evidence.</p></div><a href="/paper/vector/fig2a-fast.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig2a-fast.svg" width="540" height="404" alt="Decoded cue sweep for fast physical histories"/></a></article>
        </div>
        <div className="evidenceBand">
          <div className="bigStat"><span>88.5%</span><strong>23 OF 26 ENDPOINT FAILURES</strong><p>switch to the history-consistent frequency at the purple cue: 8/10 fast histories and 15/16 slow histories.</p></div>
          <div><span>ESTIMAND</span><h3>Conditional correction among endpoint shortcut failures</h3><p>The denominator contains trajectories for which the strongly conflicting endpoint already produces the shortcut-associated frequency. This quantity is not an unconditional accuracy estimate.</p></div>
        </div>
        <FigurePanel label="FIGURE 2B · MATCHED EXAMPLES" title="Motion and appearance switch jointly" src="/paper/vector/fig2b-examples.svg" width={1020} height={430} alt="Matched decoded examples showing joint changes in motion and appearance">
          <p>With physical history and future window fixed, changing only the cue alters both fitted frequency and generated appearance. The selected object is therefore a joint appearance–dynamics continuation.</p>
        </FigurePanel>
        <FigurePanel label="FIGURE 2C · HISTORY CONTROL" title="Longer histories resist stronger conflicting cues" src="/paper/vector/fig2c-history.svg" width={1210} height={280} alt="Short versus long history control" reverse compact>
          <p>Long histories retain physics-consistent behavior under stronger cue conflict than short histories, providing a same-seed control for the strength of physical evidence.</p>
        </FigurePanel>
        <div className="videoSection">
          <div className="mediaIntro"><span>DECODED ROLLOUTS</span><h3>Matched behavioral comparisons</h3><p>These are natural generations. Within each pair, physical history, generation seed, and future window are fixed; only the appearance cue changes.</p></div>
          <div className="videoPair"><p className="pairContract">Same fast physical history; only the appearance cue changes.</p><div className="videoGrid two"><ResearchVideo label="FAST HISTORY · RED ENDPOINT" title="Shortcut-consistent continuation" detail="red cue · red/slow" src="/videos/selection-fast-red-endpoint.mp4"/><ResearchVideo label="FAST HISTORY · PURPLE CUE" title="History-consistent continuation" detail="purple cue · blue-ish/fast" src="/videos/selection-fast-purple.mp4"/></div></div>
          <div className="videoPair"><p className="pairContract">Same slow physical history; only the appearance cue changes.</p><div className="videoGrid two"><ResearchVideo label="SLOW HISTORY · BLUE ENDPOINT" title="Shortcut-consistent continuation" detail="blue cue · blue/fast" src="/videos/selection-slow-blue-endpoint.mp4"/><ResearchVideo label="SLOW HISTORY · PURPLE CUE" title="History-consistent continuation" detail="purple cue · red-ish/slow" src="/videos/selection-slow-purple.mp4"/></div></div>
        </div>
      </section>

      <section className="section" id="rewrite">
        <ChapterIntro number="03" eyebrow="STATE-STRUCTURED CAUSAL CONTROL" title={<>A compact route can be called<br/><em>without a held-out donor.</em></>}>
          <p>Matched replacement identifies the writable route; low-rank analysis and a fit-only controller then test whether the route can be synthesized prospectively.</p>
        </ChapterIntro>
        <FigurePanel label="FIGURE 3A · MATCHED REPLACEMENT" title="Matched state replacement recovers the rejected future" src="/paper/vector/fig3a-replacement.svg" width={1120} height={190} alt="Matched state replacement recovering the rejected continuation">
          <p>Replacing the receiver state with the matched target difference recovers the alternative continuation and identifies a functional route site. Donor activations are used only for route discovery.</p>
        </FigurePanel>
        <div className="triptych">
          <article><span>FIGURE 3B · TARGET FAST</span><h3>Fast-route coordinates vary with boundary phase</h3><a href="/paper/vector/fig3b-fast-phase.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig3b-fast-phase.svg" width="370" height="410" alt="Fast-route coordinates organized by boundary phase"/></a><p>The frozen coordinates trace a smooth phase-dependent family.</p></article>
          <article><span>FIGURE 3B · TARGET SLOW</span><h3>Slow-route coordinates form a complementary family</h3><a href="/paper/vector/fig3b-slow-phase.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig3b-slow-phase.svg" width="420" height="410" alt="Slow-route coordinates organized by boundary phase"/></a><p>The opposite target direction remains structured by the same low-order boundary variables.</p></article>
          <article><span>FIGURE 3B · LOW-RANK EFFECT</span><h3>Four coordinates retain nearly full recovery</h3><a href="/paper/vector/fig3b-recovery.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig3b-recovery.svg" width="330" height="380" alt="Recovery retained by a four-dimensional edit"/></a><p>The four-dimensional edit approaches the decoded effect of the complete matched difference.</p></article>
        </div>
        <div className="controllerGrid">
          <figure><a href="/paper/vector/fig3c-controller-model.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig3c-controller-model.svg" width="450" height="340" alt="Fit-only boundary-phase controller"/></a><figcaption><span>FIGURE 3C · CONTROLLER</span><strong>Direction chooses the route; phase locates the state</strong><p>For each requested direction, the controller uses [1, cos θ*, sin θ*] to predict the top-four route coordinates from input-boundary phase.</p></figcaption></figure>
          <div className="controllerCopy"><span>HELD-OUT INTERVENTION</span><h3>Prospective control without donor activations</h3><p>After fitting and freezing the controller, a requested target direction and held-out boundary phase are sufficient to synthesize the condition-state edit. Held-out coordinate R² is .83–.88, compared with .51–.60 for direction alone.</p><div><b>R = .99</b><small>decoded recovery in the illustrated held-out replay</small></div></div>
          <figure><a href="/paper/vector/fig3c-controller-replay.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig3c-controller-replay.svg" width="610" height="440" alt="Held-out decoded replay after controller intervention"/></a><figcaption><span>FIGURE 3C · DECODED EFFECT</span><strong>The alternative continuation is recovered</strong><p>The intervention changes the final pixel trajectory under fixed receiver input and noise.</p></figcaption></figure>
        </div>
        <div className="videoSection lightVideo"><div className="mediaIntro"><span>HELD-OUT ROLLOUT</span><h3>Natural and intervened generation</h3><p>The same held-out receiver, noise, and future window are used. The fit-only controller changes only the condition prefix at the frozen rewrite site.</p></div><div className="videoGrid two"><ResearchVideo label="NATURAL CONFLICT" title="Selected shortcut" detail="red/slow · no intervention" src="/videos/controller-natural-conflict.mp4"/><ResearchVideo label="STATE-CONDITIONED EDIT" title="Recovered continuation" detail="blue/fast · no held-out donor" src="/videos/controller-state-edit.mp4"/></div></div>
        <div className="interactiveSection"><PCAView task="Spring" raw="/interactives/spring_raw_projection_interactive.html" difference="/interactives/spring_matched_difference_interactive.html"/><p className="interactiveNote">The same fit-only PCA basis projects held-out raw activations and matched differences. B6, seed 3408, 50K; 128 fit pairs and 128 held-out pairs. Color denotes decoded frequency, with aligned frequency used for differences.</p></div>
      </section>

      <section className="section tinted" id="commit">
        <ChapterIntro number="04" eyebrow="CAUSAL WRITEABILITY" title={<>Writeability predicts fate<br/><em>as training consolidates a future.</em></>}>
          <p>Layerwise intervention profiles quantify where a selected future remains causally revisable.</p>
        </ChapterIntro>
        <div className="panelGrid twoByTwo">
          <article><a href="/paper/vector/fig4a-training.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig4a-training.svg" width="550" height="490" alt="Writeability contraction over training"/></a><div><span>FIGURE 4A · TRAINING</span><h3>Writeability contracts with training</h3><p>The depth range over which intervention redirects shortcut failures becomes progressively narrower.</p></div></article>
          <article><a href="/paper/vector/fig4b-solutions.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig4b-solutions.svg" width="670" height="490" alt="Physics-following improves while remaining errors become less writable"/></a><div><span>FIGURE 4B · TRAINING TRAJECTORIES</span><h3>Better behavior, less writable remaining errors</h3><p>Across 15 seeds, mean full-grid physics-follow rate rises from .52 to .60 while remaining-error writability falls from 10.88 to 9.55 sites between 5K and 100K.</p></div></article>
          <article><a href="/paper/vector/fig4c-fate.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig4c-fate.svg" width="550" height="464" alt="Early writeability predicting later error fate"/></a><div><span>FIGURE 4C · PROSPECTIVE FATE</span><h3>Early writeability predicts subsequent correction</h3><p>Failures later corrected by training (n = 157) are writable at 3.79 [1.57, 6.53] more sites than persistent failures (n = 953).</p></div></article>
          <article><a href="/paper/vector/fig4d-history.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig4d-history.svg" width="660" height="464" alt="Short versus long history writeability profiles"/></a><div><span>FIGURE 4D · HISTORY CONTROL</span><h3>Long histories remain writable at greater depth</h3><p>The effect appears in all three matched-seed 50K comparisons. Short and Long use checkpoint-local strict banks, not trajectory-paired banks.</p></div></article>
        </div>
      </section>

      <section className="section dark" id="realize">
        <ChapterIntro number="05" eyebrow="DOWNSTREAM CAUSAL AUTHORITY" title={<>Condition K/V writes<br/><em>realize the selected future.</em></>}>
          <p>Target-route persistence, cross-run transfer, and path restoration localize how the shared route acquires downstream authority.</p>
        </ChapterIntro>
        <div className="panelGrid mechanismGrid">
          <article><a href="/paper/vector/fig5a-persistence.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig5a-persistence.svg" width="690" height="300" alt="Target-route response after direct writeability closes"/></a><div><span>FIGURE 5A · PERSISTENCE</span><h3>The target-route signal persists after direct rewrite closes</h3><p>Route-aligned target responses remain measurable after condition-side frequency writeability has closed.</p></div></article>
          <article><a href="/paper/vector/fig5b-transfer.svg" target="_blank" rel="noreferrer"><img src="/paper/vector/fig5b-transfer.svg" width="460" height="230" alt="Cross-run causal coordinate transfer"/></a><div><span>FIGURE 5B · CROSS-RUN TRANSFER</span><h3>Causal coordinates transfer across independent runs</h3><p>All six directed run-pair transfers achieve decoded recovery R = .94–.99 using fit-only scale and rotation.</p></div></article>
        </div>
        <FigurePanel label="FIGURE 5C · PATH RESTORATION" title="Restoring conflict K/V erases recovery" src="/paper/vector/fig5c-kv.svg" width={1180} height={320} alt="Restoration of conflict keys and values erasing causal recovery">
          <p>Within a successful matched edit, restoring conflict K or K+V reduces final recovery to approximately zero, whereas restoring Q leaves recovery near one. The causal bottleneck is therefore localized to condition-to-target key/value writes.</p>
        </FigurePanel>
        <FigurePanel label="FIGURE 5D · SINGLE-HEAD INTERVENTION" title="One value head exhibits a finite causal margin" src="/paper/vector/fig5d-head.svg" width={1250} height={340} alt="Dose response of a single value-head intervention" reverse>
          <p>Changing one of nine condition-token value heads yields a continuous dose response. Clean physics peaks at gain 8; continuous frequency recovery can overshoot at larger gains and increasingly leave the supported modes.</p>
        </FigurePanel>
        <div className="videoSection"><div className="mediaIntro"><span>BOTTLENECK INTERVENTION</span><h3>Decoded causal effect</h3><p>The same selection-clean receiver and future window are used; only condition-token V in head 8 is edited.</p></div><div className="videoGrid two"><ResearchVideo label="NATURAL CONFLICT" title="Natural continuation" detail="red/slow" src="/videos/realization-natural-conflict.mp4"/><ResearchVideo label="TARGETED V/H8 EDIT" title="Intervened continuation" detail="blue/fast" src="/videos/realization-vh8-edit.mp4"/></div></div>
        <div className="fmTimingSection" id="fm-timing">
          <div className="fmTimingIntro">
            <span>GENERATION TIME · APPENDIX FIGURE 31</span>
            <h3>When the write acts matters</h3>
            <p>Network depth tells us where a write acts within one forward pass. Video generation also unfolds over 20 flow-matching calls, each passing through the network again. We therefore test when the same physical write can influence the final video.</p>
            <p>Here, early and late refer to the first and second halves of the denoising sequence, not training checkpoints, network layers, or the first and second halves of the generated video.</p>
          </div>
          <figure className="fmTimingFigure">
            <a href="/paper/appendix/figure-31.svg" target="_blank" rel="noreferrer"><img src="/paper/appendix/figure-31.svg" alt="On the same 48 trajectories, early calls 0–9 produce an internal response but little decoded recovery; late calls 10–19 recover motion, and all calls 0–19 improve recovery further." loading="lazy"/></a>
            <figcaption>The same gain-8 condition-V/head-8 write is applied to the same 48 receivers. Early-only writing changes the internal route response but is nearly ineffective in the decoded motion. Late-call writing restores motion; writing over all calls improves recovery further. An internal response alone does not establish control over the final generation. <a href="/paper/main.pdf#page=33" target="_blank" rel="noreferrer">Figure 31 and methods ↗</a></figcaption>
          </figure>
          <details className="fmTimingDetails"><summary>Temporal-window control</summary><p>In a separate 16-receiver comparison, neither calls 10–14 nor calls 15–19 rescues motion alone, whereas their joint 10–19 window does. The result is not simply “edit the last few steps”: it identifies an effective late window for this intervention. These tests do not establish a universal timing rule for every model, task or write.</p></details>
        </div>
      </section>

      <PendulumGenerality/>
      <FreefallGenerality/>

      <section className="section" id="atlas">
        <ChapterIntro number="08" eyebrow="SUPPLEMENTARY RESULTS" title={<>Robustness and<br/><em>mechanistic scope.</em></>}>
          <p>Figures 7–33 from the final-paper appendix cover behavior, pretrained adaptation, controller controls, Pendulum and Free Fall, training-time writability, and downstream mechanisms.</p>
        </ChapterIntro>
        <SupplementGallery/>
      </section>

      <section className="scope section" id="resources">
        <div className="scopeGrid">
          <div><span className="chapterNo">07</span><h2>Scope of inference</h2><p>The evidence supports a compact, state-dependent causal route whose availability can be dissociated from its natural causal authority. The route jointly controls appearance and dynamics.</p></div>
          <div className="claimCards"><article><span>SUPPORTED</span><strong>State-structured causal accessibility</strong><p>Alternative continuations remain internally callable even when they do not control natural generation.</p></article><article><span>NOT ESTABLISHED</span><strong>A pure physics representation</strong><p>The results do not imply classical disentanglement, a universal mechanism across solutions, or calibrated model confidence.</p></article></div>
        </div>
        <details className="publicationDetails"><summary>Authors and citation</summary><p>Xingyun Wang*, Haomin Zheng*, Man Yuan, Leqian Yang, Ziming Liu</p><p>Tsinghua University · Peking University · University of Science and Technology of China · MetaCircle · Shanghai Qi Zhi Institute</p><p>* Equal contribution. <a href="mailto:xingyun-24@mails.tsinghua.edu.cn">xingyun-24@mails.tsinghua.edu.cn</a> · <a href="mailto:zmliu@tsinghua.edu.cn">zmliu@tsinghua.edu.cn</a></p><pre>{citation}</pre></details>
        <footer><div className="brand"><span className="brandMark">WM</span><span>WORLD MODEL MECHANISMS</span></div><p>Research website · manuscript figures and quantitative results</p><a href="#top">Back to top ↑</a></footer>
      </section>
    </main>
  );
}
