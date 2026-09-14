"use client";

import { useState } from "react";
import { PCAView } from "./ResearchMedia";
import { assetUrl } from "./asset-url";

type Mode = "behavior" | "causal";

const modes = {
  behavior: {
    eyebrow: "NATURAL CUE CHANGE",
    title: "Endpoint conflict versus ambiguous cue",
    description: "Physical history, trajectory, renderer, generation seed, and future window are fixed. Only cue appearance changes.",
    videos: [
      { src: "/videos/pendulum/pendulum-selection-conflict.mp4", label: "Endpoint conflict", detail: "red cue · shortcut-associated slow/red future" },
      { src: "/videos/pendulum/pendulum-selection-ambiguous.mp4", label: "Ambiguous cue", detail: "purple cue · history-associated high-frequency future" },
    ],
  },
  causal: {
    eyebrow: "DONOR-FREE CAUSAL EDIT",
    title: "Natural conflict versus state-conditioned edit",
    description: "The held-out receiver, noise, flow-matching schedule, and future window are fixed. The fit-only top-four edit reads no held-out activation donor.",
    videos: [
      { src: "/videos/pendulum/pendulum-controller-natural-conflict.mp4", label: "Natural conflict", detail: "B_082 · no intervention · normalized recovery 0.000" },
      { src: "/videos/pendulum/pendulum-controller-state-edit.mp4", label: "State-conditioned edit", detail: "B_082 · fit-only top-four · normalized recovery 0.879" },
    ],
  },
} as const;

export default function PendulumGenerality() {
  const [mode, setMode] = useState<Mode>("behavior");
  const current = modes[mode];

  return (
    <section className="section pendulumSection" id="pendulum">
      <div className="pendulumIntro">
        <div><span className="chapterNo">06</span><p>SECOND-SYSTEM REPLICATION</p></div>
        <h2>The same causal abstraction<br/><em>appears in Pendulum.</em></h2>
        <p>Appearance and physical history compete in decoded rollouts, while low-order boundary state predicts a donor-free edit that recovers held-out dynamics.</p>
      </div>

      <div className="pendulumTabs" role="tablist" aria-label="Pendulum evidence">
        <button type="button" role="tab" aria-selected={mode === "behavior"} aria-controls="pendulum-tab-panel" onClick={() => setMode("behavior")}>Behavior</button>
        <button type="button" role="tab" aria-selected={mode === "causal"} aria-controls="pendulum-tab-panel" onClick={() => setMode("causal")}>Causal edit</button>
      </div>

      <div className="pendulumTabPanel" id="pendulum-tab-panel" role="tabpanel">
        <div className="pendulumMediaIntro"><span>{current.eyebrow}</span><h3>{current.title}</h3><p>{current.description}</p></div>
        <div className="pendulumVideos">
          {current.videos.map((video) => (
            <figure key={video.src}>
              <video src={assetUrl(video.src)} controls muted loop playsInline preload="metadata" aria-label={`${video.label}: ${video.detail}`}/>
              <figcaption><strong>{video.label}</strong><span>{video.detail}</span></figcaption>
            </figure>
          ))}
        </div>
      </div>

      <div className="pendulumPreviews">
        <a href={assetUrl("/pendulum/behavior.svg")} target="_blank" rel="noreferrer"><img src={assetUrl("/pendulum/behavior.svg")} width="1590" height="1490" alt="Pendulum eleven-hue behavior sweep across low and high histories"/><span>Behavior · 2,816 decoded futures</span></a>
        <a href={assetUrl("/pendulum/decoded-recovery.svg")} target="_blank" rel="noreferrer"><img src={assetUrl("/pendulum/decoded-recovery.svg")} width="3887" height="1983" alt="Pendulum donor-free decoded recovery across 64 held-out receivers"/><span>Causal edit · 64 held-out receivers</span></a>
      </div>

      <div className="interactiveSection"><PCAView task="Pendulum" raw="/interactives/pendulum-project-pca.html" difference="/interactives/pendulum-difference-pca.html"/><p className="interactiveNote">64 fit pairs, 64 held-out pairs at B12. The raw view shows 128 endpoints; the difference view shows 64 edits. Point color is decoded frequency, using the aligned endpoint for differences.</p></div>
      <details className="pendulumDetails">
        <summary>Technical details</summary>
        <div className="pendulumContract">
          <div><span>CHECKPOINT</span><strong>Large · width 1152 · seed 3407 · step 50K</strong><p>Short/Long frequency_color_circle, using the paper's final checkpoints.</p></div>
          <div><span>COHORT</span><strong>64 fit / 64 disjoint held-out</strong><p>Balanced 32/32 by target direction; top-four coordinate basis and scale frozen on fit only.</p></div>
          <div><span>GEOMETRY</span><strong>Held-out R² .975 / .985</strong><p>Target-low and target-high edit coordinates follow direction-specific phase-aligned planes.</p></div>
          <div><span>DECODED RECOVERY</span><strong>63 / 64 near-full</strong><p>Both top-four oracle and fit-only top-four; median recovery .926 and .917, respectively.</p></div>
          <div><span>WRITEABILITY</span><strong>Long closes later in both directions</strong><p>ΔL₅₀ = +9.46 sites for target-low and +7.95 for target-high on n=38 receivers per direction.</p></div>
          <div><span>EVALUATOR</span><strong>One contract for every condition</strong><p>All displayed rollouts pass the current appearance-tolerant geometry and frequency-fit gates.</p></div>
        </div>
        <div className="pendulumDetailFigures">
          <a href={assetUrl("/pendulum/state-geometry.svg")} target="_blank" rel="noreferrer"><img src={assetUrl("/pendulum/state-geometry.svg")} width="2958" height="2006" alt="Fit and held-out Pendulum top-four causal edit-coordinate geometry"/></a>
          <a href={assetUrl("/pendulum/writeability.svg")} target="_blank" rel="noreferrer"><img src={assetUrl("/pendulum/writeability.svg")} width="3837" height="2449" alt="Pendulum Short and Long localized writeability profiles for both target directions"/></a>
        </div>
      </details>
    </section>
  );
}
