import { PCAView, VideoPair } from "./ResearchMedia";

export default function FreefallGenerality() {
  return <section className="section pendulumSection" id="freefall">
    <div className="pendulumIntro"><div><span className="chapterNo">07</span><p>NON-OSCILLATORY REPLICATION</p></div>
      <h2>A physical alternative<br/><em>in Free Fall.</em></h2>
      <p>A fit-only gravity controller changes the decoded trajectory in a non-oscillatory system, using the same 64-fit/64-held-out protocol as the paper.</p>
    </div>
    <div className="freefallPairs">
      <VideoPair clips={[{src:"/videos/freefall/originals/eval_50073_high__natural_conflict.mp4",label:"Natural · high gravity"},{src:"/videos/freefall/originals/eval_50073_high__fit_only_controller.mp4",label:"Controller · high gravity"}]} caption="Held-out eval_50073_high. Same input and noise; only the internal condition-state edit changes."/>
      <VideoPair clips={[{src:"/videos/freefall/originals/eval_50281_low__natural_conflict.mp4",label:"Natural · low gravity"},{src:"/videos/freefall/originals/eval_50281_low__fit_only_controller.mp4",label:"Controller · low gravity"}]} caption="Held-out eval_50281_low. Original-resolution panel-kit exports from the final train-only experiment."/>
    </div>
    <div className="interactiveSection"><PCAView task="Free Fall" raw="/interactives/freefall_raw_projection_interactive.html" difference="/interactives/freefall_matched_difference_interactive.html"/>
      <p className="interactiveNote">B1, hist32, 100K. A basis fitted on 64 pairs projects 128 endpoints from the other 64 pairs; each input-color/gravity group contains 32 points. Color is decoded gravity, not frequency. Displaying PC3 does not change the rank-2 controller.</p>
    </div>
    <details className="pendulumDetails"><summary>Technical details</summary><p className="interactiveNote">The same frozen pairs, 32 observed frames, and all 20 flow-matching calls are used. Endpoint projections reproduce the archived difference coordinates. The paper's gravity-error success rate is 61/64 for the controller and 0/64 for natural conflicts; this is distinct from normalized recovery.</p></details>
  </section>;
}
