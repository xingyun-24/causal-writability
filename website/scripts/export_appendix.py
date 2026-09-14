"""Export the final paper's complete appendix figure catalogue as vectors."""
import argparse
import json
from pathlib import Path
import re

from export_vector_panels import export

# Labels resolve figure and page numbers from the compiled final manuscript.
ITEMS = [
    ("Behavior", "slow-matched-example", "generated/appendix_figure2_slow_matched_example", "Matched cue reversal", "Changing appearance redirects both motion and generated color under fixed slow history."),
    ("Behavior", "multiseed-behavior", "generated/appendix_figure2_multiseed_behavior", "Behavior across training seeds", "The cue-dependent allocation of generated motion recurs across 15 independently trained models."),
    ("Pretrained replication", "pretrained-wan-route-boundary", "generated/appendix_pretrained_wan_replication", "Executable routes in pretrained Wan", "Held-out phase geometry, controller recovery and the matched-identity boundary comparison."),
    ("Pretrained replication", "pretrained-behavior-appendix", "generated/appendix_pretrained_behavior_v1", "Neutral-first adaptation", "The achromatic starting solution and subsequent biased adaptation on the frozen held-out population."),
    ("Pretrained replication", "pretrained-causal-timing-appendix", "generated/appendix_pretrained_causal_timing_v1", "Failure age, later fate and reopening", "Early writability predicts near-term departure from the shortcut mode; fixed identities can close and reopen."),
    ("Controller controls", "boundary-phase-controller-ablation", "audits/figure3_state_model_controls", "Direction and boundary phase", "Within the tested harmonic family, target-specific phase is needed to predict the state-varying coordinates."),
    ("Controller controls", "state-parameterizations", "audits/figure3_state_parameterizations", "Two information contracts", "Input-boundary control and after-conflict state differences are distinguished by which information they read."),
    ("Controller controls", "short50k-fullmean-controls", "generated/appendix_short50k_controller_comparison", "Full-dimensional mean controls", "State-conditioned Top-4 controllers are compared with constant full-dimensional means and receiver-specific oracle edits."),
    ("Cross-system replication", "pendulum-behavior", "pendulum/pendulum_behavior", "Pendulum behavior", "Cue sweeps and Short/Long motion-following comparisons."),
    ("Cross-system replication", "pendulum-state-geometry", "pendulum/pendulum_state_geometry", "Pendulum edit geometry", "Physical state organizes held-out edit coordinates in the fit-only space."),
    ("Cross-system replication", "pendulum-decoded-recovery", "pendulum/pendulum_decoded_recovery", "Pendulum decoded recovery", "Full, Top-4 and predicted edits are evaluated on decoded held-out futures."),
    ("Cross-system replication", "pendulum-writability", "pendulum/pendulum_writeability", "Pendulum layer scans", "Longer visible history delays closure in both rewrite directions."),
    ("Cross-system replication", "freefall-behavior", "freefall/freefall_behavior", "Free Fall behavior", "Decoded gravity follows the observed-motion family across appearance cues."),
    ("Cross-system replication", "freefall-geometry", "freefall/freefall_state_geometry", "Free Fall edit geometry", "A direction-specific target-gravity law predicts coordinates fitted on the separate training split."),
    ("Cross-system replication", "freefall-decoded-recovery", "freefall/freefall_decoded_recovery", "Free Fall decoded recovery", "Full-edit, Top-2 and donor-free controller outcomes use the same 64 held-out receivers."),
    ("Cross-system replication", "freefall-writability", "freefall/freefall_writability", "Free Fall layer scans", "Gravity-error success across intervention blocks on the fixed 128-pair cohort."),
    ("Training and writability", "persistent-rigidity", "generated/appendix_figure4_persistent_rigidity", "Fixed persistent-error identities", "The composition control tracks the same unresolved errors during training."),
    ("Training and writability", "writability-robustness", "audits/figure4_writeability_robustness", "Alternative definitions and controls", "Early writability and closure are tested under alternative scores, error severity and boundary-state controls."),
    ("Training and writability", "writability-cross-solution", "audits/figure4_direction_decomposition", "Same-checkpoint behavioral association", "Across-run association is separated from training step and decomposed by target direction."),
    ("Training and writability", "fate-target-write", "audits/figure4_5_fate_target_write_bridge", "Early fate and downstream writing", "Future-rescued errors already exhibit stronger route-aligned target responses at the early checkpoint."),
    ("Transfer and downstream mechanisms", "cross-run-transfer-controls", "audits/figure5_cross_run_transfer_controls", "Cross-run transfer controls", "Held-out geometry alignment is compared with shuffles and separate decoded interventions."),
    ("Transfer and downstream mechanisms", "cross-step-exact100", "generated/appendix_figure5_cross_step_exact100k", "Cross-checkpoint compatibility", "All 90 off-diagonal maps test matched-coordinate compatibility, not donor-free controller transfer."),
    ("Transfer and downstream mechanisms", "implementation-multiplicity", "generated/appendix_figure5_implementation_multiplicity", "Different K/V implementations", "Shared physical corrections coexist with different downstream bottlenecks across trained models."),
    ("Transfer and downstream mechanisms", "single-head-specificity", "audits/figure5_target_write_specificity", "Write specificity", "Route-aligned response is distinguished from generic activation growth."),
    ("Transfer and downstream mechanisms", "fm-time-decoded-rescue", "generated/appendix_figure5_fm_and_decoded", "Flow-matching time and decoded rescue", "Early and late call allocations distinguish an internally visible write from a behaviorally effective one."),
    ("Training and writability", "spring-short-long-layer-scans", "generated/appendix_spring_short_long_layer_scans", "Short/Long layer scans", "Complete checkpoint-specific strict banks for three Spring runs, without claiming trajectory-paired Short/Long cohorts."),
    ("Transfer and downstream mechanisms", "short-attention-gain-sweeps", "generated/appendix_short_attention_gain_sweeps", "K/V dose responses", "Different operators and gain scales reveal solution-specific responses, not an equal-dose component ranking."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    aux = (args.paper / "main.aux").read_text()
    labels = {label: (int(number), int(page)) for label, number, page in
              re.findall(r"\\newlabel\{fig:([^}]+)\}\{\{(\d+)\}\{(\d+)\}", aux)}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, label, relative, title, note in ITEMS:
        number, page = labels[label]
        filename = f"figure-{number:02d}.svg"
        details = export(args.paper / "figures" / (relative + ".pdf"), args.out / filename)
        rows.append({"group": group, "label": label, "number": number, "page": page,
                     "title": title, "note": note, "src": "/paper/appendix/" + filename,
                     "vector_paths": details["vector_paths"], "source": "figures/" + relative + ".pdf"})
    assert sorted(r["number"] for r in rows) == list(range(7, 34))
    (args.out / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Exported all 27 final-manuscript appendix figures, numbered 7 through 33.")


if __name__ == "__main__":
    main()
