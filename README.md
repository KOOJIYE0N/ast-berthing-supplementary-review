# Supplementary Data for AESCTE-D-26-04219

## Current evaluation package

The current revision records are in [R2_Analysis](R2_Analysis/). The same four files are available in [R2_Analysis.zip](R2_Analysis.zip).

- [Phase 1 outcome records](R2_Analysis/phase1_720_outcomes.csv): 720 outcomes from the same 24 scenes across three training seeds, five camera conditions, and two policies. Scene blocks, rather than individual model outcomes, are the independent units for the reported inference.
- [Phase 2 paired summaries](R2_Analysis/phase2_50_pair_summary.csv): 50 paired trials from the separate exploratory diagnostic with a 6,000-policy-step horizon. RGB-only and depth-fused crossings are 12/50 and 23/50, respectively.
- [Integrated analysis summaries](R2_Analysis/r2_integrated_analysis.json): reported aggregate counts, intervals, statistical tests, and trajectory diagnostics.
- [Package README](R2_Analysis/README.txt): provenance, statistical definitions, and the scope of the supplied records.

The current package matches the accompanying submission files byte for byte. Earlier records below are preserved for traceability and are not pooled with the current evaluation. The older scripts below apply to the earlier archive; they do not compute the current common-scene statistics or the 50-pair diagnostics.

## Earlier evaluation archive

The remainder of this README describes the earlier archive and its original evaluation design.

This archive supports reanalysis of the controlled evaluation and the saved depth diagnostic. It does not contain a complete training or simulator-reproduction package. Initial-protocol aggregate checks are separated from controlled outcomes and are not primary statistical evidence.

## Contents

- supplementary/phase1_outcomes_474.csv: 240 primary and 186 architectural/perception-control rollouts, plus 48 reference-extension rollouts on six added seed-0 scenes (stage=primary_ext). Reference control rows reuse primary and reference-extension trials.
- supplementary/physical/phase1_rgb_depth_trials_240.csv: whole-episode mechanical summaries, including 30 failed attempts.
- supplementary/physical/phase1_trial_records_full.json: contact/follow-up records and retained mechanical quantities for the same 240 trials.
- supplementary/physical/phase2_rgb_depth_trials_64.csv: 60 controlled trials (initial ten pairs plus a post hoc extension to 30 pairs) and four separate initial-checkpoint diagnostics.
- supplementary/physical/phase2_per_joint_metrics_320rows.csv: five-joint summaries for each of those 64 trials.
- supplementary/depth_no_msaa/: all 160 unchanged saved NPZ files, fixed calibration, collection plan, and reported metrics.
- metadata/: retained reference statistics, depth hash/size records, Phase 2 follow-up records, and limited diagnostic reports. These references are checks, not additional trials.
- reproduce/: portable summary/array reanalysis and panel-generation scripts.
- calculation_sources/: retained acquisition-code references for the physics-step work and peak definitions, with private paths removed. Their simulator and policy dependencies are not supplied; these files are not standalone rollout executables.

## Reanalysis

Use Python 3.12 and the packages in requirements.txt. From this directory, run:

~~~sh
python reproduce/analyse_outcomes.py
python reproduce/aggregate_mechanical.py
python reproduce/verify_depth.py
python reproduce/render_depth_panels.py
~~~

Outputs are written to analysis/. No command above trains a policy, accesses a server, runs MuJoCo, or creates new experimental observations. Panel generation uses the saved arrays. Library versions may affect PNG bytes without changing the source arrays or specified display.

## Definitions and Scope

Phase 1 has 24 paired scene clusters reused under five conditions. Each training seed uses a different eight-scene group; totals combine training-seed and scene-group variation and are not 120 independent scenes. Exact McNemar tests are Holm-adjusted across the five primary conditions. The Phase 2 comparison uses one training seed and fixed geometry; the 20 added pairs followed inspection of the initial ten, and the pooled 30-pair test is exploratory and unadjusted.

Phase 1 success means marker contact. Phase 2 success means reference-point proximity below 0.10 m. The count with no recorded post-success loss is the complement of any recorded loss in the follow-up interval, not secure capture or a final-state retention criterion. Success-conditioned policy subsets do not constitute a paired safety comparison.

Mechanical work uses stored physics-step accumulators over whole observed episodes, including follow-up. Generalized actuator torque, not control commands, is multiplied by joint velocity. Phase 1 peak is a simultaneous five-joint absolute-power sum peak; Phase 2 retains individual-joint peaks. Complete high-frequency traces and work snapshots at first success are unavailable; electrical input energy cannot be inferred from these records.

Depth arrays comprise 16 calibration views and 144 test views under nominal, lighting/material-change, and synthetic-occlusion conditions. Head-view target accuracy is unavailable. Synthetic occlusion replaces RGB pixels, not physical geometry. The separate lighting-only and boundary checks do not establish strong-reflection robustness or validate the simulator-depth policy as an upper bound.

## Anonymization and Integrity

Server addresses, account names, absolute paths, execution queues, and broad internal inventories are excluded. Host codes H1/H2/H3 preserve equality checks but do not identify machines. Record codes are consistent pseudonyms. Values beginning with withheld-source- are provenance identifiers, not paths to supplied files. Actual depth files use relative paths within this archive.

Experimental measurements, outcomes, seeds, and scientific hashes are preserved. SHA256SUMS.json covers the published files. Dataset and unavailable-checkpoint hashes are retained identifiers; the underlying demonstration images and model weights are not included or independently rehashed here.

The package supports outcome, summary, and diagnostic-image checks only. Complete demonstration data, trained policies, simulator assets, and raw physics-rate trajectories would be required for standalone rollout reproduction.
