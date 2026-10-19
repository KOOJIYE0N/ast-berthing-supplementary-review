AESCTE-D-26-04219 accompanying analysis package

phase1_720_outcomes.csv
720 outcome records: 24 scenes x 3 training seeds x 5 conditions x 2 policies.
The origin field preserves its existing provenance labels: R1_verified denotes
240 earlier outcomes and R2_new denotes 480 additional outcomes. All 240 earlier
records are retained without outcome-based exclusion. The three earlier
eight-scene groups form the same 24-scene set used in the completed cross-design.
The independent analysis block is the scene, not each model/condition record.

phase2_50_pair_summary.csv
50 paired summaries from a separate 6000-policy-step diagnostic at 20 Hz.
First 20 pairs and added 30 pairs were evaluated sequentially after inspection.
These records are not pooled with the earlier 30-pair, 20000-step cohort.
The actual_host field identifies GPU hardware, not a network/server address.

r2_integrated_analysis.json
Aggregate counts, intervals, scene-block tests, and trajectory diagnostics.
This preserves the supplied analysis result, including its scope limitations.
The top-level COMPLETE status refers to that analysis, not journal readiness.

Scope
The CSV files support outcome-count and paired-summary reanalysis. The JSON
contains aggregate closest-point and retention diagnostics; their full raw
trajectories and event-extraction inputs are not included in this small archive.
No new simulator rollouts, interventions, or physics-rate work integrals are
implied by this packaging. Depth arrays, calibration source data, analysis or
plotting scripts, and detailed Phase 2 diagnostic inputs are not included in
this submission. The Supplementary Material PDF reports methods, diagnostic
figures, and historical summary tables, not the underlying raw arrays or code.
The host fields retain the original hardware labels for analysis provenance.
Host assignment was not randomized; these labels do not support a causal
hardware-performance comparison. CSV and JSON values are unchanged.
See Supplementary Material Sections F.1-F.4 for the statistical definitions.
Appendix G retains earlier-cohort summaries, separate from the primary results.
