# Processed Data Files

All files in `data/main/` are processed summaries. They do not contain raw
model responses, private provider logs, API keys, or full task trajectories.

The companion `data/examples/` directory contains the selected-case manifest
and a small set of sanitized trajectory summaries. These files are for audit
and interpretation; quantitative claims should be recomputed from `data/main/`.

## Missing-Parameter Slice

- `missing_parameter_discovery.csv`:
  single-run discovery screen over targeted missing-parameter baseline
  failures.
- `missing_parameter_arm_success_x5.csv`:
  aggregate repeated successes over 11 selected full-repaired cases.
- `missing_parameter_case_counts_x5.csv`:
  case-arm matrix with five repeats per selected case-arm pair.
- `missing_parameter_case_counts_x10.csv`:
  ten-repeat companion used for repeat-budget checks.
- `missing_parameter_pairwise_vs_full_x5.csv`:
  pairwise discordance summaries against the full state-aware arm.
- `missing_parameter_residual_failures.csv`:
  coarse residual failure counts for failed full-state-aware repeats.
- `single_run_counterfactual_summary.csv`:
  Monte Carlo summary of one-shot ablation conclusions sampled from the
  repeated matrix.

## Missing-Function Slice

- `missing_function_aggregate.csv`:
  selected-set and all-baseline-failure calibration counts.
- `missing_function_case_counts_x5.csv`:
  case-arm matrix with five repeats per selected repair candidate.
- `missing_function_composition.csv`:
  component-composition follow-up counts.
- `missing_function_prompt_sensitivity.csv`:
  paraphrase-sensitivity follow-up counts.

## Strong-Model Calibration

- `strong_model_discovery_summary.csv`:
  discovery counts from Qwen3.7-Plus, Doubao Seed 2.1 Pro, and DeepSeek V4 Pro
  after reselecting failures from those models' own baseline runs.
- `strong_model_missing_parameter_stability.csv`:
  selected missing-parameter case-arm counts with five repeats per arm. These
  cases are strong-model baseline failures repaired by the full state-aware arm
  in discovery.
- `strong_model_missing_function_stability.csv`:
  selected missing-function case-arm counts with five repeats per arm. These
  cases are strong-model baseline failures repaired by at least one non-oracle
  arm in discovery.
- `strong_model_missing_parameter_all_failure_x3.csv`:
  three-repeat calibration over all strong-model targeted missing-parameter
  baseline failures, restricted to key arms.
- `strong_model_missing_function_all_failure_x3.csv`:
  three-repeat calibration over all strong-model missing-function baseline
  failures, restricted to key arms.

## External tau2 Checks

- `tau2_stress_discovery.csv`:
  co-equal external tau2 discovery matrix over three domains, twenty tasks per
  domain, three API models, and three arms.
- `tau2_stress_selected_stability.csv`:
  selected reruns for tau2 standard-arm failures repaired by at least one
  scaffold arm in discovery. Some cells are partial because long dialogues
  timed out before five repeats.

- `tau2_anchor.csv`:
  earlier aggregate stability counts over airline, retail, and telecom domains.
- `tau2_scaffold_ablation_x5.csv`:
  earlier scaffold-ablation case matrix over standard tau2 LLM agent,
  policy-reminder prompt, and two-phase ReAct arms with five repeats per
  case-arm pair.
- `tau2_failure_taxonomy.csv`:
  coarse automatic failure taxonomy for failed tau2 anchor records.
