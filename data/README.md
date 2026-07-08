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

- `tau2_external_discovery.csv`:
  external tau2 discovery matrix over three domains, twenty tasks per domain,
  three API models, and four arms: standard, policy reminder, progress guard,
  and full progress-aware scaffold.
- `tau2_external_selected_stability.csv`:
  selected reruns for one-shot standard-arm failures repaired by the full
  progress-aware scaffold. All selected case-arm cells have five repeats.
- `tau2_external_baseflip_control.csv`:
  five-repeat standard-arm reruns for all one-shot standard failures, used to
  estimate how often a selected standard failure flips under repetition.
- `tau2_external_regression_control.csv`:
  five-repeat standard and full-scaffold reruns for one-shot rows where
  standard succeeds and the full scaffold fails.
