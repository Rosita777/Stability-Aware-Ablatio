# Processed Data Files

All files in `data/main/` are processed summaries. They do not contain raw
model responses, private provider logs, API keys, or full task trajectories.

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

## External Sanity Check

- `tau2_anchor.csv`:
  aggregate stability counts over airline, retail, and telecom domains.
- `tau2_failure_taxonomy.csv`:
  coarse automatic failure taxonomy for failed tau2 records.
