# Stability-Aware Ablation Artifact

This anonymous artifact accompanies the paper on stability-aware ablation for
tool-using agent evaluation. It is designed for readers who want to inspect and
reproduce the main diagnostic results without access to the authors' private
API logs.

## What Is Included

The release contains the minimal materials needed to reproduce the main paper
readouts:

- `data/main/`: processed case-arm-repeat count tables and aggregate summaries.
- `data/examples/`: selected-case manifest and sanitized trajectory summaries
  for auditing the main case-level readouts.
- `src/reproduce_main_results.py`: offline reproduction script for main tables
  and figures.
- `src/bfcl_subset_runner.py`: optional OpenAI-compatible BFCL V4 subset runner
  for users who want to rerun API experiments.
- `scripts/reproduce.sh`: one-command offline reproduction.

The default reproduction path does not require GPUs or API calls.

## What Is Not Included

To keep the artifact compact and blind-review safe, this package does not
include raw provider logs, raw model trajectories, API responses, private API
configuration, internal exploratory prompts, or runs not used in the main
diagnostic claims.

The released data are processed records: model family, task id, scaffold arm,
and repeat success counts. They are sufficient to recompute the headline
case-level matrices, attribution flags, single-run counterfactual summary,
missing-function transfer readout, strong-model calibration, and tau2 external
stress test.

The examples directory adds human-readable audit context. It identifies the
selected cases used by the stability readouts and gives a small set of
paraphrased trajectory summaries, without releasing raw provider responses.

## Environment

Offline reproduction requires only Python 3.10 or newer and the standard
library.

```bash
python3 --version
```

No GPU is needed. API access is needed only for optional live reruns.

## Offline Reproduction

From this directory:

```bash
bash scripts/reproduce.sh
```

or directly:

```bash
python3 src/reproduce_main_results.py \
  --data-dir data/main \
  --output-dir outputs
```

Expected outputs:

- `outputs/summary.md`
- `outputs/tables/missing_parameter_stability_table.csv`
- `outputs/tables/missing_function_aggregate_table.csv`
- `outputs/tables/strong_model_discovery_summary.csv`
- `outputs/tables/strong_model_missing_parameter_stability.csv`
- `outputs/tables/strong_model_missing_function_stability.csv`
- `outputs/tables/strong_model_missing_parameter_all_failure_x3.csv`
- `outputs/tables/strong_model_missing_function_all_failure_x3.csv`
- `outputs/tables/tau2_anchor_table.csv`
- `outputs/tables/tau2_scaffold_ablation_x5_table.csv`
- `outputs/tables/tau2_stress_discovery_table.csv`
- `outputs/tables/tau2_stress_selected_stability_table.csv`
- `outputs/figures/missing_parameter_case_matrix.svg`
- `outputs/figures/missing_function_case_matrix.svg`
- `outputs/figures/missing_function_followups.svg`

The script also checks the headline values used in the paper:

- missing-parameter simpler-arm match or beat: `7/11`;
- missing-parameter full beats every simpler arm: `4/11`;
- missing-function availability prompt: `41/60`;
- missing-function original full state-aware scaffold: `0/60`;
- strong-model missing-parameter selected stability: full `28/35`, missing
  guard `28/35`, validator guard `28/32`;
- strong-model missing-parameter simpler-arm match or beat: `6/7`;
- strong-model missing-parameter strict full-only repairs: `0/7`;
- strong-model missing-function selected stability: availability `15/30`,
  full state-aware `4/30`;
- strong-model missing-function availability-or-composed arm matches/beats
  full: `5/6`;
- strong-model missing-parameter all-failure calibration: missing guard
  `30/122`, validator guard `27/122`, full `22/121`, simpler match/beat
  `40/41`;
- strong-model missing-function all-failure calibration: availability
  `10/203`, full `3/218`, availability+full `10/223`, availability-or-composed
  match/beat `83/83`;
- tau2 partial task-model pairs: `24/48`.
- tau2 scaffold-ablation simpler-arm match or beat: `9/10`.
- tau2 external stress-test discovery: standard `19/178`, policy `18/178`,
  ReAct `6/178`;
- tau2 external stress-test selected rerun: any scaffold beats standard on
  `1/7` cases, while standard matches or beats the best scaffold on `6/7`.

## Optional Live BFCL Reruns

The live runner is optional. It is provided to document how the BFCL subset
experiments were run, but exact API reproduction may vary with provider model
versions, routing, rate limits, and stochasticity.

1. Clone the public Berkeley Function Calling Leaderboard repository into
   `third_party/`:

```bash
mkdir -p third_party
git clone https://github.com/ShishirPatil/gorilla.git third_party/gorilla
```

2. Point the runner at the BFCL data directory. Depending on the upstream
   repository layout, this is usually:

```bash
export BFCL_ROOT="$PWD/third_party/gorilla/berkeley-function-call-leaderboard"
export BFCL_DATA_DIR="$BFCL_ROOT/bfcl_eval/data"
```

3. Configure an OpenAI-compatible endpoint:

```bash
cp .env.example .env
# edit .env, then:
set -a
source .env
set +a
```

4. Run a small smoke test:

```bash
python3 src/bfcl_subset_runner.py \
  --categories multi_turn_miss_param \
  --limit-per-category 1 \
  --models "$MODEL_LIST" \
  --intervention-mode model \
  --repeat-count 1 \
  --concurrency 1 \
  --output outputs/live_smoke.jsonl \
  --overwrite
```

5. Run a selected arm by changing `--intervention-mode`. The main paper uses
   arms such as:

- `model`
- `parameter_only_validator`
- `state_aware_prompt`
- `heuristic_missing_param_guard`
- `argument_validator_repair`
- `validator_heuristic_guard`
- `state_aware_validator_heuristic`
- `function_availability_prompt`
- `oracle_empty_turn_guard`

For the paper's full selected-case stability passes, run each arm over the
same `MODEL::TASK_ID` pairs with the same repeat budget, then aggregate the
resulting JSONL files into case-arm success counts.

## Data Preparation Notes

The paper's default experiments use public BFCL V4 multi-turn tasks. The
processed tables in `data/main/` are derived from these public tasks plus API
model outputs. Because provider outputs can include long trajectories and
private routing metadata, this artifact releases processed counts rather than
raw trajectories.

For case-level audit, `data/examples/selected_case_manifest.csv` records the
selection rule, case-arm counts, flags, and source table for each selected case.
`data/examples/sanitized_trace_examples.md` gives representative paraphrased
examples that explain how several rows map to tool-use behavior.

The tau2 release includes processed tables for the external stress test:
complete discovery triples and selected scaffold-repair reruns. It also keeps
the earlier aggregate stability anchor and coarse failure taxonomy for audit
context, but not raw simulated dialogues.

## Double-Blind Notes

This artifact intentionally omits author names, private API keys, local
absolute paths, raw logs, and internal consultation files. If you upload it as a
submission artifact, upload the contents of this directory as the repository
root.
