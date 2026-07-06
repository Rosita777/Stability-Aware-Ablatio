# Stability-Aware Ablation Artifact

This anonymous artifact accompanies the paper on stability-aware ablation for
tool-using agent evaluation. It is designed for readers who want to inspect and
reproduce the main diagnostic results without access to the authors' private
API logs.

## What Is Included

The release contains the minimal materials needed to reproduce the main paper
readouts:

- `data/main/`: processed case-arm-repeat count tables and aggregate summaries.
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
missing-function transfer readout, and tau2 sanity check.

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
- `outputs/tables/tau2_anchor_table.csv`
- `outputs/figures/missing_parameter_case_matrix.svg`
- `outputs/figures/missing_function_case_matrix.svg`
- `outputs/figures/missing_function_followups.svg`

The script also checks the headline values used in the paper:

- missing-parameter simpler-arm match or beat: `7/11`;
- missing-parameter full beats every simpler arm: `4/11`;
- missing-function availability prompt: `41/60`;
- missing-function original full state-aware scaffold: `0/60`;
- tau2 partial task-model pairs: `24/48`.

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

The tau2 anchor is included only as an external sanity check. The release keeps
its aggregate stability table and coarse failure taxonomy, not raw simulated
dialogues.

## Double-Blind Notes

This artifact intentionally omits author names, private API keys, local
absolute paths, raw logs, and internal consultation files. If you upload it as a
submission artifact, upload the contents of this directory as the repository
root.
