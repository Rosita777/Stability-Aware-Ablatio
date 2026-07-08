# Sanitized Trace Examples

These examples explain how the processed case-arm counts map to concrete
tool-use behavior. They are paraphrased and shortened. They do not include raw
provider responses, private routing metadata, full prompts, or full benchmark
trajectories.

## MP-11: Stable Full-Only Missing-Parameter Repair

- Source row: `selected_case_manifest.csv`, `case_id=MP-11`.
- Count pattern: baseline and all simpler arms are `0/5`; the full
  state-aware arm is `5/5`.
- Interpretation: this is the cleanest support for a full-scaffold mechanism
  claim in the missing-parameter slice.
- Sanitized trajectory summary: the user asks for a search using a placeholder
  such as a "specific word" before providing the word. The baseline treats the
  placeholder as the argument and calls the search tool prematurely. The full
  state-aware scaffold asks for clarification and waits until the missing word
  is supplied.

## MP-02: Simple Guard Sufficiency

- Source row: `selected_case_manifest.csv`, `case_id=MP-02`.
- Count pattern: missing guard, validator guard, and full state-aware arm are
  all `5/5`.
- Interpretation: the full scaffold works, but the repeated matched readout
  does not support a full-specific mechanism claim.
- Sanitized trajectory summary: the user asks for the "last several lines" of
  a file without specifying a line count. A narrow missing-parameter guard is
  sufficient to ask for the missing count before calling the file-inspection
  tool. The full scaffold behaves similarly, but it is not needed for this
  local repair.

## MP-10: Single-Run Attribution Inversion

- Source row: `selected_case_manifest.csv`, `case_id=MP-10`.
- Count pattern: the case was selected because full repaired it in discovery,
  but in the repeated readout full is `0/5`, state prompt is `3/5`, and
  missing guard is `2/5`.
- Interpretation: a single successful full-scaffold trajectory would have
  produced the wrong mechanism story.
- Sanitized trajectory summary: the task contains an underspecified support
  ticket turn. The unstable full arm often calls a resolution function before
  the ticket id is provided. Simpler prompt or guard variants sometimes defer
  the call and ask for the missing id.

## MF-04: Availability Beats Original Full Scaffold

- Source row: `selected_case_manifest.csv`, `case_id=MF-04`.
- Count pattern: availability prompt and oracle-style timing succeed `5/5`;
  the original full state-aware arm is `0/5`.
- Interpretation: the missing-parameter full validator does not transfer to a
  neighboring missing-function failure mode.
- Sanitized trajectory summary: the relevant function is absent from the early
  tool list and appears later. A successful availability-aware run emits no
  substitute tool call, waits for the tool-list update, and then calls the
  newly available function. The original full scaffold tends to explore nearby
  tools before the hidden function is available.

## SMF-04: Strong-Model Mechanism Transfer Check

- Source row: `selected_case_manifest.csv`, `case_id=SMF-04`.
- Count pattern: for a stronger model, availability is `4/5`, the original
  full state-aware arm is `0/5`, and the oracle guard is `5/5`.
- Interpretation: stronger base models do not remove the need for
  failure-mode-specific attribution checks.
- Sanitized trajectory summary: the useful behavior is still a timing policy:
  wait until the required function becomes available rather than substituting a
  nearby tool. Larger or stronger models can still make premature calls when
  the scaffold pressures them toward immediate tool use.
