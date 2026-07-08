# Sanitized Examples and Selection Manifest

This directory provides lightweight audit material for the processed counts in
`data/main/`. It does not contain raw provider logs, raw model responses, API
metadata, private prompts, or full benchmark trajectories.

- `selected_case_manifest.csv` lists every selected case used by the main
  stability readouts. Each row records the slice, anonymized case id, model,
  public task id, selection rule, case-arm success counts, attribution flags,
  and the processed source table.
- `sanitized_trace_examples.md` gives representative trajectory summaries.
  These examples are paraphrased from the diagnostic notes used in the paper.
  They are intended to help readers interpret the count tables, not to create
  additional quantitative claims.

The manifest is the right starting point for checking how the paper moves from
selected cases to case-level attribution flags. The examples are deliberately
small and qualitative because the official quantitative claims are already
encoded in the processed CSV files.
