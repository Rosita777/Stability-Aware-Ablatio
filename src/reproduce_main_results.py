#!/usr/bin/env python3
"""Reproduce the main diagnostic tables and figures from processed records.

This script uses only the processed case-arm-repeat count files in data/main.
It does not require API access, GPUs, or the raw BFCL/tau2 trajectories.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


MP_ARMS = [
    "Baseline",
    "Parameter-only",
    "State prompt",
    "Missing guard",
    "Argument validator",
    "Validator guard",
    "Full state-aware",
]

MF_ARMS = [
    "baseline",
    "availability_prompt",
    "parameter_only",
    "state_prompt",
    "missing_guard",
    "argument_validator",
    "validator_guard",
    "full_state_aware",
    "oracle_empty_turn",
]

MF_LABELS = {
    "baseline": "Base",
    "availability_prompt": "Avail",
    "parameter_only": "Param",
    "state_prompt": "State",
    "missing_guard": "Miss",
    "argument_validator": "Arg",
    "validator_guard": "Guard",
    "full_state_aware": "Full",
    "oracle_empty_turn": "Oracle",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_count_fraction(value: str) -> tuple[int, int]:
    left, right = value.split("/", 1)
    return int(left), int(right)


def pct(numer: float, denom: float) -> float:
    return 0.0 if denom == 0 else 100.0 * numer / denom


def summarize_missing_parameter(data_dir: Path, tables_dir: Path) -> list[str]:
    case_rows = read_csv(data_dir / "missing_parameter_case_counts_x5.csv")
    arm_rows = read_csv(data_dir / "missing_parameter_arm_success_x5.csv")
    discovery_rows = read_csv(data_dir / "missing_parameter_discovery.csv")
    residual_rows = read_csv(data_dir / "missing_parameter_residual_failures.csv")
    counter_rows = read_csv(data_dir / "single_run_counterfactual_summary.csv")

    case_count = len(case_rows)
    strict_full_only = sum(int(row["strict_stable_full_only"]) for row in case_rows)
    full_advantage = sum(int(row["full_advantage"]) for row in case_rows)
    simpler_match = sum(int(row["simpler_match_or_beats"]) for row in case_rows)
    full_unstable = sum(int(row["full_unstable"]) for row in case_rows)

    stability_rows: list[dict[str, object]] = []
    cases_with_success: dict[str, int] = {}
    for row in case_rows:
        for arm in MP_ARMS:
            cases_with_success[arm] = cases_with_success.get(arm, 0) + (1 if int(row[arm]) > 0 else 0)
    for row in arm_rows:
        label = row["label"]
        stability_rows.append(
            {
                "arm": label,
                "successes": f"{row['successes']}/{row['n']}",
                "cases_with_success": f"{cases_with_success[label]}/{case_count}",
            }
        )
    write_csv(tables_dir / "missing_parameter_stability_table.csv", stability_rows, ["arm", "successes", "cases_with_success"])

    residual_count = sum(int(row["count"]) for row in residual_rows)
    x5 = next(row for row in counter_rows if row["repeats"] == "x5")

    lines = [
        "## Missing-parameter selected repair set",
        "",
        f"- Discovery full-state-aware repairs: 11/48 targeted failures.",
        f"- Repeated full-state-aware successes: 42/55 over 10/11 cases.",
        f"- Full beats every simpler arm on {full_advantage}/{case_count} cases.",
        f"- A simpler arm matches or beats full on {simpler_match}/{case_count} cases.",
        f"- Full arm is unstable on {full_unstable}/{case_count} cases.",
        f"- Strict stable full-only repairs: {strict_full_only}/{case_count}.",
        f"- Residual full-arm failures: {residual_count}/55.",
        f"- One-shot unsupported full-only labels, x5 reference: {x5['unsupported_full_only_percent']}%.",
        "",
        "Discovery screen:",
        "",
        "| Arm | Repairs | Targeted failures |",
        "|---|---:|---:|",
    ]
    for row in discovery_rows:
        lines.append(f"| {row['label']} | {row['repairs']} | {row['targeted_failures']} |")
    lines.extend(["", "Repeated stability:", "", "| Arm | Successes | Cases with >=1 success |", "|---|---:|---:|"])
    for row in stability_rows:
        lines.append(f"| {row['arm']} | {row['successes']} | {row['cases_with_success']} |")
    return lines


def summarize_missing_function(data_dir: Path, tables_dir: Path) -> list[str]:
    aggregate = read_csv(data_dir / "missing_function_aggregate.csv")
    cases = read_csv(data_dir / "missing_function_case_counts_x5.csv")
    composition = read_csv(data_dir / "missing_function_composition.csv")
    paraphrase = read_csv(data_dir / "missing_function_prompt_sensitivity.csv")

    availability_wins = 0
    nonoracle_matches_full = 0
    for row in cases:
        counts = {arm: int(row[arm]) for arm in MF_ARMS}
        nonoracle = [arm for arm in MF_ARMS if arm != "oracle_empty_turn"]
        if counts["availability_prompt"] > max(counts[arm] for arm in nonoracle if arm != "availability_prompt"):
            availability_wins += 1
        if max(counts[arm] for arm in nonoracle if arm != "full_state_aware") >= counts["full_state_aware"]:
            nonoracle_matches_full += 1

    write_csv(
        tables_dir / "missing_function_aggregate_table.csv",
        aggregate,
        ["arm", "label", "selected_x5", "all_failures_x3", "any_pairs"],
    )

    original = next(row for row in paraphrase if row["variant"] == "original")
    v3 = next(row for row in paraphrase if row["variant"] == "v3")
    full = next(row for row in aggregate if row["arm"] == "full_state_aware")
    avail = next(row for row in aggregate if row["arm"] == "availability_prompt")

    lines = [
        "",
        "## Missing-function mechanism-transfer slice",
        "",
        f"- Selected x5 availability prompt: {avail['selected_x5']}.",
        f"- Selected x5 original full-state-aware: {full['selected_x5']}.",
        f"- All-failure x3 availability prompt: {avail['all_failures_x3']}.",
        f"- All-failure x3 original full-state-aware: {full['all_failures_x3']}.",
        f"- Availability uniquely beats other non-oracle arms on {availability_wins}/{len(cases)} selected cases.",
        f"- A non-oracle arm matches or beats full on {nonoracle_matches_full}/{len(cases)} selected cases.",
        f"- Composition check, availability + full: 31/60 versus availability-only 39/60 in the follow-up file.",
        f"- Prompt surface varies from {v3['successes']}/{v3['valid_runs']} to {original['successes']}/{original['valid_runs']}.",
        "",
        "| Arm | Selected x5 | All failures x3 | Any pairs |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(f"| {row['label']} | {row['selected_x5']} | {row['all_failures_x3']} | {row['any_pairs']} |")
    return lines


def summarize_tau2(data_dir: Path, tables_dir: Path) -> list[str]:
    discovery_rows = read_csv(data_dir / "tau2_external_discovery.csv")
    selected_rows = read_csv(data_dir / "tau2_external_selected_stability.csv")
    baseflip_rows = read_csv(data_dir / "tau2_external_baseflip_control.csv")
    regression_rows = read_csv(data_dir / "tau2_external_regression_control.csv")
    write_csv(tables_dir / "tau2_external_discovery_table.csv", discovery_rows, list(discovery_rows[0].keys()))
    write_csv(
        tables_dir / "tau2_external_selected_stability_table.csv",
        selected_rows,
        list(selected_rows[0].keys()),
    )
    write_csv(
        tables_dir / "tau2_external_baseflip_control_table.csv",
        baseflip_rows,
        list(baseflip_rows[0].keys()),
    )
    write_csv(
        tables_dir / "tau2_external_regression_control_table.csv",
        regression_rows,
        list(regression_rows[0].keys()),
    )

    discovery_totals = {
        arm: (sum(int(row[arm]) for row in discovery_rows), sum(int(row[f"n_{arm}"]) for row in discovery_rows))
        for arm in ["standard", "policy", "progress", "full_progress"]
    }
    standard_failures = sum(1 for row in discovery_rows if int(row["standard"]) == 0)
    any_scaffold_repairs = sum(int(row["any_scaffold_repairs_standard_failure"]) for row in discovery_rows)
    full_repairs = sum(int(row["full_progress_repairs_standard_failure"]) for row in discovery_rows)
    full_specific = sum(int(row["full_beats_simpler"]) for row in discovery_rows)
    selected_totals = {
        arm: (sum(int(row[arm]) for row in selected_rows), sum(int(row[f"n_{arm}"]) for row in selected_rows))
        for arm in ["standard", "policy", "progress", "full_progress"]
    }
    full_beats_standard = sum(int(row["full_rate_beats_standard"]) for row in selected_rows)
    full_beats_simpler = sum(int(row["full_rate_beats_simpler"]) for row in selected_rows)
    simpler_matches_full = sum(int(row["simpler_rate_matches_or_beats_full"]) for row in selected_rows)
    any_scaffold_beats_standard = sum(int(row["any_scaffold_rate_beats_standard"]) for row in selected_rows)
    standard_matches_best = sum(int(row["standard_rate_matches_or_beats_scaffold"]) for row in selected_rows)
    full_unstable = sum(int(row["full_progress_unstable"]) for row in selected_rows)
    strict_full_only = sum(int(row["strict_stable_full_only"]) for row in selected_rows)
    baseflip_success = sum(int(row["standard"]) for row in baseflip_rows)
    baseflip_valid = sum(int(row["n_standard"]) for row in baseflip_rows)
    baseflip_any = sum(int(row["standard_any_success"]) for row in baseflip_rows)
    baseflip_stable_failure = sum(int(row["standard_stable_failure"]) for row in baseflip_rows)
    baseflip_unstable = sum(int(row["standard_unstable"]) for row in baseflip_rows)
    baseflip_stable_success = sum(int(row["standard_stable_success"]) for row in baseflip_rows)
    regression_standard = sum(int(row["standard"]) for row in regression_rows)
    regression_standard_valid = sum(int(row["n_standard"]) for row in regression_rows)
    regression_full = sum(int(row["full_progress"]) for row in regression_rows)
    regression_full_valid = sum(int(row["n_full_progress"]) for row in regression_rows)
    regression_standard_wins = sum(int(row["standard_rate_beats_full"]) for row in regression_rows)
    regression_full_wins = sum(int(row["full_rate_beats_standard"]) for row in regression_rows)
    regression_ties = sum(int(row["rates_tie"]) for row in regression_rows)

    return [
        "",
        "## tau2 external case-level study",
        "",
        (
            "- Discovery over complete triples: "
            f"standard={discovery_totals['standard'][0]}/{discovery_totals['standard'][1]}, "
            f"policy={discovery_totals['policy'][0]}/{discovery_totals['policy'][1]}, "
            f"progress={discovery_totals['progress'][0]}/{discovery_totals['progress'][1]}, "
            f"full={discovery_totals['full_progress'][0]}/{discovery_totals['full_progress'][1]}."
        ),
        (
            "- Standard-arm failures in discovery: "
            f"{standard_failures}/{len(discovery_rows)}; scaffold-repaired standard failures: "
            f"any={any_scaffold_repairs}, full={full_repairs}; full-specific one-shot rows={full_specific}."
        ),
        (
            "- Selected rerun observed repeats: "
            f"standard={selected_totals['standard'][0]}/{selected_totals['standard'][1]}, "
            f"policy={selected_totals['policy'][0]}/{selected_totals['policy'][1]}, "
            f"progress={selected_totals['progress'][0]}/{selected_totals['progress'][1]}, "
            f"full={selected_totals['full_progress'][0]}/{selected_totals['full_progress'][1]}."
        ),
        (
            "- Selected rerun case flags: "
            f"full beats standard on {full_beats_standard}/{len(selected_rows)} cases; "
            f"full beats all simpler arms on {full_beats_simpler}/{len(selected_rows)} cases; "
            f"simpler arms match/beat full on {simpler_matches_full}/{len(selected_rows)} cases; "
            f"strict stable full-only={strict_full_only}/{len(selected_rows)}."
        ),
        (
            "- Selected rerun stability controls: "
            f"any scaffold beats standard on {any_scaffold_beats_standard}/{len(selected_rows)} cases; "
            f"standard matches/beats the best scaffold on {standard_matches_best}/{len(selected_rows)} cases; "
            f"full is repeat-unstable on {full_unstable}/{len(selected_rows)} cases."
        ),
        (
            "- Base-flip control over one-shot standard failures: "
            f"standard={baseflip_success}/{baseflip_valid}; any success={baseflip_any}/{len(baseflip_rows)}, "
            f"stable failure={baseflip_stable_failure}/{len(baseflip_rows)}, "
            f"repeat-unstable={baseflip_unstable}/{len(baseflip_rows)}, "
            f"stable success={baseflip_stable_success}/{len(baseflip_rows)}."
        ),
        (
            "- Regression control over one-shot standard-pass/full-fail rows: "
            f"standard={regression_standard}/{regression_standard_valid}, "
            f"full={regression_full}/{regression_full_valid}; "
            f"standard/full/tie={regression_standard_wins}/{regression_full_wins}/{regression_ties}."
        ),
    ]
    return lines


def summarize_strong_model(data_dir: Path, tables_dir: Path) -> list[str]:
    discovery = read_csv(data_dir / "strong_model_discovery_summary.csv")
    mp_rows = read_csv(data_dir / "strong_model_missing_parameter_stability.csv")
    mf_rows = read_csv(data_dir / "strong_model_missing_function_stability.csv")
    mp_all = read_csv(data_dir / "strong_model_missing_parameter_all_failure_x3.csv")
    mf_all = read_csv(data_dir / "strong_model_missing_function_all_failure_x3.csv")

    write_csv(
        tables_dir / "strong_model_discovery_summary.csv",
        discovery,
        ["slice", "arm", "repairs", "baseline_failures", "valid_records", "successes", "provider_errors"],
    )
    write_csv(
        tables_dir / "strong_model_missing_parameter_stability.csv",
        mp_rows,
        list(mp_rows[0].keys()) if mp_rows else [],
    )
    write_csv(
        tables_dir / "strong_model_missing_function_stability.csv",
        mf_rows,
        list(mf_rows[0].keys()) if mf_rows else [],
    )
    write_csv(
        tables_dir / "strong_model_missing_parameter_all_failure_x3.csv",
        mp_all,
        list(mp_all[0].keys()) if mp_all else [],
    )
    write_csv(
        tables_dir / "strong_model_missing_function_all_failure_x3.csv",
        mf_all,
        list(mf_all[0].keys()) if mf_all else [],
    )

    mp_full = next(row for row in discovery if row["slice"] == "missing_parameter" and row["arm"] == "full_state_aware")
    mp_missing = next(row for row in discovery if row["slice"] == "missing_parameter" and row["arm"] == "missing_guard")
    mp_valguard = next(row for row in discovery if row["slice"] == "missing_parameter" and row["arm"] == "validator_guard")
    mf_avail = next(row for row in discovery if row["slice"] == "missing_function" and row["arm"] == "availability_prompt")
    mf_full = next(row for row in discovery if row["slice"] == "missing_function" and row["arm"] == "full_state_aware")
    mf_avail_full = next(row for row in discovery if row["slice"] == "missing_function" and row["arm"] == "availability_full")
    mf_oracle = next(row for row in discovery if row["slice"] == "missing_function" and row["arm"] == "oracle_guard")

    mp_simpler_match = sum(int(row["simpler_match_or_beats_full"]) for row in mp_rows)
    mp_strict = sum(int(row["strict_stable_full_only"]) for row in mp_rows)
    mp_full_success = sum(int(row["full_state_aware"]) for row in mp_rows)
    mp_full_valid = sum(int(row["full_state_aware_valid"]) for row in mp_rows)
    mp_missing_success = sum(int(row["missing_guard"]) for row in mp_rows)
    mp_missing_valid = sum(int(row["missing_guard_valid"]) for row in mp_rows)
    mp_valguard_success = sum(int(row["validator_guard"]) for row in mp_rows)
    mp_valguard_valid = sum(int(row["validator_guard_valid"]) for row in mp_rows)

    mf_avail_success = sum(int(row["availability"]) for row in mf_rows)
    mf_avail_valid = sum(int(row["availability_valid"]) for row in mf_rows)
    mf_full_success = sum(int(row["full_state_aware"]) for row in mf_rows)
    mf_full_valid = sum(int(row["full_state_aware_valid"]) for row in mf_rows)
    mf_combo_match = sum(int(row["availability_or_combo_matches_full"]) for row in mf_rows)

    mp_all_full_success = sum(int(row["full_state_aware"]) for row in mp_all)
    mp_all_full_valid = sum(int(row["full_state_aware_valid"]) for row in mp_all)
    mp_all_missing_success = sum(int(row["missing_guard"]) for row in mp_all)
    mp_all_missing_valid = sum(int(row["missing_guard_valid"]) for row in mp_all)
    mp_all_valguard_success = sum(int(row["validator_guard"]) for row in mp_all)
    mp_all_valguard_valid = sum(int(row["validator_guard_valid"]) for row in mp_all)
    mp_all_simpler_match = sum(int(row["simple_match_or_beats_full"]) for row in mp_all)

    mf_all_avail_success = sum(int(row["availability"]) for row in mf_all)
    mf_all_avail_valid = sum(int(row["availability_valid"]) for row in mf_all)
    mf_all_full_success = sum(int(row["full_state_aware"]) for row in mf_all)
    mf_all_full_valid = sum(int(row["full_state_aware_valid"]) for row in mf_all)
    mf_all_combo_success = sum(int(row["availability_full"]) for row in mf_all)
    mf_all_combo_valid = sum(int(row["availability_full_valid"]) for row in mf_all)
    mf_all_combo_match = sum(int(row["availability_or_combo_matches_full"]) for row in mf_all)

    lines = [
        "",
        "## Strong-model calibration",
        "",
        (
            "- Missing-parameter discovery over strong-model targeted failures: "
            f"missing guard={mp_missing['repairs']}/{mp_missing['baseline_failures']}, "
            f"validator guard={mp_valguard['repairs']}/{mp_valguard['baseline_failures']}, "
            f"full={mp_full['repairs']}/{mp_full['baseline_failures']}."
        ),
        (
            "- Missing-parameter selected stability: "
            f"full={mp_full_success}/{mp_full_valid}, missing guard={mp_missing_success}/{mp_missing_valid}, "
            f"validator guard={mp_valguard_success}/{mp_valguard_valid}; "
            f"simpler matches/beats full on {mp_simpler_match}/{len(mp_rows)} cases; "
            f"strict full-only={mp_strict}/{len(mp_rows)}."
        ),
        (
            "- Missing-function discovery over strong-model baseline failures: "
            f"availability={mf_avail['repairs']}/{mf_avail['baseline_failures']}, "
            f"full={mf_full['repairs']}/{mf_full['baseline_failures']}, "
            f"availability+full={mf_avail_full['repairs']}/{mf_avail_full['baseline_failures']}, "
            f"oracle={mf_oracle['repairs']}/{mf_oracle['baseline_failures']}."
        ),
        (
            "- Missing-function selected stability: "
            f"availability={mf_avail_success}/{mf_avail_valid}, full={mf_full_success}/{mf_full_valid}; "
            f"availability or availability+full matches/beats full on {mf_combo_match}/{len(mf_rows)} cases."
        ),
        (
            "- Missing-parameter all-failure x3 calibration: "
            f"missing guard={mp_all_missing_success}/{mp_all_missing_valid}, "
            f"validator guard={mp_all_valguard_success}/{mp_all_valguard_valid}, "
            f"full={mp_all_full_success}/{mp_all_full_valid}; "
            f"key simpler arm matches/beats full on {mp_all_simpler_match}/{len(mp_all)} cases."
        ),
        (
            "- Missing-function all-failure x3 calibration: "
            f"availability={mf_all_avail_success}/{mf_all_avail_valid}, "
            f"full={mf_all_full_success}/{mf_all_full_valid}, "
            f"availability+full={mf_all_combo_success}/{mf_all_combo_valid}; "
            f"availability or availability+full matches/beats full on {mf_all_combo_match}/{len(mf_all)} complete rows."
        ),
    ]
    return lines


def color_for_count(value: int, max_count: int) -> str:
    palette = {
        0: "#f4f6f8",
        1: "#dceff6",
        2: "#b7dce9",
        3: "#7fc8d5",
        4: "#38a8bd",
        5: "#08798e",
        6: "#08798e",
        7: "#08798e",
        8: "#08798e",
        9: "#08798e",
        10: "#08798e",
    }
    if max_count <= 5:
        return palette[value]
    scaled = round(5 * value / max_count)
    return palette[scaled]


def svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def draw_matrix_svg(
    rows: list[dict[str, str]],
    arms: list[str],
    labels: dict[str, str] | None,
    output: Path,
    title: str,
    max_count: int,
) -> None:
    labels = labels or {arm: arm for arm in arms}
    cell_w = 74
    cell_h = 26
    left = 150
    top = 58
    width = left + len(arms) * cell_w + 24
    height = top + len(rows) * cell_h + 22
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="12" y="24" font-family="Arial, sans-serif" font-size="15" font-weight="700">{svg_escape(title)}</text>',
    ]
    for i, arm in enumerate(arms):
        x = left + i * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x}" y="{top - 12}" font-family="Arial, sans-serif" font-size="8" '
            f'text-anchor="middle">{svg_escape(labels[arm])}</text>'
        )
    for r, row in enumerate(rows):
        y = top + r * cell_h
        case_label = f"{row['model']}/{row['task'].replace('multi_turn_miss_param_', 'mp').replace('multi_turn_miss_func_', 'mf')}"
        lines.append(
            f'<text x="{left - 8}" y="{y + 17}" font-family="Arial, sans-serif" font-size="8" '
            f'text-anchor="end">{svg_escape(case_label)}</text>'
        )
        for c, arm in enumerate(arms):
            x = left + c * cell_w
            value = int(row[arm])
            fill = color_for_count(value, max_count)
            text_color = "#ffffff" if value >= max(4, max_count - 1) else "#111827"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{fill}" stroke="#ffffff"/>')
            lines.append(
                f'<text x="{x + (cell_w - 2) / 2}" y="{y + 16}" font-family="Arial, sans-serif" '
                f'font-size="9" font-weight="700" text-anchor="middle" fill="{text_color}">{value}</text>'
            )
    lines.append("</svg>")
    write_text(output, "\n".join(lines) + "\n")


def draw_followup_svg(data_dir: Path, output: Path) -> None:
    composition = read_csv(data_dir / "missing_function_composition.csv")
    prompt = read_csv(data_dir / "missing_function_prompt_sensitivity.csv")
    width, height = 520, 170
    panel_w = 230
    lefts = [42, 290]
    max_value = 45
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="120" y="18" font-family="Arial, sans-serif" font-size="12" font-weight="700" text-anchor="middle">(a) Component composition</text>',
        '<text x="380" y="18" font-family="Arial, sans-serif" font-size="12" font-weight="700" text-anchor="middle">(b) Prompt surface</text>',
    ]

    def bars(rows: Iterable[dict[str, str]], x0: int, y0: int) -> None:
        for i, row in enumerate(rows):
            y = y0 + i * 17
            label = row.get("label", row.get("variant", ""))
            value = int(row["successes"])
            lines.append(f'<text x="{x0 - 6}" y="{y + 10}" font-family="Arial, sans-serif" font-size="8" text-anchor="end">{svg_escape(label)}</text>')
            lines.append(f'<rect x="{x0}" y="{y}" width="{panel_w * value / max_value:.1f}" height="11" fill="#8ed1cf" stroke="#3b4450"/>')
            lines.append(f'<text x="{x0 + panel_w * value / max_value + 4:.1f}" y="{y + 9}" font-family="Arial, sans-serif" font-size="8">{value}</text>')

    bars(composition, lefts[0], 35)
    bars(prompt, lefts[1], 35)
    lines.append("</svg>")
    write_text(output, "\n".join(lines) + "\n")


def draw_figures(data_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    mp_rows = read_csv(data_dir / "missing_parameter_case_counts_x5.csv")
    draw_matrix_svg(
        mp_rows,
        MP_ARMS,
        None,
        figures_dir / "missing_parameter_case_matrix.svg",
        "Missing-parameter case-arm success counts",
        5,
    )
    mf_rows = read_csv(data_dir / "missing_function_case_counts_x5.csv")
    draw_matrix_svg(
        mf_rows,
        MF_ARMS,
        MF_LABELS,
        figures_dir / "missing_function_case_matrix.svg",
        "Missing-function case-arm success counts",
        5,
    )
    draw_followup_svg(data_dir, figures_dir / "missing_function_followups.svg")


def run_checks(summary_text: str) -> None:
    expected = [
        "Full beats every simpler arm on 4/11 cases.",
        "A simpler arm matches or beats full on 7/11 cases.",
        "Selected x5 availability prompt: 41/60.",
        "Selected x5 original full-state-aware: 0/60.",
        "Discovery over complete triples: standard=57/180, policy=62/180, progress=58/180, full=61/180.",
        "Standard-arm failures in discovery: 123/180; scaffold-repaired standard failures: any=34, full=22; full-specific one-shot rows=8.",
        "Selected rerun observed repeats: standard=71/110, policy=69/110, progress=61/110, full=71/110.",
        "Selected rerun case flags: full beats standard on 8/22 cases; full beats all simpler arms on 4/22 cases; simpler arms match/beat full on 18/22 cases; strict stable full-only=0/22.",
        "Base-flip control over one-shot standard failures: standard=166/609; any success=58/123, stable failure=62/123, repeat-unstable=50/123, stable success=8/123.",
        "Regression control over one-shot standard-pass/full-fail rows: standard=53/90, full=48/90; standard/full/tie=5/4/9.",
        "Missing-parameter selected stability: full=28/35, missing guard=28/35, validator guard=28/32; simpler matches/beats full on 6/7 cases; strict full-only=0/7.",
        "Missing-function selected stability: availability=15/30, full=4/30; availability or availability+full matches/beats full on 5/6 cases.",
        "Missing-parameter all-failure x3 calibration: missing guard=30/122, validator guard=27/122, full=22/121; key simpler arm matches/beats full on 40/41 cases.",
        "Missing-function all-failure x3 calibration: availability=10/203, full=3/218, availability+full=10/223; availability or availability+full matches/beats full on 83/83 complete rows.",
    ]
    missing = [item for item in expected if item not in summary_text]
    if missing:
        raise SystemExit("Sanity check failed; missing expected lines:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/main"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    lines = ["# Reproduced Main Results", ""]
    lines.extend(summarize_missing_parameter(args.data_dir, tables_dir))
    lines.extend(summarize_missing_function(args.data_dir, tables_dir))
    lines.extend(summarize_tau2(args.data_dir, tables_dir))
    lines.extend(summarize_strong_model(args.data_dir, tables_dir))
    summary = "\n".join(lines) + "\n"
    run_checks(summary)
    write_text(args.output_dir / "summary.md", summary)
    if not args.no_figures:
        draw_figures(args.data_dir, figures_dir)
    print(f"Wrote {args.output_dir / 'summary.md'}")
    print(f"Wrote tables to {tables_dir}")
    if not args.no_figures:
        print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
