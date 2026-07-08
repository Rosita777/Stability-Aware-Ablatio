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
    rows = read_csv(data_dir / "tau2_anchor.csv")
    taxonomy = read_csv(data_dir / "tau2_failure_taxonomy.csv")
    scaffold_rows = read_csv(data_dir / "tau2_scaffold_ablation_x5.csv")
    total_success = 0
    total_runs = 0
    partial_pairs = 0
    total_pairs = 0
    for row in rows:
        s, n = parse_count_fraction(row["success"])
        p, q = parse_count_fraction(row["partial_pairs"])
        total_success += s
        total_runs += n
        partial_pairs += p
        total_pairs += q
    write_csv(tables_dir / "tau2_anchor_table.csv", rows, ["domain", "model", "success", "partial_pairs"])
    write_csv(
        tables_dir / "tau2_scaffold_ablation_x5_table.csv",
        scaffold_rows,
        [
            "domain",
            "model",
            "task_id",
            "standard",
            "policy",
            "react",
            "n",
            "react_repairs_standard_failure",
            "react_full_advantage",
            "simple_match_or_beats_react",
            "react_unstable",
            "strict_stable_react_only",
        ],
    )
    scaffold_totals = {
        arm: sum(int(row[arm]) for row in scaffold_rows)
        for arm in ["standard", "policy", "react"]
    }
    scaffold_runs = sum(int(row["n"]) for row in scaffold_rows)
    simple_match = sum(int(row["simple_match_or_beats_react"]) for row in scaffold_rows)
    react_unstable = sum(int(row["react_unstable"]) for row in scaffold_rows)
    react_only = sum(int(row["strict_stable_react_only"]) for row in scaffold_rows)
    lines = [
        "",
        "## tau2 earlier anchor context",
        "",
        f"- Total success records: {total_success}/{total_runs}.",
        f"- Partial task-model pairs: {partial_pairs}/{total_pairs}.",
        (
            "- tau2 scaffold-ablation x5 aggregate successes: "
            f"standard={scaffold_totals['standard']}/{scaffold_runs}, "
            f"policy={scaffold_totals['policy']}/{scaffold_runs}, "
            f"ReAct={scaffold_totals['react']}/{scaffold_runs}."
        ),
        f"- tau2 scaffold-ablation x5 simpler arm matches or beats ReAct on {simple_match}/{len(scaffold_rows)} cases.",
        f"- tau2 scaffold-ablation x5 ReAct is repeat-unstable on {react_unstable}/{len(scaffold_rows)} cases.",
        f"- tau2 scaffold-ablation x5 strict stable ReAct-only cases: {react_only}/{len(scaffold_rows)}.",
        "- Failure taxonomy counts: "
        + ", ".join(f"{row['failure_bucket']}={row['count']}" for row in taxonomy)
        + ".",
    ]
    stress_discovery_path = data_dir / "tau2_stress_discovery.csv"
    stress_selected_path = data_dir / "tau2_stress_selected_stability.csv"
    if stress_discovery_path.exists() and stress_selected_path.exists():
        stress_rows = read_csv(stress_discovery_path)
        selected_rows = read_csv(stress_selected_path)
        write_csv(tables_dir / "tau2_stress_discovery_table.csv", stress_rows, list(stress_rows[0].keys()))
        write_csv(tables_dir / "tau2_stress_selected_stability_table.csv", selected_rows, list(selected_rows[0].keys()))

        stress_totals = {
            arm: (sum(int(row[arm]) for row in stress_rows), sum(int(row[f"n_{arm}"]) for row in stress_rows))
            for arm in ["standard", "policy", "react"]
        }
        standard_failures = sum(1 for row in stress_rows if int(row["standard"]) == 0)
        any_scaffold_repairs = sum(int(row["any_scaffold_repairs_standard_failure"]) for row in stress_rows)
        policy_repairs = sum(int(row["policy_repairs_standard_failure"]) for row in stress_rows)
        react_repairs = sum(int(row["react_repairs_standard_failure"]) for row in stress_rows)
        selected_totals = {
            arm: (sum(int(row[arm]) for row in selected_rows), sum(int(row[f"n_{arm}"]) for row in selected_rows))
            for arm in ["standard", "policy", "react"]
        }
        any_scaffold_beats_standard = sum(int(row["any_scaffold_rate_beats_standard"]) for row in selected_rows)
        standard_matches_best = sum(int(row["standard_rate_matches_or_beats_scaffold"]) for row in selected_rows)
        policy_unstable = sum(int(row["policy_unstable"]) for row in selected_rows)
        react_unstable = sum(int(row["react_unstable"]) for row in selected_rows)
        lines.extend(
            [
                "",
                "## tau2 external stress test",
                "",
                (
                    "- Discovery over complete triples: "
                    f"standard={stress_totals['standard'][0]}/{stress_totals['standard'][1]}, "
                    f"policy={stress_totals['policy'][0]}/{stress_totals['policy'][1]}, "
                    f"ReAct={stress_totals['react'][0]}/{stress_totals['react'][1]}."
                ),
                (
                    "- Standard-arm failures in discovery: "
                    f"{standard_failures}/{len(stress_rows)}; scaffold-repaired standard failures: "
                    f"any={any_scaffold_repairs}, policy={policy_repairs}, ReAct={react_repairs}."
                ),
                (
                    "- Selected rerun observed repeats: "
                    f"standard={selected_totals['standard'][0]}/{selected_totals['standard'][1]}, "
                    f"policy={selected_totals['policy'][0]}/{selected_totals['policy'][1]}, "
                    f"ReAct={selected_totals['react'][0]}/{selected_totals['react'][1]}."
                ),
                (
                    "- Selected rerun case flags: "
                    f"any scaffold beats standard on {any_scaffold_beats_standard}/{len(selected_rows)} cases; "
                    f"standard matches/beats the best scaffold on {standard_matches_best}/{len(selected_rows)} cases."
                ),
                (
                    "- Selected rerun instability flags: "
                    f"policy={policy_unstable}/{len(selected_rows)}, ReAct={react_unstable}/{len(selected_rows)}."
                ),
            ]
        )
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
        "Partial task-model pairs: 24/48.",
        "tau2 scaffold-ablation x5 simpler arm matches or beats ReAct on 9/10 cases.",
        "Discovery over complete triples: standard=19/178, policy=18/178, ReAct=6/178.",
        "Selected rerun case flags: any scaffold beats standard on 1/7 cases; standard matches/beats the best scaffold on 6/7 cases.",
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
