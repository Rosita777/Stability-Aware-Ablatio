#!/usr/bin/env python3
"""Run a lightweight BFCL multi-turn external validation subset.

This runner uses BFCL's public multi-turn prompts, function docs, possible
answers, and official multi-turn checker. It is intentionally a subset adapter,
not a replacement for the official BFCL runner.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BFCL_ROOT = Path(os.environ.get("BFCL_ROOT", "third_party/berkeley-function-call-leaderboard"))
DEFAULT_DATA_DIR = Path(
    os.environ.get("BFCL_DATA_DIR", str(DEFAULT_BFCL_ROOT / "bfcl_eval" / "data"))
)

CATEGORY_FILES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}

CLASS_DOC_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


@dataclass(frozen=True)
class BFCLMultiTurnCase:
    category: str
    prompt: dict[str, Any]
    ground_truth: list[list[str]]


def ensure_bfcl_imports(bfcl_root: Path) -> None:
    sys.path.insert(0, str(bfcl_root))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def needs_max_completion_tokens(model: str) -> bool:
    return model.startswith("openai/gpt-5") or model.startswith("openai/o")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_cases(data_dir: Path, categories: list[str], limit_per_category: int, offset: int) -> list[BFCLMultiTurnCase]:
    cases: list[BFCLMultiTurnCase] = []
    for category in categories:
        if category not in CATEGORY_FILES:
            raise SystemExit(f"Unsupported category: {category}")
        prompt_path = data_dir / CATEGORY_FILES[category]
        answer_path = data_dir / "possible_answer" / CATEGORY_FILES[category]
        prompts = load_jsonl(prompt_path)
        answers = {item["id"]: item["ground_truth"] for item in load_jsonl(answer_path)}
        selected = prompts[offset:]
        if limit_per_category > 0:
            selected = selected[:limit_per_category]
        for prompt in selected:
            cases.append(BFCLMultiTurnCase(category=category, prompt=prompt, ground_truth=answers[prompt["id"]]))
    return cases


def parse_model_task_pairs(raw: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if not raw.strip():
        return pairs
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "::" not in item:
            raise SystemExit(f"Invalid --model-task-pairs item: {item!r}. Expected MODEL::TASK_ID.")
        model, task_id = item.split("::", 1)
        model = model.strip()
        task_id = task_id.strip()
        if not model or not task_id:
            raise SystemExit(f"Invalid --model-task-pairs item: {item!r}. Expected MODEL::TASK_ID.")
        pairs.add((model, task_id))
    return pairs


def sanitize_function_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]


def convert_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [convert_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "optional":
            continue
        if key == "type":
            if value == "dict":
                converted[key] = "object"
            elif value == "float":
                converted[key] = "number"
            elif value == "tuple":
                converted[key] = "array"
            elif value == "any":
                converted[key] = "string"
            else:
                converted[key] = value
        else:
            converted[key] = convert_schema(value)
    return converted


def load_tool_docs(data_dir: Path, involved_classes: list[str], excluded: list[str]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    excluded_set = {sanitize_function_name(name) for name in (excluded or [])}
    func_doc_dir = data_dir / "multi_turn_func_doc"
    for class_name in involved_classes:
        doc_file = CLASS_DOC_FILES.get(class_name)
        if not doc_file:
            continue
        for item in load_jsonl(func_doc_dir / doc_file):
            name = sanitize_function_name(item["name"])
            if name in excluded_set:
                continue
            parameters = convert_schema(item.get("parameters", {"type": "object", "properties": {}}))
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            parameters.setdefault("required", [])
            docs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": item.get("description", ""),
                        "parameters": parameters,
                    },
                }
            )
    return docs


def tool_names_from_missed_function(missed_function: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(missed_function, dict):
        return names
    for raw_names in missed_function.values():
        if not isinstance(raw_names, list):
            continue
        for raw_name in raw_names:
            if isinstance(raw_name, str):
                names.add(sanitize_function_name(raw_name))
            elif isinstance(raw_name, dict) and isinstance(raw_name.get("name"), str):
                names.add(sanitize_function_name(raw_name["name"]))
    return names


def tool_docs_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        if isinstance(name, str):
            by_name[name] = tool
    return by_name


def missed_function_names_for_turn(missed_function: Any, turn_idx: int) -> list[str]:
    if not isinstance(missed_function, dict):
        return []
    raw_names = missed_function.get(str(turn_idx), missed_function.get(turn_idx, []))
    if not isinstance(raw_names, list):
        return []
    names: list[str] = []
    for raw_name in raw_names:
        if isinstance(raw_name, str):
            names.append(sanitize_function_name(raw_name))
        elif isinstance(raw_name, dict) and isinstance(raw_name.get("name"), str):
            names.append(sanitize_function_name(raw_name["name"]))
    return names


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
    }
    if needs_max_completion_tokens(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if model.startswith("openai/"):
        payload["parallel_tool_calls"] = True
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {error.code}: {body[:1000]}") from error
        except urllib.error.URLError as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"URL error after retries: {last_error}") from last_error


def parse_tool_args(raw: str | dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"invalid_json_args:{exc}"
    if not isinstance(parsed, dict):
        return {}, "non_object_args"
    return parsed, None


def py_literal(value: Any) -> str:
    return repr(value)


def call_to_string(name: str, args: dict[str, Any]) -> str:
    if not args:
        return f"{name}()"
    rendered = ", ".join(f"{key}={py_literal(value)}" for key, value in sorted(args.items()))
    return f"{name}({rendered})"


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    if normalized.get("content") is None:
        normalized["content"] = ""
    return normalized


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return repr(value)


def execute_for_feedback(
    calls: list[str],
    case: BFCLMultiTurnCase,
    run_model_name: str,
    bfcl_root: Path,
) -> list[str]:
    ensure_bfcl_imports(bfcl_root)
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    results, _instances = execute_multi_turn_func_call(
        func_call_list=calls,
        initial_config=case.prompt["initial_config"],
        involved_classes=case.prompt["involved_classes"],
        model_name=run_model_name,
        test_entry_id=case.prompt["id"],
        long_context="long_context" in case.category,
        is_evaL_run=False,
    )
    return results


def official_check(
    decoded_turns: list[list[list[str]]],
    case: BFCLMultiTurnCase,
    checker_model_name: str,
    bfcl_root: Path,
) -> dict[str, Any]:
    ensure_bfcl_imports(bfcl_root)
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
        multi_turn_irrelevance_checker,
    )

    irrelevance = multi_turn_irrelevance_checker(decoded_turns, case.ground_truth)
    if not irrelevance.get("valid"):
        return irrelevance
    return multi_turn_checker(
        multi_turn_model_result_list_decoded=decoded_turns,
        multi_turn_ground_truth_list=case.ground_truth,
        test_entry=case.prompt,
        test_category=case.category,
        model_name=checker_model_name,
    )


def classify_failure(check_result: dict[str, Any], format_error: str | None) -> tuple[bool, str, str]:
    if format_error:
        if format_error == "max_steps_per_turn_exceeded":
            return False, "turn_loop", format_error
        return False, "format_error", format_error
    if check_result.get("valid"):
        return True, "none", "Official BFCL multi-turn checker passed."
    error_type = str(check_result.get("error_type", "unknown"))
    message = str(check_result.get("error_message", check_result))[:1000]
    if "irrelevance_error" in error_type:
        return False, "unnecessary_tool_call", message
    if "empty_turn_model_response" in error_type:
        return False, "failed_followthrough", message
    if "instance_state_mismatch" in error_type:
        return False, "state_mismatch", message
    if "execution_response_mismatch" in error_type:
        return False, "execution_response_mismatch", message
    return False, "checker_failure", message


def expected_call_count(ground_truth: list[list[str]]) -> int:
    return sum(len(turn) for turn in ground_truth)


def observed_call_count(decoded_turns: list[list[list[str]]]) -> int:
    return sum(len(step) for turn in decoded_turns for step in turn)


def uses_state_reminder(intervention_mode: str) -> bool:
    return intervention_mode in {
        "state_aware_prompt",
        "state_aware_validator_heuristic",
        "function_availability_state_prompt",
        "function_availability_full_state_aware",
    }


def uses_function_availability_prompt(intervention_mode: str) -> bool:
    return intervention_mode in {
        "function_availability_prompt",
        "function_availability_state_prompt",
        "function_availability_missing_guard",
        "function_availability_argument_validator",
        "function_availability_validator_guard",
        "function_availability_full_state_aware",
        "function_availability_prompt_v2",
        "function_availability_prompt_v3",
        "function_availability_prompt_v4",
    }


def function_availability_instruction(intervention_mode: str) -> str:
    if intervention_mode == "function_availability_prompt_v2":
        return (
            " Use only functions that are currently listed. If no listed function exactly supports the user's requested action, "
            "do not use a nearby or approximate tool. Say briefly that the needed function is not available yet, keep the request pending, "
            "and try again if the user later says the function set has changed."
        )
    if intervention_mode == "function_availability_prompt_v3":
        return (
            " Treat tool availability as part of the conversation state. When the required capability is absent from the current tool list, "
            "pause instead of substituting another function. Remember the pending operation, and when additional functions become available, "
            "continue with the best matching newly available function."
        )
    if intervention_mode == "function_availability_prompt_v4":
        return (
            " Before each tool call, check whether the current tool list contains the intended operation. If it does not, make no tool call for that operation. "
            "Do not satisfy the request by guessing with a different function. Wait for clarification or for an updated tool list, then proceed."
        )
    return (
        " Some turns may request an operation whose matching function is not currently available. "
        "If the current tool list has no suitable function for the requested operation, do not call an approximate substitute. "
        "Briefly state that the needed function is unavailable and wait for additional functions or user clarification. "
        "If a later user message says that functions have been updated, resume the pending operation using the newly available tools."
    )


def build_system_prompt(intervention_mode: str) -> str:
    system = (
        "You are a careful multi-turn tool-use agent. Use the provided tools when they are needed. "
        "If the user request is missing a required parameter that cannot be inferred, ask a brief clarification instead of calling tools. "
        "Do not call extra tools after the current user turn is satisfied."
    )
    if intervention_mode == "sham_validator_control":
        system += (
            " A separate scaffold may perform a cosmetic audit of tool names, such as whether names are lowercase, "
            "but that audit is advisory only and will not reject calls. Do not change the task plan because of this audit."
        )
    if intervention_mode == "recovery_terminal_stop":
        system += (
            " After each user turn, stop tool use as soon as the needed calls for that turn have succeeded. "
            "Do not continue with unrelated cleanup or exploratory calls."
        )
    if intervention_mode in {
        "tool_contract_prompt",
        "tool_contract_oracle_empty_guard",
        "state_aware_prompt",
        "state_aware_validator_heuristic",
        "function_availability_state_prompt",
        "function_availability_full_state_aware",
    }:
        system += (
            " Strictly obey each tool's parameter contract. For file-system tools, arguments such as file_name, source, "
            "and destination are names in the current directory unless the tool description explicitly allows paths. "
            "Use cd one folder level at a time before file operations instead of passing paths as file names. "
            "Do not invent file names, and do not create directories unless the user requested it or the operation requires it."
        )
    if uses_state_reminder(intervention_mode):
        system += (
            " Maintain an explicit current-working-directory state across turns. If a tool returns an error, treat it as blocking: "
            "do not claim success, and repair by inspecting the current directory or navigating to the correct folder. "
            "When a find result contains a path, navigate to the containing folder with one-level cd calls before using local file tools."
        )
    if intervention_mode == "missing_param_prompt":
        system += (
            " When a user turn is ambiguous, such as asking for one of several files or for several lines without a number, "
            "do not inspect, choose, or execute tools in that turn. Ask exactly one concise clarification question and wait."
        )
    if uses_function_availability_prompt(intervention_mode):
        system += function_availability_instruction(intervention_mode)
    return system


def uses_oracle_empty_turn_guard(intervention_mode: str) -> bool:
    return intervention_mode in {"oracle_empty_turn_guard", "tool_contract_oracle_empty_guard"}


def uses_heuristic_missing_param_guard(intervention_mode: str) -> bool:
    return intervention_mode in {
        "heuristic_missing_param_guard",
        "validator_heuristic_guard",
        "state_aware_validator_heuristic",
        "function_availability_missing_guard",
        "function_availability_validator_guard",
        "function_availability_full_state_aware",
    }


def uses_argument_validator(intervention_mode: str) -> bool:
    return intervention_mode in {
        "argument_validator_repair",
        "validator_heuristic_guard",
        "state_aware_validator_heuristic",
        "parameter_only_validator",
        "function_availability_argument_validator",
        "function_availability_validator_guard",
        "function_availability_full_state_aware",
    }


def uses_parameter_only_validator(intervention_mode: str) -> bool:
    return intervention_mode == "parameter_only_validator"


def has_explicit_file_name(text: str) -> bool:
    return bool(re.search(r"'[^']+\.[^']+'|\"[^\"]+\.[^\"]+\"", text))


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def should_guard_missing_param(turn_messages: list[dict[str, Any]]) -> bool:
    text = " ".join(str(message.get("content", "")) for message in turn_messages).lower()
    if not text:
        return False
    if "one of the file" in text or "one of my file" in text:
        return True
    if "one of the files" in text or "one of my files" in text:
        return not has_explicit_file_name(text)
    if ("last several lines" in text or "last few lines" in text or "several lines" in text) and not has_digit(text):
        return True
    if "specific word" in text and ("file name" in text or "filename" in text):
        return True
    if "mention someone" in text or "mention somebody" in text:
        return True
    if ("add a comment" in text or "give it a comment" in text or "comment underneath" in text) and not re.search(
        r"'[^']+'|\"[^\"]+\"|with a phrase|comment content", text
    ):
        return True
    if "folder name" in text or "directory is" in text or "specific file is" in text or "to be exact" in text:
        return False
    return False


def required_args_by_tool(tools: list[dict[str, Any]]) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name", "")
        parameters = function.get("parameters", {})
        required[name] = set(parameters.get("required", []) or [])
    return required


def parameter_schema_by_tool(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name", "")
        parameters = function.get("parameters", {})
        if isinstance(parameters, dict):
            schemas[name] = parameters
        else:
            schemas[name] = {"type": "object", "properties": {}, "required": []}
    return schemas


def expected_json_types(schema_type: Any) -> set[str]:
    if isinstance(schema_type, list):
        return {str(item) for item in schema_type}
    if isinstance(schema_type, str):
        return {schema_type}
    return set()


def json_type_matches(value: Any, schema_type: Any) -> bool:
    expected = expected_json_types(schema_type)
    if not expected or "any" in expected:
        return True
    if value is None:
        return "null" in expected
    if isinstance(value, bool):
        return "boolean" in expected
    if isinstance(value, int) and not isinstance(value, bool):
        return bool({"integer", "number"} & expected)
    if isinstance(value, float):
        return "number" in expected
    if isinstance(value, str):
        return "string" in expected
    if isinstance(value, list):
        return "array" in expected
    if isinstance(value, dict):
        return "object" in expected
    return False


def validate_parameter_only_args(name: str, args: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> str | None:
    schema = schemas.get(name, {"type": "object", "properties": {}, "required": []})
    required = set(schema.get("required", []) or [])
    missing = sorted(required - set(args))
    if missing:
        return f"Rejected by scaffold: missing required argument(s): {', '.join(missing)}."

    properties = schema.get("properties", {}) or {}
    if not isinstance(properties, dict):
        properties = {}
    for arg_name, value in sorted(args.items()):
        prop_schema = properties.get(arg_name)
        if not isinstance(prop_schema, dict):
            continue
        if "type" in prop_schema and not json_type_matches(value, prop_schema.get("type")):
            expected = prop_schema.get("type")
            return f"Rejected by scaffold: `{name}.{arg_name}` has invalid type; expected {expected}."
        if "enum" in prop_schema and isinstance(prop_schema.get("enum"), list) and value not in prop_schema["enum"]:
            return f"Rejected by scaffold: `{name}.{arg_name}` is not one of the allowed enum values."
    return None


def is_path_like(value: Any) -> bool:
    return isinstance(value, str) and ("/" in value or "\\" in value)


def path_navigation_hint(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.replace("\\", "/")
    parts = [part for part in cleaned.split("/") if part and part != "."]
    if len(parts) <= 1:
        return ""
    dirs = parts[:-1]
    leaf = parts[-1]
    cd_plan = ", then ".join(f"`cd(folder='{part}')`" for part in dirs)
    return f" For `{value}`, use {cd_plan}, then use local name `{leaf}`."


def state_turn_reminder(turn_idx: int) -> str:
    return (
        f"Scaffold state reminder for turn {turn_idx}: BFCL file-system tools operate in the current directory. "
        "`file_name`, `source`, and `destination` should be local names, not paths. "
        "If the target is in another folder, navigate with one-level `cd` calls first. "
        "If a file or folder operation fails, do not report success; inspect with `ls`/`find` or repair the cwd before continuing."
    )


def state_error_hint(result: str) -> str:
    lowered = result.lower()
    error_markers = [
        "error",
        "no such file",
        "no such directory",
        "invalid character",
        "missing",
        "unexpected keyword",
        "required positional argument",
    ]
    if any(marker in lowered for marker in error_markers):
        return (
            "\n\nScaffold state hint: Treat this tool result as blocking. Do not claim the task succeeded. "
            "Check the current directory, use one-level `cd` calls, and retry with local file names only."
        )
    return ""


def validate_tool_call_args(name: str, args: dict[str, Any], required: dict[str, set[str]], state_aware: bool = False) -> str | None:
    missing = sorted(required.get(name, set()) - set(args))
    if missing:
        return f"Rejected by scaffold: missing required argument(s): {', '.join(missing)}."
    local_name_args = {
        "cat": {"file_name"},
        "diff": {"file_name1", "file_name2"},
        "grep": {"file_name"},
        "head": {"file_name"},
        "mkdir": {"dir_name"},
        "mv": {"source", "destination"},
        "cp": {"source", "destination"},
        "rm": {"file_name"},
        "sort": {"file_name"},
        "tail": {"file_name"},
        "touch": {"file_name"},
    }
    for arg_name in local_name_args.get(name, set()):
        if is_path_like(args.get(arg_name)):
            hint = path_navigation_hint(args.get(arg_name)) if state_aware else ""
            return (
                f"Rejected by scaffold: `{name}.{arg_name}` must be a name in the current directory, "
                "not a path. Use `cd` one folder level at a time first." + hint
            )
    if name == "cd" and is_path_like(args.get("folder")):
        hint = path_navigation_hint(args.get("folder")) if state_aware else ""
        return "Rejected by scaffold: `cd.folder` changes one folder level at a time; do not pass paths." + hint
    return None


def run_case(
    base_url: str,
    api_key: str,
    model: str,
    case: BFCLMultiTurnCase,
    max_steps_per_turn: int,
    max_tokens: int,
    temperature: float,
    data_dir: Path,
    bfcl_root: Path,
    intervention_mode: str,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    missed_function = case.prompt.get("missed_function", {})
    initially_hidden = tool_names_from_missed_function(missed_function)
    excluded_function = case.prompt.get("excluded_function") or []
    base_excluded = [*excluded_function, *sorted(initially_hidden)]
    tools = load_tool_docs(data_dir, case.prompt["involved_classes"], base_excluded)
    full_tools_by_name = tool_docs_by_name(
        load_tool_docs(data_dir, case.prompt["involved_classes"], excluded_function)
    )
    available_tool_names = {tool.get("function", {}).get("name", "") for tool in tools}
    required = required_args_by_tool(tools)
    parameter_schemas = parameter_schema_by_tool(tools)
    system = build_system_prompt(intervention_mode)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    trajectory: list[dict[str, Any]] = [
        {"step": 0, "actor": "system", "type": "message", "content": system, "metadata": {}}
    ]
    decoded_turns: list[list[list[str]]] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0}
    rejected_tool_calls = 0
    state_reminder_messages = 0
    format_error: str | None = None
    start = time.time()

    for turn_idx, turn_messages in enumerate(case.prompt["question"]):
        restored_names = missed_function_names_for_turn(missed_function, turn_idx)
        restored_docs: list[dict[str, Any]] = []
        for name in restored_names:
            if name in full_tools_by_name and name not in available_tool_names:
                restored_docs.append(full_tools_by_name[name])
                available_tool_names.add(name)
        if restored_docs:
            tools.extend(restored_docs)
            required = required_args_by_tool(tools)
            parameter_schemas = parameter_schema_by_tool(tools)
            restored_label = ", ".join(sorted(name for name in restored_names if name in full_tools_by_name))
            trajectory.append(
                {
                    "step": len(trajectory),
                    "actor": "scaffold",
                    "type": "tool_availability_update",
                    "content": f"Additional function(s) are now available: {restored_label}",
                    "metadata": {"turn_idx": turn_idx, "restored_functions": sorted(restored_names)},
                }
            )
            if not turn_messages:
                turn_messages = [
                    {
                        "role": "user",
                        "content": "I have updated some more functions you can choose from. What about now?",
                    }
                ]
        for turn_message in turn_messages:
            messages.append(turn_message)
            trajectory.append(
                {
                    "step": len(trajectory),
                    "actor": turn_message.get("role", "user"),
                    "type": "message",
                    "content": turn_message.get("content", ""),
                    "metadata": {"turn_idx": turn_idx},
                }
            )
        if uses_state_reminder(intervention_mode):
            reminder = state_turn_reminder(turn_idx)
            messages.append({"role": "system", "content": reminder})
            state_reminder_messages += 1
            trajectory.append(
                {
                    "step": len(trajectory),
                    "actor": "scaffold",
                    "type": "state_reminder",
                    "content": reminder,
                    "metadata": {"turn_idx": turn_idx},
                }
            )
        if uses_oracle_empty_turn_guard(intervention_mode) and turn_idx < len(case.ground_truth) and not case.ground_truth[turn_idx]:
            content = "Could you clarify the missing required detail before I call tools?"
            assistant_message = {"role": "assistant", "content": content}
            messages.append(assistant_message)
            trajectory.append(
                {
                    "step": len(trajectory),
                    "actor": "agent",
                    "type": "message",
                    "content": content,
                    "metadata": {"turn_idx": turn_idx, "oracle_empty_turn_guard": True},
                }
            )
            decoded_turns.append([])
            continue
        if uses_heuristic_missing_param_guard(intervention_mode) and should_guard_missing_param(turn_messages):
            content = "Could you clarify the missing required detail before I call tools?"
            assistant_message = {"role": "assistant", "content": content}
            messages.append(assistant_message)
            trajectory.append(
                {
                    "step": len(trajectory),
                    "actor": "agent",
                    "type": "message",
                    "content": content,
                    "metadata": {"turn_idx": turn_idx, "heuristic_missing_param_guard": True},
                }
            )
            decoded_turns.append([])
            continue
        turn_decoded_steps: list[list[str]] = []
        for step_idx in range(max_steps_per_turn):
            data = chat_completion(base_url, api_key, model, messages, tools, max_tokens, temperature)
            usage = data.get("usage", {}) or {}
            usage_totals["input_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            usage_totals["output_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            message = normalize_message(data["choices"][0]["message"])
            messages.append(message)
            trajectory.append(
                {
                    "step": len(trajectory),
                    "actor": "agent",
                    "type": "message",
                    "content": message.get("content", ""),
                    "metadata": {
                        "turn_idx": turn_idx,
                        "step_idx": step_idx,
                        "finish_reason": data["choices"][0].get("finish_reason"),
                    },
                }
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            step_calls: list[str] = []
            tool_call_ids: list[tuple[str, str]] = []
            rejected_results: list[tuple[str, str, str]] = []
            for tool_call in tool_calls:
                function = tool_call.get("function", {}) or {}
                name = sanitize_function_name(function.get("name", ""))
                args, error = parse_tool_args(function.get("arguments"))
                if error and uses_argument_validator(intervention_mode):
                    rejection = f"Rejected by scaffold: {error}. Return valid JSON object arguments for `{name}`."
                    if uses_state_reminder(intervention_mode):
                        rejection += " Treat the rejection as blocking and retry only after fixing the arguments."
                    rejected_tool_calls += 1
                    rejected_results.append((tool_call.get("id", ""), name, rejection))
                    trajectory.append(
                        {
                            "step": len(trajectory),
                            "actor": "scaffold",
                            "type": "tool_rejection",
                            "content": rejection,
                            "tool_name": name,
                            "tool_args": {},
                            "metadata": {"turn_idx": turn_idx, "step_idx": step_idx},
                        }
                    )
                    continue
                elif error:
                    format_error = error
                    args = {}
                elif uses_parameter_only_validator(intervention_mode):
                    rejection = validate_parameter_only_args(name, args, parameter_schemas)
                    if rejection:
                        rejected_tool_calls += 1
                        rejected_results.append((tool_call.get("id", ""), name, rejection))
                        trajectory.append(
                            {
                                "step": len(trajectory),
                                "actor": "scaffold",
                                "type": "tool_rejection",
                                "content": rejection,
                                "tool_name": name,
                                "tool_args": args,
                                "metadata": {"turn_idx": turn_idx, "step_idx": step_idx},
                            }
                        )
                        continue
                elif uses_argument_validator(intervention_mode):
                    rejection = validate_tool_call_args(name, args, required, state_aware=uses_state_reminder(intervention_mode))
                    if rejection:
                        rejected_tool_calls += 1
                        rejected_results.append((tool_call.get("id", ""), name, rejection))
                        trajectory.append(
                            {
                                "step": len(trajectory),
                                "actor": "scaffold",
                                "type": "tool_rejection",
                                "content": rejection,
                                "tool_name": name,
                                "tool_args": args,
                                "metadata": {"turn_idx": turn_idx, "step_idx": step_idx},
                            }
                        )
                        continue
                call = call_to_string(name, args)
                step_calls.append(call)
                tool_call_ids.append((tool_call.get("id", ""), name))
                trajectory.append(
                    {
                        "step": len(trajectory),
                        "actor": "agent",
                        "type": "tool_call",
                        "content": "",
                        "tool_name": name,
                        "tool_args": args,
                        "metadata": {"turn_idx": turn_idx, "step_idx": step_idx, "call": call},
                    }
                )
            for tool_call_id, name, rejection in rejected_results:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": rejection})
                trajectory.append(
                    {
                        "step": len(trajectory),
                        "actor": "tool",
                        "type": "tool_result",
                        "content": rejection,
                        "tool_name": name,
                        "tool_result": {"execution_result": rejection},
                        "metadata": {"turn_idx": turn_idx, "step_idx": step_idx, "tool_call_id": tool_call_id, "rejected": True},
                    }
                )
            if not step_calls:
                if format_error:
                    break
                continue
            turn_decoded_steps.append(step_calls)
            try:
                execution_results = execute_for_feedback(
                    step_calls,
                    case,
                    f"{model}_{run_id}_feedback",
                    bfcl_root,
                )
            except Exception as exc:  # noqa: BLE001 - return tool execution error as feedback.
                execution_results = [f"Error during execution: {type(exc).__name__}: {exc}"] * len(step_calls)
            for (tool_call_id, name), result in zip(tool_call_ids, execution_results):
                model_result = result + (state_error_hint(result) if uses_state_reminder(intervention_mode) else "")
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": model_result})
                trajectory.append(
                    {
                        "step": len(trajectory),
                        "actor": "tool",
                        "type": "tool_result",
                        "content": model_result,
                        "tool_name": name,
                        "tool_result": {"execution_result": model_result},
                        "metadata": {"turn_idx": turn_idx, "step_idx": step_idx, "tool_call_id": tool_call_id},
                    }
                )
            if format_error:
                break
        else:
            format_error = "max_steps_per_turn_exceeded"
        decoded_turns.append(turn_decoded_steps)
        if format_error:
            break

    if not format_error and len(decoded_turns) < len(case.ground_truth):
        decoded_turns.extend([] for _ in range(len(case.ground_truth) - len(decoded_turns)))

    check_result = {"valid": False, "error_message": "not_checked", "error_type": "not_checked"}
    if not format_error:
        try:
            check_result = official_check(decoded_turns, case, f"{model}_{run_id}_checker", bfcl_root)
        except Exception as exc:  # noqa: BLE001 - checker failures should be logged.
            format_error = f"checker_exception:{type(exc).__name__}: {exc}"
    success, failure_type, notes = classify_failure(check_result, format_error)
    expected_calls = expected_call_count(case.ground_truth)
    observed_calls = observed_call_count(decoded_turns)

    return {
        "run_id": run_id,
        "timestamp_utc": now_iso(),
        "project_version": "v0.1",
        "benchmark": "bfcl-multiturn-external",
        "split": "external_subset",
        "task_id": case.prompt["id"],
        "category": case.category,
        "scenario": "bfcl_multiturn",
        "task_family": "external_tool_benchmark",
        "model": {"provider": "360", "name": model, "api_base": "redacted", "temperature": temperature, "max_tokens": max_tokens},
        "agent": {
            "scaffold": "direct_tools",
            "version": "bfcl_multiturn_external_v0.2",
            "intervention_mode": intervention_mode,
            "max_steps_per_turn": max_steps_per_turn,
        },
        "result": {"success": success, "score": 1.0 if success else 0.0, "failure_type": failure_type, "failure_notes": notes},
        "cost": {
            "input_tokens": usage_totals["input_tokens"],
            "output_tokens": usage_totals["output_tokens"],
            "tool_calls": observed_calls,
            "expected_tool_calls": expected_calls,
            "excess_tool_calls": max(0, observed_calls - expected_calls),
            "rejected_tool_calls": rejected_tool_calls,
            "state_reminder_messages": state_reminder_messages,
            "wall_time_s": round(time.time() - start, 3),
            "estimated_usd": 0.0,
        },
        "trajectory": trajectory,
        "bfcl": {
            "ground_truth": case.ground_truth,
            "decoded_turns": decoded_turns,
            "checker_result": json_safe(check_result),
            "involved_classes": case.prompt.get("involved_classes", []),
            "excluded_function": excluded_function,
            "missed_function": json_safe(missed_function),
            "dynamic_missed_function_tools": bool(missed_function),
        },
        "evaluator": {
            "type": "bfcl_official_multi_turn_checker",
            "version": "bfcl_v4_local",
            "notes": "Lightweight external subset adapter around BFCL public multi-turn data and checker.",
        },
    }


def error_record(model: str, case: BFCLMultiTurnCase, error: str, temperature: float, max_tokens: int, intervention_mode: str) -> dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": now_iso(),
        "project_version": "v0.1",
        "benchmark": "bfcl-multiturn-external",
        "split": "external_subset",
        "task_id": case.prompt["id"],
        "category": case.category,
        "scenario": "bfcl_multiturn",
        "task_family": "external_tool_benchmark",
        "model": {"provider": "360", "name": model, "api_base": "redacted", "temperature": temperature, "max_tokens": max_tokens},
        "agent": {
            "scaffold": "direct_tools",
            "version": "bfcl_multiturn_external_v0.2",
            "intervention_mode": intervention_mode,
        },
        "result": {"success": False, "score": 0.0, "failure_type": "provider_error", "failure_notes": error[:1000]},
        "cost": {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "expected_tool_calls": expected_call_count(case.ground_truth), "excess_tool_calls": 0, "wall_time_s": 0.0, "estimated_usd": 0.0},
        "trajectory": [{"step": 0, "actor": "evaluator", "type": "error", "content": error[:1000], "metadata": {}}],
        "bfcl": {"ground_truth": case.ground_truth, "decoded_turns": [], "checker_result": {"valid": False, "error_type": "provider_error"}},
        "evaluator": {"type": "bfcl_official_multi_turn_checker", "version": "bfcl_v4_local", "notes": "Provider or harness error."},
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(record["model"]["name"], []).append(record)
    return {
        model: {
            "n": len(items),
            "success_rate": sum(1 for item in items if item["result"]["success"]) / max(1, len(items)),
        }
        for model, items in sorted(by_model.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--bfcl-root", default=str(DEFAULT_BFCL_ROOT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models", default=os.environ.get("MODEL_LIST", "openai/gpt-4o-mini"))
    parser.add_argument("--categories", default="multi_turn_base,multi_turn_miss_param")
    parser.add_argument("--limit-per-category", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", default="runs/bfcl_multiturn_external_subset.jsonl")
    parser.add_argument("--max-steps-per-turn", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--model-task-pairs",
        default="",
        help="Comma-separated MODEL::TASK_ID filters. When set, only these exact pairs are queued.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--job-order",
        choices=["model_major", "round_robin"],
        default="round_robin",
        help="Order queued jobs before they enter the thread pool. round_robin interleaves models for faster multi-model runs.",
    )
    parser.add_argument(
        "--intervention-mode",
        choices=[
            "model",
            "noop_control",
            "sham_validator_control",
            "recovery_terminal_stop",
            "tool_contract_prompt",
            "missing_param_prompt",
            "function_availability_prompt",
            "oracle_empty_turn_guard",
            "tool_contract_oracle_empty_guard",
            "heuristic_missing_param_guard",
            "argument_validator_repair",
            "validator_heuristic_guard",
            "parameter_only_validator",
            "state_aware_prompt",
            "state_aware_validator_heuristic",
            "function_availability_state_prompt",
            "function_availability_missing_guard",
            "function_availability_argument_validator",
            "function_availability_validator_guard",
            "function_availability_full_state_aware",
            "function_availability_prompt_v2",
            "function_availability_prompt_v3",
            "function_availability_prompt_v4",
        ],
        default="model",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    bfcl_root = Path(args.bfcl_root)
    data_dir = Path(args.data_dir)
    ensure_bfcl_imports(bfcl_root)

    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    cases = load_cases(data_dir, categories, args.limit_per_category, args.offset)
    model_task_pairs = parse_model_task_pairs(args.model_task_pairs)
    if model_task_pairs:
        for pair_model, _ in sorted(model_task_pairs):
            if pair_model not in models:
                models.append(pair_model)
    if args.job_order == "model_major":
        base_jobs = [(model, case) for model in models for case in cases]
    else:
        base_jobs = [(model, case) for case in cases for model in models]
    if model_task_pairs:
        base_jobs = [(model, case) for model, case in base_jobs if (model, case.prompt["id"]) in model_task_pairs]
        found_pairs = {(model, case.prompt["id"]) for model, case in base_jobs}
        missing_pairs = sorted(model_task_pairs - found_pairs)
        if missing_pairs:
            missing = ", ".join(f"{model}::{task_id}" for model, task_id in missing_pairs)
            raise SystemExit(f"--model-task-pairs did not match loaded cases/models: {missing}")
    if args.repeat_count < 1:
        raise SystemExit("--repeat-count must be >= 1")
    jobs = [
        (repeat_idx, model, case)
        for repeat_idx in range(args.repeat_count)
        for model, case in base_jobs
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    open_mode = "w" if args.overwrite else "a"
    with open(args.output, open_mode, encoding="utf-8") as handle:
        def worker(repeat_idx: int, model: str, case: BFCLMultiTurnCase) -> dict[str, Any]:
            try:
                record = run_case(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=model,
                    case=case,
                    max_steps_per_turn=args.max_steps_per_turn,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    data_dir=data_dir,
                    bfcl_root=bfcl_root,
                    intervention_mode=args.intervention_mode,
                )
                record["run"] = {"repeat_idx": repeat_idx, "model_task_pair": f"{model}::{case.prompt['id']}"}
                return record
            except Exception as exc:  # noqa: BLE001 - preserve failed attempts.
                record = error_record(model, case, f"{type(exc).__name__}: {exc}", args.temperature, args.max_tokens, args.intervention_mode)
                record["run"] = {"repeat_idx": repeat_idx, "model_task_pair": f"{model}::{case.prompt['id']}"}
                return record

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_job = {executor.submit(worker, repeat_idx, model, case): (repeat_idx, model, case) for repeat_idx, model, case in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                record = future.result()
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                print(
                    f"repeat={record.get('run', {}).get('repeat_idx', 0)} "
                    f"{record['model']['name']} {record['category']} {record['task_id']}: "
                    f"success={record['result']['success']} failure={record['result']['failure_type']} "
                    f"tools={record['cost']['tool_calls']} expected={record['cost']['expected_tool_calls']} "
                    f"arm={record['agent']['intervention_mode']}"
                )

    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
