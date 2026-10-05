"""Opt-in API caller for the released zero-shot client-side protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {
    "type": "object",
    "properties": {
        "long_parameter_list": {"type": "boolean"},
        "long_method": {"type": "boolean"},
    },
    "required": ["long_parameter_list", "long_method"],
    "additionalProperties": False,
}
RESPONSES_MODELS = ("gpt-5.6-sol", "gpt-6-astra", "gemini-3.8-flash")
DEEPSEEK_MODEL = "deepseek-v4-pro-ga-260813"
MODELS = RESPONSES_MODELS + (DEEPSEEK_MODEL,)


def frozen_inputs() -> list[dict]:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    paths = ("data/test_matched_decoder.jsonl", "prompts/system.txt", "prompts/user.txt")
    for relative in paths:
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != protocol["sha256"][relative]:
            raise ValueError(f"Frozen input changed: {relative}")
    rows = [json.loads(line) for line in
            (ROOT / paths[0]).read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != protocol["test_samples"] or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate test input")
    for row in rows:
        if set(row) != {"sample_id", "code", "code_sha256"}:
            raise ValueError("Inference input may contain only ID, code, and code hash")
        if hashlib.sha256(row["code"].encode("utf-8")).hexdigest() != row["code_sha256"]:
            raise ValueError(f"Code hash mismatch: {row['sample_id']}")
    return rows


def request_body(model: str, system: str, user: str) -> dict:
    if model in RESPONSES_MODELS:
        return {
            "model": model,
            "store": False,
            "input": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 8192,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "code_smell_labels",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        }
    return {
        "model": model,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "service_tier": "default",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def parse_labels(text: str) -> int:
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        match = re.fullmatch(re.escape(fence) + r"(?:json)?\s*([\s\S]*?)\s*" +
                             re.escape(fence), text)
        if not match:
            raise ValueError("Invalid JSON fence")
        text = match.group(1)

    def unique_pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj

    obj = json.loads(text, object_pairs_hook=unique_pairs)
    if not isinstance(obj, dict) or set(obj) != set(SCHEMA["required"]):
        raise ValueError("Unexpected JSON keys")
    if any(type(value) is not bool for value in obj.values()):
        raise ValueError("Both decisions must be JSON booleans")
    return int(obj["long_parameter_list"]) + 2 * int(obj["long_method"])


def parse_transport(raw: str) -> dict:
    stripped = raw.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    terminal = None
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        item = line[5:].strip()
        if item == "[DONE]":
            continue
        try:
            event = json.loads(item)
        except ValueError:
            continue
        if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
            terminal = event.get("response")
        elif event.get("type") == "error":
            terminal = {"error": event}
    if not isinstance(terminal, dict):
        raise ValueError("Missing terminal response")
    return terminal


def extract(response: dict, model: str) -> dict:
    if model in RESPONSES_MODELS:
        blocks = [block for item in response.get("output", []) if item.get("type") == "message"
                  for block in item.get("content", [])]
        text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "output_text")
        complete = response.get("status") == "completed" and not any(
            block.get("type") == "refusal" for block in blocks
        )
    else:
        choices = response.get("choices", [])
        item = choices[0] if choices else {}
        text = (item.get("message") or {}).get("content") or ""
        complete = item.get("finish_reason") == "stop"
    status, prediction, error = "invalid_response", None, "incomplete_or_refused"
    if complete:
        try:
            prediction = parse_labels(text)
            status, error = "valid", ""
        except (ValueError, TypeError):
            error = "invalid_label_json"
    return {
        "status": status,
        "prediction": prediction,
        "error": error,
        "response_model": response.get("model"),
        "usage": response.get("usage"),
        "raw_model_output": text,
    }


def endpoint_and_key(model: str) -> tuple[str, str]:
    if model == "gemini-3.8-flash":
        return (os.environ.get("GEMINI_RESPONSES_ENDPOINT", ""),
                os.environ.get("GEMINI_API_KEY", ""))
    if model in RESPONSES_MODELS:
        return (os.environ.get("RESPONSES_ENDPOINT", "https://api.openai.com/v1/responses"),
                os.environ.get("OPENAI_API_KEY", ""))
    return (os.environ.get("ARK_CHAT_ENDPOINT",
                           "https://ark.cn-beijing.volces.com/api/v3/chat/completions"),
            os.environ.get("ARK_API_KEY", ""))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise urllib.error.HTTPError(request.full_url, code, "Redirect refused", headers, fp)


def call(endpoint: str, key: str, body: dict) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "PromptSmell-ZeroShot/1",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=180) as response:
        return parse_transport(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--execute", action="store_true", help="Send billable API requests")
    parser.add_argument("--limit", type=int, default=0, help="First N test methods; 0 means all")
    parser.add_argument("--output", type=Path, help="JSONL result path")
    args = parser.parse_args()
    rows = frozen_inputs()
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    rows = rows[:args.limit] if args.limit else rows
    system = (ROOT / "prompts/system.txt").read_text(encoding="utf-8")
    template = (ROOT / "prompts/user.txt").read_text(encoding="utf-8")
    if template.count("{code}") != 1:
        raise ValueError("Prompt must contain exactly one code placeholder")
    if not args.execute:
        print(json.dumps({"model": args.model, "selected_samples": len(rows),
                          "sample_request": request_body(
                              args.model, system, template.replace("{code}", rows[0]["code"]))},
                         indent=2))
        return

    endpoint, key = endpoint_and_key(args.model)
    if not endpoint or not key:
        raise ValueError("Set an authorized endpoint and API key for the selected model")
    output = args.output or ROOT / ".local" / (args.model + ".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line:
                old = json.loads(line)
                if old.get("model") != args.model:
                    raise ValueError("Output file contains a different model")
                completed[old["sample_id"]] = old["status"]
    with output.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            if completed.get(sample_id) in ("valid", "invalid_response"):
                continue
            body = request_body(args.model, system, template.replace("{code}", row["code"]))
            record = {"sample_id": sample_id, "model": args.model, "code_sha256": row["code_sha256"]}
            max_attempts = 4 if args.model == DEEPSEEK_MODEL else 3
            for attempt in range(1, max_attempts + 1):
                try:
                    response = call(endpoint, key, body)
                    if response.get("error"):
                        raise ValueError("Provider returned an error object")
                    record.update(extract(response, args.model))
                    break
                except urllib.error.HTTPError as exc:
                    record.update(status="api_error", prediction=None, error=f"HTTP_{exc.code}")
                    if exc.code not in (429, 500, 502, 503, 504):
                        break
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    record.update(status="api_error", prediction=None, error=type(exc).__name__)
                if attempt < max_attempts:
                    time.sleep(min(3 * attempt, 15))
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{len(rows)} {sample_id}: {record['status']}")


if __name__ == "__main__":
    main()
