# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/json_lenient.py
"""Lenient JSON extraction for OCR model answers (TextPecker / TextDART).

Model answers are frequently *almost* JSON: a fenced block with prose around
it, raw newlines inside strings, unescaped quotes in OCR text, or a truncated
tail that is missing closing ``}`` / ``]``.  Strict ``json.loads`` rejects all
of these; this module repairs the common cases with stdlib only (optionally
delegating to ``json_repair`` when installed) and falls back to regex salvage
of ``recognized_text`` / region objects.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_REGION_OBJECT_RE = re.compile(
    r'\{\s*"quad"\s*:\s*(\[[^\]]*\])\s*,\s*"recognized_text"\s*:\s*"((?:\\.|[^"\\])*)"\s*\}'
    r"|"
    r'\{\s*"recognized_text"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*"quad"\s*:\s*(\[[^\]]*\])\s*\}',
    re.DOTALL,
)


def extract_json_payload(raw_response: str) -> str:
    """Best-effort isolation of the JSON body from a raw model answer.

    Prefers a fenced `` ```json `` block; otherwise falls back to the outermost
    ``{...}`` / ``[...]`` span so prose around the JSON does not defeat parsing.
    """
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)
    match = _JSON_BLOCK_RE.search(raw_response)
    if match:
        return match.group(1).strip()
    stripped = raw_response.strip()
    # Drop a leading bare fence line when the model forgot the closing fence.
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    opens = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos != -1]
    closes = [pos for pos in (stripped.rfind("}"), stripped.rfind("]")) if pos != -1]
    if opens and closes:
        start, end = min(opens), max(closes)
        if end > start:
            return stripped[start : end + 1]
    return stripped


def _unescape_json_string(value: str) -> str:
    """Decode a JSON string body (content between quotes) into a Python str."""
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return (
            value.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def _close_unbalanced_brackets(text: str) -> str:
    """Append missing ``]`` / ``}`` for truncated objects/arrays (outside strings)."""
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_string:
        text = text + '"'
    if not stack:
        return text
    # Drop a trailing comma before we close, a common truncation artefact.
    trimmed = text.rstrip()
    if trimmed.endswith(","):
        trimmed = trimmed[:-1]
    return trimmed + "".join(reversed(stack))


def _escape_raw_controls_in_strings(text: str) -> str:
    """Turn raw newlines/tabs inside JSON strings into ``\\n`` / ``\\t``."""
    out: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def _next_non_space(text: str, index: int) -> str:
    j = index
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    return "" if j >= len(text) else text[j]


def _escape_unescaped_quotes_in_strings(text: str) -> str:
    """Escape interior ``"`` that are not valid JSON string terminators.

    Tracks a tiny JSON structural state so object **keys** may end with ``":``
    while string **values** may only end with ``",`` / ``"}`` / ``"]`` / EOF.
    OCR brand quotes immediately following CJK punctuation therefore get escaped, but
    the ``"recognized_text"`` key delimiter is left alone.
    """
    out: List[str] = []
    in_string = False
    escape = False
    # Structural expectation outside strings.
    #   key    — next string is an object key (ends before ':')
    #   value  — next token is a value (string ends before ',' / '}' / ']')
    expect = "value"
    stack: List[str] = []  # "object" | "array"
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                i += 1
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                i += 1
                continue
            if ch == '"':
                follower = _next_non_space(text, i + 1)
                if expect == "key":
                    is_end = follower in (":", "")
                else:
                    is_end = follower in (",", "}", "]", "")
                if is_end:
                    out.append(ch)
                    in_string = False
                    if expect == "key":
                        expect = "value"
                    # after a string value, comma/} will update expect below
                else:
                    out.append('\\"')
                i += 1
                continue
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "{":
            stack.append("object")
            expect = "key"
        elif ch == "[":
            stack.append("array")
            expect = "value"
        elif ch == "}":
            if stack:
                stack.pop()
            expect = "key" if stack and stack[-1] == "object" else "value"
        elif ch == "]":
            if stack:
                stack.pop()
            expect = "key" if stack and stack[-1] == "object" else "value"
        elif ch == ",":
            expect = "key" if stack and stack[-1] == "object" else "value"
        elif ch == ":":
            expect = "value"
        out.append(ch)
        i += 1
    return "".join(out)


def repair_json_text(text: str) -> str:
    """Apply cheap structural repairs that often salvage OCR JSON."""
    repaired = text.strip()
    # Strip a UTF-8 BOM if present.
    if repaired.startswith("\ufeff"):
        repaired = repaired.lstrip("\ufeff")
    # Quotes first: raw-control escaping toggles on unescaped ``"`` and would
    # otherwise treat the OCR-broken interior quote as a string terminator.
    repaired = _escape_unescaped_quotes_in_strings(repaired)
    repaired = _escape_raw_controls_in_strings(repaired)
    repaired = _close_unbalanced_brackets(repaired)
    return repaired


def _try_json_repair_lib(text: str) -> Optional[Any]:
    try:
        from json_repair import repair_json  # type: ignore
    except ImportError:
        return None
    try:
        repaired = repair_json(text, return_objects=True)
    except Exception as exc:  # pylint: disable=broad-except
        # Optional repair failures must fall back to the remaining parsers.
        logger.debug("json_repair failed: %s", exc)
        return None
    if isinstance(repaired, (dict, list, str, int, float, bool)) or repaired is None:
        return repaired
    return None


def loads_json_lenient(raw_response: str) -> Any:
    """Parse model JSON with progressive repair; raise ``JSONDecodeError`` if all fail."""
    body = extract_json_payload(raw_response)
    attempts = [body, repair_json_text(body)]
    # Also try closing brackets on the unrepaired body (truncation-only case).
    closed_only = _close_unbalanced_brackets(body.strip())
    if closed_only not in attempts:
        attempts.append(closed_only)

    errors: List[Exception] = []
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(exc)

    lib_parsed = _try_json_repair_lib(body)
    if lib_parsed is not None:
        return lib_parsed

    if errors:
        raise errors[0]
    raise json.JSONDecodeError("empty JSON payload", body, 0)


def extract_recognized_text_fallback(raw_response: str) -> Optional[str]:
    """Regex salvage of ``recognized_text`` when JSON parsing fully fails.

    Extract recognized text: prefer a
    value bounded by the next key / object end; otherwise take the longest
    plausible quoted span after the key (handles unescaped interior quotes by
    reading until the last ``"`` before ``}``).
    """
    body = extract_json_payload(raw_response)
    patterns = (
        r'"recognized_text"\s*:\s*"(.*?)"\s*,\s*"duplicate_text"',
        r'"recognized_text"\s*:\s*"(.*?)"\s*(?:,\s*"|})',
        r'"text"\s*:\s*"(.*?)"\s*(?:,\s*"|})',
    )
    for pattern in patterns:
        match = re.search(pattern, body, re.DOTALL)
        if match:
            return _unescape_json_string(match.group(1))

    # Last resort for unescaped quotes: from key to the final " before closing }.
    loose = re.search(
        r'"recognized_text"\s*:\s*"(.*)"\s*\}\s*$',
        body,
        re.DOTALL,
    )
    if loose:
        return _unescape_json_string(loose.group(1).replace('\\"', '"'))
    loose = re.search(r'"recognized_text"\s*:\s*"(.*)', body, re.DOTALL)
    if not loose:
        return None
    inner = loose.group(1)
    # Trim a trailing ``"}`` / ``"\n}`` if present.
    inner = re.sub(r'"\s*\}\s*$', "", inner)
    if inner.endswith('"'):
        inner = inner[:-1]
    return _unescape_json_string(inner)


def extract_regions_fallback(raw_response: str) -> List[Dict[str, Any]]:
    """Regex salvage of ``{quad, recognized_text}`` objects from a broken answer."""
    body = extract_json_payload(raw_response)
    regions: List[Dict[str, Any]] = []
    for match in _REGION_OBJECT_RE.finditer(body):
        if match.group(1) is not None:
            quad_raw, text_raw = match.group(1), match.group(2)
        else:
            text_raw, quad_raw = match.group(3), match.group(4)
        try:
            quad = json.loads(quad_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(quad, list):
            continue
        regions.append(
            {
                "quad": quad,
                "recognized_text": _unescape_json_string(text_raw),
            }
        )
    return regions


def normalize_ocr_json_answer(raw_response: str) -> str:
    """Rewrite a model answer as strict JSON when salvage succeeds.

    Used by the TextPecker reward path so ``eval.GenTextEval.scoring.get_score_v2``
    receives a clean ``{"recognized_text": "..."}`` payload instead of failing
    (or truncating) on unescaped quotes / missing braces.
    """
    try:
        parsed = loads_json_lenient(raw_response)
    except json.JSONDecodeError:
        text = extract_recognized_text_fallback(raw_response)
        if text is None:
            return raw_response
        return json.dumps({"recognized_text": text}, ensure_ascii=False)

    if isinstance(parsed, dict) and (
        "recognized_text" in parsed or "text" in parsed or "regions" in parsed
    ):
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, list):
        return json.dumps({"regions": parsed}, ensure_ascii=False)
    if isinstance(parsed, str):
        return json.dumps({"recognized_text": parsed}, ensure_ascii=False)

    text = extract_recognized_text_fallback(raw_response)
    if text is not None:
        return json.dumps({"recognized_text": text}, ensure_ascii=False)
    return raw_response


def parse_recognized_text(raw_response: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(recognized_text, parse_error)`` for whole-image TextPecker answers."""
    body = extract_json_payload(raw_response)
    try:
        parsed = loads_json_lenient(raw_response)
    except json.JSONDecodeError as exc:
        fallback = extract_recognized_text_fallback(raw_response)
        if fallback is not None:
            return fallback, None
        return None, f"json decode failed: {exc}"

    if isinstance(parsed, str):
        return parsed, None
    if isinstance(parsed, list):
        parts: List[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                raw = item.get("recognized_text", item.get("text", item.get("text_content")))
                if isinstance(raw, str) and raw.strip():
                    parts.append(raw.strip())
                else:
                    return None, (
                        f"recognized_text must be str or list[str|dict], got "
                        f"{type(parsed).__name__}: {parsed!r}"
                    )
            else:
                return None, (
                    f"recognized_text must be str or list[str|dict], got "
                    f"{type(parsed).__name__}: {parsed!r}"
                )
        return " ".join(parts), None
    if not isinstance(parsed, dict):
        fallback = extract_recognized_text_fallback(raw_response)
        if fallback is not None:
            return fallback, None
        return None, f"expected a JSON object, got {type(parsed).__name__}: {body[:200]!r}"

    text = parsed.get("recognized_text", parsed.get("text"))
    if text is None and "regions" in parsed and isinstance(parsed["regions"], list):
        # Occasional drift: TextPecker emits TextDART-shaped output.
        parts = []
        for region in parsed["regions"]:
            if isinstance(region, dict):
                raw = region.get("recognized_text", region.get("text"))
                if isinstance(raw, str) and raw.strip():
                    parts.append(raw.strip())
        if parts:
            return "\n".join(parts), None
    if text is None:
        fallback = extract_recognized_text_fallback(raw_response)
        if fallback is not None:
            return fallback, None
        return None, f"no 'recognized_text' key, got keys {sorted(parsed)}"
    if isinstance(text, str):
        return text, None
    if isinstance(text, (list, tuple)):
        coerced, err = parse_recognized_text(
            json.dumps({"recognized_text": text}, ensure_ascii=False)
        )
        return coerced, err
    return None, (
        f"recognized_text must be str or list[str|dict], got " f"{type(text).__name__}: {text!r}"
    )
