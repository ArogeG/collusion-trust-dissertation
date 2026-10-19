"""Extract the structured POSITION block every agent turn must end with.

Agents are told to finish each turn with one line:
    POSITION: {"position": "Vendor B", "confidence": 0.8, "cites": ["a2"]}
Agreement, escalation and pooling signals (plan section 5.2) are computed from
these fields, so no extra LLM call is needed to measure behaviour.
"""
from __future__ import annotations

import json
import re

_MARKER = re.compile(r"POSITION\s*:\s*", re.IGNORECASE)


def _first_json_object(text: str) -> dict | None:
    """Return the first balanced {...} object in text that parses as JSON."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_position(text: str, options: list[str] | None = None) -> dict:
    """Return {"position", "confidence", "cites", "parsed"} from an agent reply.

    Looks after the last "POSITION:" marker first, then anywhere in the text.
    If `options` is given, the position is normalised to the matching option
    (case-insensitive); an unknown position is kept but flagged invalid.
    """
    obj = None
    markers = list(_MARKER.finditer(text or ""))
    if markers:
        obj = _first_json_object(text[markers[-1].end() :])
    if obj is None:
        obj = _first_json_object(text or "")

    if obj is None or "position" not in obj:
        return {"position": None, "confidence": None, "cites": [], "parsed": False}

    position = str(obj.get("position")).strip()
    valid = True
    if options:
        match = [o for o in options if o.lower() == position.lower()]
        valid = bool(match)
        position = match[0] if match else position

    try:
        confidence = float(obj.get("confidence"))
        confidence = min(max(confidence, 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = None

    cites = obj.get("cites") or []
    if not isinstance(cites, list):
        cites = [cites]
    cites = [str(c) for c in cites]

    return {"position": position, "confidence": confidence, "cites": cites, "parsed": valid}
