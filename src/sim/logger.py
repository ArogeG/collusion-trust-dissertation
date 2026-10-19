"""JSONL run logger and log schema (plan section 4.5).

One file per run. Line 1 is a header (config, seed, model, git commit);
then one line per agent turn; the last line is a footer. Raw logs are never
edited after they are written.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


class Header(BaseModel):
    type: Literal["header"]
    run_id: str
    started: str
    git_commit: str
    seed: int
    model: str
    provider: str
    task_id: str
    scenario: str
    colluders: list[str]
    agents: list[str]
    config: dict


class MessageEvent(BaseModel):
    type: Literal["message"]
    run_id: str
    round: int = Field(ge=1)
    turn: int = Field(ge=1)
    ts: str
    sender: str
    recipients: list[str]
    reply_to: int | None
    text: str
    position: str | None
    confidence: float | None
    cites: list[str]
    parsed: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Filled in later by the monitors (section 5.3-5.4).
    trust: dict[str, float] = {}
    flags: list[dict] = []


class Footer(BaseModel):
    type: Literal["footer"]
    run_id: str
    finished: str
    turns: int
    parse_rate: float
    final_positions: dict[str, str | None]


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"{path} exists; raw logs are never overwritten")
        self._fh = path.open("w", encoding="utf-8")

    def write(self, record: BaseModel) -> None:
        self._fh.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def validate_log(path: str | Path) -> dict:
    """Check every line against the schema. Returns a summary; raises on error."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("log has no events")
    Header.model_validate_json(lines[0])
    events = [MessageEvent.model_validate_json(l) for l in lines[1:-1]]
    footer = Footer.model_validate_json(lines[-1])
    turns = [e.turn for e in events]
    if turns != list(range(1, len(events) + 1)):
        raise ValueError("turn numbers are not consecutive")
    return {"events": len(events), "parse_rate": footer.parse_rate, "rounds": max(e.round for e in events)}
