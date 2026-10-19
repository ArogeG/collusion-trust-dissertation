"""Load and validate an experiment config (configs/*.yaml)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    provider: Literal["ollama", "mock"] = "ollama"
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(400, gt=0)


class RunConfig(BaseModel):
    n_agents: int = Field(2, ge=2)
    rounds: int = Field(3, ge=1)
    seed: int = 42
    task_file: str
    out_dir: str = "data/dev"


class ScenarioConfig(BaseModel):
    type: Literal["normal", "single", "coalition"] = "normal"
    colluders: list[str] = []


class ExperimentConfig(BaseModel):
    experiment: str
    llm: LLMConfig
    run: RunConfig
    scenario: ScenarioConfig = ScenarioConfig()

    @model_validator(mode="after")
    def _check_scenario(self) -> "ExperimentConfig":
        ids = {f"a{i}" for i in range(1, self.run.n_agents + 1)}
        unknown = set(self.scenario.colluders) - ids
        if unknown:
            raise ValueError(f"colluders {sorted(unknown)} are not agents {sorted(ids)}")
        if self.scenario.type == "normal" and self.scenario.colluders:
            raise ValueError("a normal scenario cannot have colluders")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = ExperimentConfig.model_validate(data)
    # Docker (and anyone else) can point at a different model server.
    if os.environ.get("LLM_BASE_URL"):
        cfg.llm.base_url = os.environ["LLM_BASE_URL"]
    return cfg


def load_task(path: str | Path) -> dict:
    task = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("task_id", "prompt", "options", "correct_answer"):
        if key not in task:
            raise ValueError(f"task file {path} is missing '{key}'")
    return task
