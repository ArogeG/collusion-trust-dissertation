"""Run one trial from a config file.

    python -m src.sim.run --config configs/small.yaml          # real model via Ollama
    python -m src.sim.run --config configs/small.yaml --mock   # no model needed
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .agents import build_agents
from .config import ExperimentConfig, load_config, load_task
from .logger import Header, RunLogger, git_commit, utc_now, validate_log
from .orchestrator import run_conversation


def check_model_server(cfg: ExperimentConfig) -> None:
    """Fail early with a clear message if the model server is not reachable."""
    url = cfg.llm.base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"Cannot reach the model server at {url} ({exc}).\n"
            "  - Is Ollama running? Start the app, or run: ollama serve\n"
            "  - Inside Docker, LLM_BASE_URL should be http://host.docker.internal:11434/v1\n"
            "  - To test the pipeline without a model, add --mock"
        )
    if cfg.llm.model not in body:
        sys.exit(f"Model '{cfg.llm.model}' is not installed. Run: ollama pull {cfg.llm.model}")


async def run_trial(cfg: ExperimentConfig, verbose: bool = True) -> Path:
    task = load_task(cfg.run.task_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{cfg.experiment}_{task['task_id']}_s{cfg.run.seed}_{stamp}"
    path = Path(cfg.run.out_dir) / f"{run_id}.jsonl"

    agents = build_agents(cfg, task)
    logger = RunLogger(path)
    try:
        logger.write(Header(
            type="header", run_id=run_id, started=utc_now(), git_commit=git_commit(),
            seed=cfg.run.seed, model=cfg.llm.model, provider=cfg.llm.provider,
            task_id=task["task_id"], scenario=cfg.scenario.type,
            colluders=cfg.scenario.colluders, agents=list(agents), config=cfg.model_dump(),
        ))
        if verbose:
            print(f"Run {run_id}: {len(agents)} agents x {cfg.run.rounds} rounds, "
                  f"model={cfg.llm.provider}:{cfg.llm.model}")
        await run_conversation(agents, task, cfg.run.rounds, cfg.run.seed, run_id, logger, verbose)
    finally:
        logger.close()
        for agent in agents.values():
            await agent._model_client.close()  # noqa: SLF001 (no public accessor)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one simulation trial.")
    ap.add_argument("--config", required=True, help="path to configs/*.yaml")
    ap.add_argument("--mock", action="store_true", help="use canned replies instead of a model")
    ap.add_argument("--seed", type=int, help="override run.seed")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.mock:
        cfg.llm.provider = "mock"
    if args.seed is not None:
        cfg.run.seed = args.seed
    if cfg.llm.provider == "ollama":
        check_model_server(cfg)

    path = asyncio.run(run_trial(cfg, verbose=not args.quiet))
    summary = validate_log(path)
    print(f"\nLog: {path}")
    print(f"Schema valid: {summary['events']} events over {summary['rounds']} rounds; "
          f"POSITION block parsed in {summary['parse_rate']:.0%} of turns.")
    if summary["parse_rate"] < 1.0:
        print("WARNING: some turns had no valid POSITION block. Read those turns in the log; "
              "a model that ignores the format is a stage-1 failure (plan section 3).")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
