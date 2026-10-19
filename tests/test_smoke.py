"""End-to-end smoke tests: a full run writes a schema-valid, reproducible log."""
import asyncio
import json
from pathlib import Path

import pytest

from src.sim.config import load_config
from src.sim.logger import validate_log
from src.sim.orchestrator import turn_order
from src.sim.run import main, run_trial

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "small.yaml"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    c = load_config(CONFIG)
    c.run.out_dir = str(tmp_path)
    return c


def read_events(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_turn_order_is_seeded():
    ids = ["a1", "a2", "a3", "a4", "a5"]
    assert turn_order(42, 1, ids) == turn_order(42, 1, ids)
    assert sorted(turn_order(42, 1, ids)) == ids
    orders = {tuple(turn_order(42, r, ids)) for r in range(1, 6)}
    assert len(orders) > 1  # order changes between rounds


def test_mock_run_writes_valid_log(cfg):
    cfg.llm.provider = "mock"
    path = asyncio.run(run_trial(cfg, verbose=False))
    summary = validate_log(path)
    assert summary == {"events": 6, "parse_rate": 1.0, "rounds": 3}


def test_same_seed_same_conversation(cfg, tmp_path):
    cfg.llm.provider = "mock"
    a = read_events(asyncio.run(run_trial(cfg, verbose=False)))
    cfg.run.out_dir = str(tmp_path / "second")
    b = read_events(asyncio.run(run_trial(cfg, verbose=False)))
    strip = lambda evs: [(e["sender"], e["position"], e["confidence"]) for e in evs[1:-1]]
    assert strip(a) == strip(b)


def test_openai_compatible_path(cfg):
    """The real (Ollama) code path, against a local stand-in server."""
    from tests.fake_openai_server import FakeOpenAIServer

    with FakeOpenAIServer(model=cfg.llm.model) as server:
        cfg.llm.base_url = server.base_url
        path = asyncio.run(run_trial(cfg, verbose=False))
        sent = server.requests
    assert validate_log(path)["parse_rate"] == 1.0
    assert len(sent) == 6
    assert all(r["model"] == "llama3.2:3b" and r["seed"] == 42 and r["temperature"] == 0.7
               for r in sent)
    # Later turns must carry the earlier conversation (shared transcript).
    assert len(sent[-1]["messages"]) > len(sent[0]["messages"])
    events = read_events(path)
    assert events[1]["prompt_tokens"] == 50


def test_cli_reports_unreachable_server(cfg, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(CONFIG)])
    assert "Cannot reach the model server" in str(exc.value)


def test_config_rejects_colluders_in_normal_scenario(tmp_path):
    bad = CONFIG.read_text().replace("colluders: []", "colluders: [a1]")
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="normal scenario"):
        load_config(p)
