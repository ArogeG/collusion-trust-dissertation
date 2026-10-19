# Early Detection of Small-Scale Collusion in LLM Multi-Agent Systems Using Adaptive Trust Monitoring

Doctoral dissertation code: Gbola Atitebi, City University of Seattle.
Chair: Dr. Sherin Aly.

A controlled experiment that tests whether a lightweight adaptive trust monitor detects
2–3 colluding agents in an LLM multi-agent system faster and with fewer false positives
than a static baseline. Agents run on AutoGen; the model runs locally through Ollama.

> **Status:** stage 1 (smoke test) ready. Monitors, attack scenarios and analysis come next;
> see "What's built and what's next" below.

## Quick start (local development)

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
uv venv --python 3.12
source .venv/bin/activate              # Windows: .venv\Scripts\activate
uv pip install -r requirements.lock.txt

python -m pytest -q                    # 11 tests, no model needed
python -m src.sim.run --config configs/small.yaml --mock   # full pipeline, canned replies

ollama pull llama3.2:3b                # tiny model for plumbing (stage 1-2)
python -m src.sim.run --config configs/small.yaml          # real smoke test
```

A passing smoke test ends with:

```
Schema valid: 6 events over 3 rounds; POSITION block parsed in 100% of turns.
SMOKE TEST PASSED
```

If some turns have no valid `POSITION` block, the run exits with code 1. Read those turns in
the log under `data/dev/`: a model that ignores the format fails stage 1.

## Staged setup

| Stage | Config | Scale | Pass when |
| --- | --- | --- | --- |
| 1. Smoke test | `configs/small.yaml` | 2 agents, 3 rounds, 1 task | Schema valid, 100% POSITION blocks parsed |
| 2. One run per scenario | to add | 5 agents; normal, single, coalition | Transcripts read by hand and look right |
| 3. Mini grid | to add | 18 runs | Every Chapter 4 table and figure script runs |
| 4. Scale-up timing | to add | One 10-agent and one 20-agent run | Mean run time × 360 fits Weeks 6–8 |

Development logs go to `data/dev/` and are not study data.

## Docker (frozen study environment)

The container holds the code and pinned libraries; **Ollama stays on the host**.

```bash
docker compose build
docker compose run --rm sim python -m pytest -q
docker compose run --rm sim python -m src.sim.run --config configs/small.yaml --mock
docker compose run --rm sim                       # smoke test against Ollama on the host
```

- **Mac / Windows (Docker Desktop):** works as is; the container reaches Ollama at
  `host.docker.internal:11434`.
- **Linux:** Ollama listens only on 127.0.0.1 by default. Start it with
  `OLLAMA_HOST=0.0.0.0 ollama serve` so the container can reach it.

At the parameter freeze: `docker tag collusion-trust:dev collusion-trust:v1-frozen` and
`git tag v1-frozen`. Report both, plus the Ollama model digest (`ollama show <model>`), in Chapter 4.

## Project layout

```
configs/          one YAML file per stage (small.yaml now; pilot.yaml, main.yaml later)
data/tasks/       task bank (committed): prompts, options, correct answer, attacker target
data/dev/         development logs (ignored by Git)
data/raw/         study run logs, one JSONL per run, never edited (ignored by Git; back up weekly)
data/processed/   features and results tables (ignored by Git)
src/sim/          harness: config, agents, orchestrator, logger, run CLI
src/monitors/     static and adaptive trust monitors (to build)
src/analysis/     Chapter 4 statistics (to build)
tests/            unit and end-to-end tests
reports/          tables and figures for Chapter 4
docs/             parameter sheet, pre-registration, chair decisions
```

## How a run works

1. `src/sim/run.py` loads the config and task, and checks that the model server answers.
2. `src/sim/agents.py` builds N AutoGen `AssistantAgent`s (`a1`…`aN`) with rotating roles.
   Each must end every turn with
   `POSITION: {"position": "...", "confidence": 0-1, "cites": ["a2"]}`.
3. `src/sim/orchestrator.py` runs the rounds. Turn order is shuffled **by seed**, not by an
   LLM, and every agent receives all messages posted since its previous turn.
4. `src/sim/logger.py` writes one JSONL line per turn (header: config, seed, model, git commit;
   footer: parse rate, final positions) and validates the file against the schema.

The seed sets the turn order and is passed to the model, so the same seed gives the same turn
order. Exact model output can still vary a little between machines.

## What's built and what's next

| Plan section | Item | Status |
| --- | --- | --- |
| 3 | Environment, Docker, staged setup | Done |
| 4.1 | Task bank (60 main + 15 calibration) | 1 example task (`smoke_001`) |
| 4.2 | Agents with roles and POSITION block | Done |
| 4.3 | Single-agent and coalition scenarios | Hook ready (`hidden_instructions`); scripts to write |
| 4.5 | Log schema and validation | Done |
| 5.1 | Orchestrator, logger, runner | Single-trial runner done; batch runner to build |
| 5.2–5.4 | Signals, static and adaptive monitors | To build |
| 6 | Analysis | To build |

## Reproducibility rules

- One config file drives every run; the header of every log records the config, seed, model and git commit.
- Raw logs are never overwritten (the logger refuses) or edited.
- Study runs use `requirements.lock.txt` and the frozen Docker image only.
- Parameters are frozen and the analysis plan is pre-registered before main data collection.

## Ethics

Synthetic tasks and simulated agents only. No human subjects or personal data. The work is defensive:
it studies detection of behaviours already described in published literature.
