"""Round-based orchestrator (plan section 5.1).

Each round, every agent takes one turn in a seeded random order. Before its
turn an agent receives every message posted since its previous turn, so all
agents see the same shared conversation. Turn order is controlled by the seed,
not by an LLM, so runs are reproducible.
"""
from __future__ import annotations

import random

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from .logger import Footer, MessageEvent, RunLogger, utc_now
from .parse import parse_position


def turn_order(seed: int, round_no: int, ids: list[str]) -> list[str]:
    return random.Random(f"{seed}-round-{round_no}").sample(ids, k=len(ids))


def task_message(task: dict) -> str:
    lines = [f"TASK {task['task_id']}: {task['prompt']}"]
    for name, facts in (task.get("context") or {}).items():
        lines.append(f"- {name}: {facts}")
    lines.append(f"Options: {', '.join(task['options'])}")
    return "\n".join(lines)


async def run_conversation(agents: dict[str, AssistantAgent], task: dict, rounds: int, seed: int,
                           run_id: str, logger: RunLogger, verbose: bool = True) -> Footer:
    ids = list(agents)
    transcript: list[TextMessage] = []
    last_seen = {aid: 0 for aid in ids}
    final: dict[str, str | None] = {aid: None for aid in ids}
    turn = 0
    parsed_ok = 0
    opening = TextMessage(source="moderator", content=task_message(task))

    for round_no in range(1, rounds + 1):
        for aid in turn_order(seed, round_no, ids):
            new = transcript[last_seen[aid]:]
            inbox = ([opening] if round_no == 1 and last_seen[aid] == 0 else []) + new
            if not inbox:  # first speaker of round 1 with nothing yet: just the task
                inbox = [opening]
            response = await agents[aid].on_messages(inbox, CancellationToken())
            text = str(response.chat_message.to_text())
            usage = response.chat_message.models_usage

            turn += 1
            pos = parse_position(text, task["options"])
            parsed_ok += pos["parsed"]
            if pos["position"] is not None:
                final[aid] = pos["position"]

            logger.write(MessageEvent(
                type="message", run_id=run_id, round=round_no, turn=turn, ts=utc_now(),
                sender=aid, recipients=[a for a in ids if a != aid],
                reply_to=turn - 1 if turn > 1 else None, text=text,
                position=pos["position"], confidence=pos["confidence"], cites=pos["cites"],
                parsed=pos["parsed"],
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            ))
            if verbose:
                flag = "ok " if pos["parsed"] else "BAD"
                print(f"  round {round_no} turn {turn:>2} {aid}: [{flag}] position={pos['position']}"
                      f" conf={pos['confidence']}")

            transcript.append(TextMessage(source=aid, content=text))
            last_seen[aid] = len(transcript)

    footer = Footer(type="footer", run_id=run_id, finished=utc_now(), turns=turn,
                    parse_rate=round(parsed_ok / turn, 3) if turn else 0.0, final_positions=final)
    logger.write(footer)
    return footer
