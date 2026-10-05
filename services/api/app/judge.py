"""An LLM grades each answer from 1 to 5 against the known true facts, and we measure how well it agrees with a human.

The judge is a model, so it has its own failure modes: it can favor long answers, favor answers that sound
like its own, and drift between runs. That is why it is checked against hand labels and why the gate also
rests on metrics computed by plain code.
"""

import json
from pathlib import Path

from app.llm.client import complete_structured
from app.prompts.registry import PromptRef, load
from app.schemas import Citation, Judgement

PROMPT_NAME = "judge"
PROMPT_VERSION = "v1"
JUDGE_MODEL = "gpt-4o-mini"


async def judge_answer(
    question: str, answer: str, citations: list[Citation], reference: list[str], model: str = JUDGE_MODEL
) -> Judgement:
    quotes = "\n".join(f'- [R{c.response_id}] "{c.quote}"' for c in citations) or "(none)"
    facts = "\n".join(f"- {text}" for text in reference)
    messages = [
        {"role": "system", "content": load(PROMPT_NAME, PROMPT_VERSION)},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{answer}\n\nCited quotes:\n{quotes}\n\nReference (all on-topic responses):\n{facts}"},
    ]
    prompt = PromptRef(PROMPT_NAME, PROMPT_VERSION, {"question": question, "answer": answer})
    return await complete_structured(messages, model, Judgement, prompt)


def quality(score: int) -> float:
    """1 to 5 mapped to 0 to 1, so it sits beside the other metrics."""
    return (score - 1) / 4


def agreement(pairs: list[tuple[int, int]]) -> dict[str, float]:
    """(judge score, human score) pairs -> how often they match exactly, are within 1, and agree on pass (4 or more)."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no labeled cases")
    return {
        "n": n,
        "exact": sum(j == h for j, h in pairs) / n,
        "within_1": sum(abs(j - h) <= 1 for j, h in pairs) / n,
        "pass_fail": sum((j >= 4) == (h >= 4) for j, h in pairs) / n,
    }


def read_labels(path: Path) -> list[tuple[int, int]]:
    """The labeled rows of a hand-label sheet. Rows with no human score yet are skipped."""
    rows = json.loads(path.read_text())
    return [(row["judge_score"], row["human_score"]) for row in rows if row.get("human_score") is not None]
