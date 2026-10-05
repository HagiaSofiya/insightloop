import asyncio
from dataclasses import dataclass

from app.llm.client import complete, complete_structured
from app.personas import DEFAULT_VERSION as PERSONA_VERSION
from app.personas import generate_personas
from app.prompts.registry import PromptRef, build_messages, load
from app.schemas import DiscussionGuide, Persona

PROMPT_NAME = "respondent"
PROMPT_VERSION = "v1"


@dataclass
class Simulation:
    questions: list[str]
    personas: list[Persona]
    answers: list[list[str]]  # answers[i][j] is persona i's answer to question j


async def interview(
    persona: Persona, questions: list[str], model: str, semaphore: asyncio.Semaphore
) -> list[str]:
    """Ask one persona every question in order, in a single growing conversation."""
    system = load(PROMPT_NAME, PROMPT_VERSION, **persona.model_dump())
    messages = [{"role": "system", "content": system}]
    answers = []
    for question in questions:
        messages.append({"role": "user", "content": question})
        prompt = PromptRef(
            PROMPT_NAME, PROMPT_VERSION, {"persona": persona.model_dump(), "question": question}
        )
        async with semaphore:  # at most `concurrency` API calls in flight across all personas
            result = await complete(list(messages), model, prompt)
        messages.append({"role": "assistant", "content": result.text})
        answers.append(result.text)
    return answers


async def interview_all(
    personas: list[Persona], questions: list[str], model: str, concurrency: int
) -> list[list[str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(
        await asyncio.gather(*(interview(p, questions, model, semaphore) for p in personas))
    )


async def simulate(
    brief: str,
    count: int,
    seed: int,
    model: str,
    concurrency: int,
    max_questions: int = 6,
    guide_version: str = "v2",
    persona_version: str = PERSONA_VERSION,
) -> Simulation:
    """Brief to discussion guide to personas to answers. Held in memory for now."""
    inputs = {"brief": brief}
    guide = await complete_structured(
        build_messages("discussion_guide", guide_version, inputs),
        model,
        DiscussionGuide,
        PromptRef("discussion_guide", guide_version, inputs),
    )
    questions = [q.text for q in guide.questions[:max_questions]]
    personas = await generate_personas(brief, count, seed, model, persona_version)
    answers = await interview_all(personas, questions, model, concurrency)
    return Simulation(questions, personas, answers)


def format_simulation(sim: Simulation, elapsed: float, shown: int = 3) -> str:
    total = len(sim.personas) * len(sim.questions)
    lines = [
        f"{len(sim.personas)} respondents x {len(sim.questions)} questions = {total} answers "
        f"(plus guide and personas) in {elapsed:.1f}s",
        "",
        *[f"Q{n}. {q}" for n, q in enumerate(sim.questions, 1)],
    ]
    for persona, answers in list(zip(sim.personas, sim.answers))[:shown]:
        lines += ["", f"--- {persona.name}, {persona.age}, {persona.city} | {persona.speaking_style}"]
        lines += [f"Q{n}: {a}" for n, a in enumerate(answers, 1)]
    return "\n".join(lines)
