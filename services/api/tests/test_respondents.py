import asyncio
from types import SimpleNamespace as NS

from app.respondents import interview_all, simulate
from app.schemas import DiscussionGuide, Persona, PersonaDetails


def persona(name):
    return Persona(
        name=name, occupation="nurse", backstory="b", speaking_style="terse",
        age=30, gender="woman", city="Austin, TX", income="under $35k", household="lives alone",
    )


def script_answers(stub_llm, delay=0.0):
    """Stub the plain-text path. Each reply is unique, and we track how many calls overlap."""
    state = {"count": 0, "in_flight": 0, "max_in_flight": 0}

    async def create(model, messages):
        stub_llm.calls.append(messages)
        state["count"] += 1
        reply = f"answer {state['count']}"
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        await asyncio.sleep(delay)
        state["in_flight"] -= 1
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(content=reply))],
        )

    stub_llm.create = create
    return state


def test_each_persona_is_interviewed_in_one_growing_conversation(stub_llm, ledger):
    script_answers(stub_llm)
    personas = [persona("Ann"), persona("Bo")]

    answers = asyncio.run(interview_all(personas, ["q1", "q2", "q3"], "m", concurrency=5))

    assert len(ledger) == 6  # 2 personas x 3 questions
    for person, person_answers in zip(personas, answers):
        rows = [r for r in ledger if r["prompt"].inputs["persona"]["name"] == person.name]
        roles = [[m["role"] for m in r["messages"]] for r in rows]
        assert roles == [
            ["system", "user"],
            ["system", "user", "assistant", "user"],
            ["system", "user", "assistant", "user", "assistant", "user"],
        ]
        last = rows[-1]["messages"]
        assert [m["content"] for m in last[1::2]] == ["q1", "q2", "q3"]
        assert [m["content"] for m in last[2:5:2]] == person_answers[:2]  # earlier answers are in the history
        assert person.name in last[0]["content"]  # the system prompt describes this persona


def test_no_more_calls_in_flight_than_the_concurrency_limit(stub_llm):
    state = script_answers(stub_llm, delay=0.01)
    personas = [persona(f"P{i}") for i in range(8)]

    asyncio.run(interview_all(personas, ["q1", "q2"], "m", concurrency=3))

    assert state["count"] == 16
    assert state["max_in_flight"] == 3  # reaches the limit but never passes it


def test_a_higher_limit_allows_more_parallelism(stub_llm):
    state = script_answers(stub_llm, delay=0.01)
    personas = [persona(f"P{i}") for i in range(8)]

    asyncio.run(interview_all(personas, ["q1"], "m", concurrency=8))

    assert state["max_in_flight"] == 8


def test_simulation_runs_guide_then_personas_then_answers(stub_llm):
    guide = DiscussionGuide.model_validate(
        {
            "objectives": ["o"],
            "questions": [{"text": f"Question {n}?", "probes": ["p"]} for n in range(1, 5)],
        }
    )

    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        if response_format is DiscussionGuide:
            parsed = guide
        else:
            parsed = PersonaDetails(name="N M", occupation="o", backstory="b", speaking_style="s")
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=parsed))],
        )

    stub_llm.parse = parse
    script_answers(stub_llm)

    sim = asyncio.run(simulate("a brief", count=3, seed=0, model="m", concurrency=4, max_questions=2))

    assert sim.questions == ["Question 1?", "Question 2?"]  # capped at max_questions
    assert len(sim.personas) == 3
    assert [len(a) for a in sim.answers] == [2, 2, 2]
    assert len(stub_llm.calls) == 1 + 3 + 6  # guide + personas + answers
