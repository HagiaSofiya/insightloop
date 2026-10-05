from typing import Literal

import asyncpg
from pydantic import BaseModel

from app.db import connect
from app.embeddings import EMBEDDING_MODEL
from app.llm.client import complete_structured, embed
from app.prompts.registry import PromptRef, load
from app.rerank import rerank as rerank_hits
from app.retrieval import CANDIDATES, hybrid_candidates
from app.schemas import Answer, Citation
from app.search import format_evidence, nearest  # noqa: F401  (format_evidence is re-exported)
from app.verifier import verify_citations

PROMPT_NAME = "ask"
PROMPT_VERSION = "v1"
ANSWER_MODEL = "gpt-4o-mini"
DEFAULT_K = 8

# Responses farther than this (cosine distance) are not treated as evidence. Calibrated on 16 questions
# against text-embedding-3-small and the three meal-kit, bank and sneaker studies: every question the data
# could answer had its best match at 0.39 or closer, and every one it could not answer at 0.62 or farther.
# Redo the calibration if the embedding model or the kind of data changes.
MAX_DISTANCE = 0.50


class AskResponse(BaseModel):
    status: Literal["answered", "not_enough_evidence"]
    answer: str | None
    citations: list[Citation]
    reason: Literal["retrieval_too_weak", "model_found_no_answer"] | None
    best_distance: float | None  # the closest response by meaning, whether or not it was close enough
    used: int  # how many responses were given to the model
    verified: bool | None = None  # whether every cited quote appears word for word in its response (None if not answered)
    problems: list[str] = []  # why it failed verification


def _not_enough(reason: str, best: float | None, used: int = 0) -> AskResponse:
    return AskResponse(
        status="not_enough_evidence", answer=None, citations=[], reason=reason, best_distance=best, used=used
    )


async def ask(
    conn: asyncpg.Connection,
    question: str,
    study_id: int | None = None,
    k: int = DEFAULT_K,
    embedding_model: str = EMBEDDING_MODEL,
    answer_model: str = ANSWER_MODEL,
    max_distance: float = MAX_DISTANCE,
    mode: Literal["vector", "hybrid"] = "hybrid",
    rerank: bool = False,
) -> AskResponse:
    vector = (await embed([question], embedding_model)).vectors[0]
    if mode == "vector":
        hits = await nearest(conn, vector, embedding_model, k, study_id)
        best = hits[0].distance if hits else None
    else:
        found = await hybrid_candidates(conn, question, vector, embedding_model, study_id, max(k, CANDIDATES))
        best = found.best_distance
        hits = await rerank_hits(question, found.hits, k) if rerank else found.hits[:k]

    # A response is evidence if it is close in meaning, or if it contains the question's key words.
    evidence = [hit for hit in hits if hit.distance <= max_distance or hit.keyword_match]
    if not evidence:
        return _not_enough("retrieval_too_weak", best)  # no answering call at all

    messages = [
        {"role": "system", "content": load(PROMPT_NAME, PROMPT_VERSION)},
        {"role": "user", "content": f"Question: {question}\n\nEvidence:\n\n{format_evidence(evidence)}"},
    ]
    prompt = PromptRef(
        PROMPT_NAME, PROMPT_VERSION,
        {"question": question, "study_id": study_id, "response_ids": [hit.response_id for hit in evidence]},
    )
    result: Answer = await complete_structured(messages, answer_model, Answer, prompt)
    if not result.answerable:
        return _not_enough("model_found_no_answer", best, used=len(evidence))
    verification = await verify_citations(conn, result.citations)
    return AskResponse(
        status="answered", answer=result.answer, citations=result.citations,
        reason=None, best_distance=best, used=len(evidence),
        verified=verification.verified,
        problems=[f"{p.claim}: {p.detail}" for p in verification.problems],
    )


async def answer_question(
    question: str,
    study_id: int | None = None,
    k: int = DEFAULT_K,
    mode: Literal["vector", "hybrid"] = "hybrid",
    rerank: bool = False,
) -> AskResponse:
    conn = await connect()
    try:
        return await ask(conn, question, study_id, k, mode=mode, rerank=rerank)
    finally:
        await conn.close()


def format_ask(response: AskResponse) -> str:
    best = f"{response.best_distance:.3f}" if response.best_distance is not None else "none"
    lines = [f"status: {response.status} ({response.used} responses used, best distance {best})"]
    if response.status == "answered":
        lines += ["", response.answer, "", "citations:"]
        lines += [f'  [R{c.response_id}] "{c.quote}"' for c in response.citations]
        lines.append("verified: every quote appears in its response" if response.verified else "UNVERIFIED:")
        lines += [f"  - {problem}" for problem in response.problems]
    else:
        lines += ["", f"The data does not answer this question ({response.reason})."]
    return "\n".join(lines)
