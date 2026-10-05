from app.llm.client import complete_structured
from app.prompts.registry import PromptRef, load
from app.schemas import Reranking
from app.search import Neighbor, format_evidence

PROMPT_NAME = "rerank"
PROMPT_VERSION = "v1"
RERANK_MODEL = "gpt-4o-mini"


async def rerank(
    question: str,
    hits: list[Neighbor],
    top_n: int,
    model: str = RERANK_MODEL,
    min_score: int = 1,
) -> list[Neighbor]:
    """Ask the model how relevant each candidate is, drop the irrelevant ones and keep the best top_n.

    Candidates the model did not score count as 0. Ties keep their earlier (fused) order.
    """
    if not hits:
        return []
    messages = [
        {"role": "system", "content": load(PROMPT_NAME, PROMPT_VERSION)},
        {"role": "user", "content": f"Question: {question}\n\nAnswers:\n\n{format_evidence(hits)}"},
    ]
    prompt = PromptRef(PROMPT_NAME, PROMPT_VERSION, {"question": question, "response_ids": [h.response_id for h in hits]})
    result: Reranking = await complete_structured(messages, model, Reranking, prompt)

    scores = {item.response_id: max(0, min(3, item.score)) for item in result.scores}  # keep scores in 0 to 3
    ranked = sorted(enumerate(hits), key=lambda pair: (-scores.get(pair[1].response_id, 0), pair[0]))
    return [hit for _, hit in ranked if scores.get(hit.response_id, 0) >= min_score][:top_n]
