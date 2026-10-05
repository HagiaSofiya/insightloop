# USD per 1M tokens as (input, output). Prices change, so check
# https://developers.openai.com/api/docs/pricing before trusting the numbers.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the estimated cost in USD, or None if the model has no known price."""
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price


# USD per 1M input tokens. Embedding models have no output tokens.
EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def estimate_embedding_cost(model: str, tokens: int) -> float | None:
    if model not in EMBEDDING_PRICES:
        return None
    return tokens / 1_000_000 * EMBEDDING_PRICES[model]
