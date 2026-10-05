import logging
import random
from asyncio import sleep
from collections.abc import Awaitable, Callable
from typing import TypeVar

import openai

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 4  # the first try plus up to 3 retries
BASE_DELAY = 0.5  # seconds
MAX_DELAY = 8.0  # seconds

# A 429 with one of these means "out of credits", which waiting will not fix.
QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted"}


def is_retryable(error: Exception) -> bool:
    if isinstance(error, openai.RateLimitError):
        return error.type not in QUOTA_CODES and error.code not in QUOTA_CODES
    return isinstance(error, (openai.InternalServerError, openai.APIConnectionError))


async def with_retries(call: Callable[[], Awaitable[T]]) -> T:
    """Run `call`, retrying transient failures with exponential backoff and full jitter."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await call()
        except Exception as error:
            if not is_retryable(error) or attempt == MAX_ATTEMPTS - 1:
                raise
            # Full jitter: a random wait between 0 and a ceiling that doubles each time.
            delay = random.uniform(0, min(MAX_DELAY, BASE_DELAY * 2**attempt))
            logger.warning(
                "llm call failed with %s, retry %d of %d in %.2fs",
                type(error).__name__,
                attempt + 1,
                MAX_ATTEMPTS - 1,
                delay,
            )
            await sleep(delay)
