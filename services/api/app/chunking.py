import re

# A sentence ends at . ! or ? followed by a space and a capital letter or opening quote. Requiring the
# capital avoids most false breaks such as "e.g. the box". It is a rough rule, not a language parser.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BREAK.split(text.strip()) if part.strip()]


def sentence_chunks(text: str) -> list[str]:
    """One chunk per sentence."""
    return split_sentences(text)


def window_chunks(text: str, size: int = 3, stride: int = 2) -> list[str]:
    """Overlapping windows of `size` sentences that move forward by `stride`, so neighbors share one."""
    sentences = split_sentences(text)
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    starts = list(range(0, len(sentences) - size + 1, stride))
    if starts[-1] + size < len(sentences):
        starts.append(len(sentences) - size)  # make sure the last sentences are covered
    return [" ".join(sentences[start : start + size]) for start in starts]


STRATEGIES = {"sentence": sentence_chunks, "window": window_chunks}
