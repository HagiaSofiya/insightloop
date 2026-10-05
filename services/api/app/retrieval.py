import re
from dataclasses import dataclass

import asyncpg

from app.embeddings import to_vector_literal
from app.search import Neighbor, nearest

CANDIDATES = 20  # how many results each list contributes before they are fused
RRF_CONSTANT = 60  # the standard value; a bigger number flattens the difference between ranks

# Words that make a question read naturally but that respondents' answers rarely contain. Postgres
# removes ordinary English stop words by itself ("what", "do", "about"), so only survey words go here.
BOILERPLATE = {
    "respondent", "respondents", "participant", "participants", "people", "customer", "customers",
    "user", "users", "mention", "mentions", "mentioned", "say", "says", "said", "think", "thinks",
    "feel", "feels", "talk", "talks", "any", "some", "someone", "anyone", "ever", "would", "could",
}


def keyword_terms(question: str) -> list[str]:
    """The content words of a question, in order and without repeats."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [w for w in dict.fromkeys(words) if len(w) >= 3 and w not in BOILERPLATE]


# A keyword hit counts as evidence only if it contains a word found in fewer than about 15% of all
# responses (ln(1 / 0.15)). Matching only common words like "shoe" or "app" says little.
MIN_KEYWORD_SCORE = 1.9


async def keyword_search(
    conn: asyncpg.Connection,
    question: str,
    k: int = CANDIDATES,
    study_id: int | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    gender: str | None = None,
) -> list[tuple[int, float]]:
    """(response id, score) for responses containing any content word of the question, best first.

    Postgres stems both sides, so "knees" matches "knee". The words are joined with OR because an answer
    rarely contains every word of a question. Each matched word adds its inverse document frequency,
    ln(N / responses containing it), so a rare word like "knee" counts far more than "shoe". The built-in
    ts_rank does not do this, which is the main gap between it and BM25. Counting word frequencies with
    ts_stat reads the whole table, so a larger system would precompute them.
    """
    terms = keyword_terms(question)
    if not terms:
        return []
    rows = await conn.fetch(
        """
        WITH terms AS (
            SELECT lexeme, ln((SELECT count(*) FROM responses)::float8 / greatest(coalesce(d.ndoc, 1), 1)) AS idf
            FROM unnest(tsvector_to_array(to_tsvector('english', $1))) AS lexeme
            LEFT JOIN ts_stat('SELECT answer_tsv FROM responses') d ON d.word = lexeme
        )
        SELECT s.id, sum(terms.idf) AS score
        FROM responses s
        JOIN respondents r ON r.id = s.respondent_id
        JOIN terms ON s.answer_tsv @@ plainto_tsquery('simple', terms.lexeme)
        WHERE ($3::bigint IS NULL OR r.study_id = $3)
          AND ($4::int IS NULL OR r.age >= $4)
          AND ($5::int IS NULL OR r.age <= $5)
          AND ($6::text IS NULL OR r.gender = $6)
        GROUP BY s.id
        ORDER BY score DESC, s.id
        LIMIT $2
        """,
        " ".join(terms), k, study_id, min_age, max_age, gender,
    )
    return [(row["id"], row["score"]) for row in rows]


def reciprocal_rank_fusion(rankings: list[list[int]], constant: int = RRF_CONSTANT) -> list[int]:
    """Merge ranked lists. An id scores 1 / (constant + rank) in each list it appears in, and the scores add."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, response_id in enumerate(ranking, start=1):
            scores[response_id] = scores.get(response_id, 0.0) + 1 / (constant + position)
    return [response_id for response_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


async def hydrate(
    conn: asyncpg.Connection, response_ids: list[int], query_vector: list[float], model: str
) -> list[Neighbor]:
    """Full rows for these ids, in the order given, with their vector distance to the query."""
    if not response_ids:
        return []
    rows = await conn.fetch(
        """
        SELECT s.id AS response_id, st.id AS study_id, st.brief, s.question_number, s.question,
               s.answer, r.age, r.gender,
               CASE WHEN s.embedding_model = $2 THEN s.embedding <=> $1::vector END AS distance
        FROM responses s
        JOIN respondents r ON r.id = s.respondent_id
        JOIN studies st ON st.id = r.study_id
        WHERE s.id = ANY($3)
        """,
        to_vector_literal(query_vector), model, response_ids,
    )
    by_id = {
        row["response_id"]: Neighbor(
            # a response without a vector from this model counts as unrelated (1.0)
            1.0 if row["distance"] is None else row["distance"],
            row["study_id"], row["brief"], row["question_number"], row["answer"], row["age"],
            row["response_id"], row["question"], row["gender"],
        )
        for row in rows
    }
    return [by_id[response_id] for response_id in response_ids if response_id in by_id]


@dataclass
class Candidates:
    hits: list[Neighbor]  # the fused ranking, best first
    best_distance: float | None  # the closest vector match, whether or not it is close enough


async def hybrid_candidates(
    conn: asyncpg.Connection,
    question: str,
    query_vector: list[float],
    model: str,
    study_id: int | None = None,
    n: int = CANDIDATES,
    min_age: int | None = None,
    max_age: int | None = None,
    gender: str | None = None,
) -> Candidates:
    """Vector search and keyword search, fused. Each hit says whether the keyword search also found it."""
    vector_hits = await nearest(conn, query_vector, model, n, study_id, min_age, max_age, gender)
    keyword_hits = await keyword_search(conn, question, n, study_id, min_age, max_age, gender)
    keyword_ids = [response_id for response_id, _ in keyword_hits]
    strong = {response_id for response_id, score in keyword_hits if score >= MIN_KEYWORD_SCORE}
    fused = reciprocal_rank_fusion([[hit.response_id for hit in vector_hits], keyword_ids])[:n]

    by_id = {hit.response_id: hit for hit in vector_hits}
    missing = [response_id for response_id in fused if response_id not in by_id]
    by_id.update({hit.response_id: hit for hit in await hydrate(conn, missing, query_vector, model)})

    hits = []
    for response_id in fused:
        hit = by_id[response_id]
        hit.keyword_match = response_id in strong
        hits.append(hit)
    return Candidates(hits, vector_hits[0].distance if vector_hits else None)
