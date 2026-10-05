import asyncio
import random
import re
from collections import Counter
from statistics import median

from app.llm.client import complete_structured
from app.prompts.registry import PromptRef, build_messages
from app.schemas import Persona, PersonaDetails

PROMPT_NAME = "persona"
DEFAULT_VERSION = "v2"

GENDERS = ["woman", "man", "non-binary"]
GENDER_WEIGHTS = [48, 48, 4]
CITIES = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Austin, TX", "Atlanta, GA",
    "Columbus, OH", "Denver, CO", "Seattle, WA", "Phoenix, AZ", "Boise, ID",
    "a small town in rural Iowa", "a suburb of Charlotte, NC",
]
INCOMES = ["under $35k", "$35k to $75k", "$75k to $125k", "over $125k"]
FIELDS = [
    "healthcare", "education", "skilled trades or construction", "retail or customer service",
    "restaurants or hospitality", "technology", "finance or insurance",
    "government or public service", "transportation or logistics", "creative or media work",
    "manufacturing or warehouse work", "self-employed or gig work",
    "caregiving or homemaking", "agriculture or outdoor work",
]
STYLES = [
    "terse and skeptical", "chatty and tangential", "polite but vague",
    "blunt with a dry sense of humor", "thoughtful and slow, qualifies everything",
    "enthusiastic and exaggerates", "guarded, short answers until warmed up",
    "rambling storyteller", "practical, talks in numbers and costs",
    "distracted, mentions other things going on",
]
INITIALS = "ABCDEFGHJKLMNPRSTVW"  # common starting letters, so names stay plausible


def households_for(age: int) -> list[str]:
    """Household types that make sense at this age."""
    options = ["lives alone"]
    if age < 26:
        options.append("lives with parents")
    if age < 40:
        options.append("lives with roommates")
    if age >= 22:
        options.append("couple with no kids")
    if 25 <= age <= 50:
        options.append("family with young kids")
    if 38 <= age <= 62:
        options.append("family with teens")
    if age >= 25:
        options.append("multigenerational household")
    return options


def fields_for(age: int) -> list[str]:
    """Work fields that make sense at this age."""
    options = list(FIELDS)
    if age <= 28:
        options.append("student")
    if age >= 62:
        options.append("retired")
    return options


def _balanced(values: list[str], count: int, rng: random.Random) -> list[str]:
    """Each value appears about equally often, in random order."""
    items = [values[i % len(values)] for i in range(count)]
    rng.shuffle(items)
    return items


def sample_profiles(count: int, seed: int) -> list[dict]:
    """Profiles chosen by code, so the spread is guaranteed and reproducible."""
    rng = random.Random(seed)
    ages = [round(18 + (i + rng.random()) * 57 / count) for i in range(count)]  # evenly spread, 18 to 75
    rng.shuffle(ages)
    cities = _balanced(CITIES, count, rng)
    incomes = _balanced(INCOMES, count, rng)
    styles = _balanced(STYLES, count, rng)
    genders = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=count)
    return [
        {
            "age": age,
            "gender": gender,
            "city": city,
            "income": income,
            "household": rng.choice(households_for(age)),
            "field": rng.choice(fields_for(age)),
            "style": style,
            "first_initial": rng.choice(INITIALS),
            "last_initial": rng.choice(INITIALS),
        }
        for age, gender, city, income, style in zip(ages, genders, cities, incomes, styles)
    ]


async def generate_persona(brief: str, profile: dict, model: str, version: str = DEFAULT_VERSION) -> Persona:
    """Fill in one sampled profile (name, job, backstory, speaking style) with the model."""
    inputs = {"brief": brief, **profile}
    messages = build_messages(PROMPT_NAME, version, inputs)
    prompt = PromptRef(PROMPT_NAME, version, inputs)
    details = await complete_structured(messages, model, PersonaDetails, prompt)
    return Persona(**profile, **details.model_dump())


async def generate_personas(
    brief: str, count: int, seed: int, model: str, version: str = DEFAULT_VERSION
) -> list[Persona]:
    return list(
        await asyncio.gather(*(generate_persona(brief, p, model, version) for p in sample_profiles(count, seed)))
    )


FUNCTION_WORDS = {"a", "an", "and", "at", "the", "of", "in", "for", "to", "with", "or", "on", "as", "part", "time"}


def repetition_summary(personas: list[Persona]) -> list[str]:
    """How much the model's free text repeats. Lower repetition means a more varied sample."""
    surnames = Counter(p.name.split()[-1] for p in personas)
    top_surname, top_surname_count = surnames.most_common(1)[0]
    job_words = Counter(
        word
        for p in personas
        for word in set(re.findall(r"[a-z]+", p.occupation.lower()))
        if word not in FUNCTION_WORDS
    )
    top_word, top_word_count = job_words.most_common(1)[0]
    return [
        f"surnames: {len(surnames)} distinct of {len(personas)} (most common: {top_surname} x{top_surname_count})",
        f'job titles: most repeated word is "{top_word}", in {top_word_count} of {len(personas)}',
    ]


def format_personas(personas: list[Persona]) -> str:
    lines = [
        f"{n:>2}. {p.name}, {p.age}, {p.gender}, {p.city} | {p.occupation} | {p.speaking_style}"
        for n, p in enumerate(personas, 1)
    ]
    ages = [p.age for p in personas]

    def spread(attribute: str) -> str:
        return ", ".join(f"{k} {v}" for k, v in Counter(getattr(p, attribute) for p in personas).most_common())

    lines += [
        "",
        f"ages {min(ages)} to {max(ages)}, median {median(ages):.0f}",
        f"genders: {spread('gender')}",
        f"incomes: {spread('income')}",
        f"households: {spread('household')}",
        f"cities: {len({p.city for p in personas})} different",
        *repetition_summary(personas),
    ]
    return "\n".join(lines)
