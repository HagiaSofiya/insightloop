import asyncio
import re
from collections import Counter
from types import SimpleNamespace as NS

from app.personas import (
    CITIES,
    INCOMES,
    INITIALS,
    STYLES,
    fields_for,
    generate_personas,
    households_for,
    repetition_summary,
    sample_profiles,
)
from app.schemas import Persona, PersonaDetails


def test_same_seed_gives_the_same_profiles_and_a_different_seed_does_not():
    assert sample_profiles(30, seed=1) == sample_profiles(30, seed=1)
    assert sample_profiles(30, seed=1) != sample_profiles(30, seed=2)


def test_ages_cover_the_whole_range():
    ages = [p["age"] for p in sample_profiles(30, seed=0)]

    assert min(ages) <= 22 and max(ages) >= 70
    assert all(18 <= age <= 76 for age in ages)


def test_cities_and_incomes_are_balanced():
    profiles = sample_profiles(36, seed=0)

    cities = Counter(p["city"] for p in profiles)
    incomes = Counter(p["income"] for p in profiles)

    assert set(cities) == set(CITIES) and set(cities.values()) == {36 // len(CITIES)}
    assert set(incomes) == set(INCOMES) and set(incomes.values()) == {36 // len(INCOMES)}


def test_households_fit_the_age():
    assert "family with young kids" not in households_for(70)
    assert "family with teens" not in households_for(19)
    assert "lives with parents" in households_for(19)
    for profile in sample_profiles(200, seed=3):
        assert profile["household"] in households_for(profile["age"])


def test_fields_and_styles_are_spread_and_fit_the_age():
    profiles = sample_profiles(60, seed=0)

    styles = Counter(p["style"] for p in profiles)
    assert set(styles) == set(STYLES) and set(styles.values()) == {60 // len(STYLES)}
    assert len({p["field"] for p in profiles}) >= 8
    for p in profiles:
        assert p["field"] in fields_for(p["age"])
        assert p["first_initial"] in INITIALS and p["last_initial"] in INITIALS
    assert "retired" not in fields_for(40) and "student" not in fields_for(40)
    assert "retired" in fields_for(70) and "student" in fields_for(20)


def test_repetition_summary_counts_repeated_surnames_and_job_words():
    def person(name, job):
        return Persona(
            name=name, occupation=job, backstory="b", speaking_style="s",
            age=30, gender="woman", city="c", income="i", household="h",
        )

    people = [
        person("Ann Thompson", "Marketing Manager"),
        person("Bo Thompson", "Marketing Associate at a tech firm"),
        person("Cy Lopez", "Electrician"),
    ]

    surnames, jobs = repetition_summary(people)

    assert surnames == "surnames: 2 distinct of 3 (most common: Thompson x2)"
    assert jobs == 'job titles: most repeated word is "marketing", in 2 of 3'


def test_the_older_prompt_version_still_works_with_the_extra_inputs(stub_llm):
    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        details = PersonaDetails(name="N M", occupation="o", backstory="b", speaking_style="s")
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=details))],
        )

    stub_llm.parse = parse

    personas = asyncio.run(generate_personas("a brief", count=2, seed=0, model="m", version="v1"))

    assert len(personas) == 2
    assert "Work field" not in stub_llm.calls[0][0]["content"]  # v1 does not mention the new fields


def test_v2_prompt_carries_the_assigned_field_style_and_initials(stub_llm):
    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        details = PersonaDetails(name="N M", occupation="o", backstory="b", speaking_style="s")
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=details))],
        )

    stub_llm.parse = parse
    profile = sample_profiles(1, seed=0)[0]

    asyncio.run(generate_personas("a brief", count=1, seed=0, model="m"))

    system = stub_llm.calls[0][0]["content"]
    assert f"Work field: {profile['field']}" in system
    assert f"Speaking style: {profile['style']}" in system
    assert f"starts with {profile['first_initial']}" in system


def test_personas_combine_the_sampled_profile_with_the_model_details(stub_llm, ledger):
    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        age = re.search(r"Age: (\d+)", messages[0]["content"]).group(1)
        details = PersonaDetails(
            name=f"Person {age}", occupation="nurse", backstory="b", speaking_style="terse"
        )
        return NS(
            usage=NS(prompt_tokens=10, completion_tokens=5),
            choices=[NS(message=NS(refusal=None, parsed=details))],
        )

    stub_llm.parse = parse

    personas = asyncio.run(generate_personas("a brief", count=5, seed=0, model="m"))

    profiles = sample_profiles(5, seed=0)
    assert len(personas) == 5 and len(stub_llm.calls) == 5
    for persona, profile in zip(personas, profiles):
        assert persona.age == profile["age"]
        assert persona.city == profile["city"]
        assert persona.name == f"Person {profile['age']}"  # details came from this profile's call
    assert {row["prompt"].name for row in ledger} == {"persona"}
    assert ledger[0]["prompt"].inputs["brief"] == "a brief"
