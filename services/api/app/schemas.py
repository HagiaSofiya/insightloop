from pydantic import BaseModel, Field, model_validator


class GuideQuestion(BaseModel):
    text: str = Field(min_length=1)
    probes: list[str] = Field(min_length=1)


class DiscussionGuide(BaseModel):
    objectives: list[str] = Field(min_length=1)
    questions: list[GuideQuestion] = Field(min_length=1)


class PersonaDetails(BaseModel):
    """What the model adds to a demographic profile."""

    name: str = Field(min_length=1)
    occupation: str = Field(min_length=1)
    backstory: str = Field(min_length=1)
    speaking_style: str = Field(min_length=1)


class Persona(PersonaDetails):
    age: int
    gender: str
    city: str
    income: str
    household: str


class Citation(BaseModel):
    response_id: int
    quote: str = Field(min_length=1)  # words copied exactly from that response


class Answer(BaseModel):
    answerable: bool
    answer: str
    citations: list[Citation]

    @model_validator(mode="after")
    def an_answer_needs_citations(self):
        if self.answerable and not self.citations:
            raise ValueError("an answerable question must cite at least one response")
        return self


class Relevance(BaseModel):
    response_id: int
    score: int  # 0 not relevant, 1 loosely related, 2 partly answers, 3 directly answers


class Reranking(BaseModel):
    scores: list[Relevance]


class ThemeLabel(BaseModel):
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GroupLabel(BaseModel):
    group: int
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GroupLabels(BaseModel):
    groups: list[GroupLabel]

    @model_validator(mode="after")
    def labels_must_differ(self):
        names = [group.label.strip().lower() for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("every group needs its own label, but two groups share one")
        return self


# Structured-output schemas by name, so a stored call can be replayed.

class Judgement(BaseModel):
    reason: str  # written before the score, so the model reasons first
    score: int = Field(ge=1, le=5)



class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=4)


class Critique(BaseModel):
    ok: bool
    gaps: list[str] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _gaps_match_verdict(self):
        if self.ok and self.gaps:
            raise ValueError("ok is true but gaps were listed; list no gaps, or set ok to false")
        if not self.ok and not self.gaps:
            raise ValueError("ok is false but no gaps were listed; name at least one gap as a sub-question")
        return self


SCHEMAS: dict[str, type[BaseModel]] = {
    "DiscussionGuide": DiscussionGuide,
    "PersonaDetails": PersonaDetails,
    "Answer": Answer,
    "Reranking": Reranking,
    "ThemeLabel": ThemeLabel,
    "GroupLabels": GroupLabels,
    "Judgement": Judgement,
    "Plan": Plan,
    "Critique": Critique,
}

# The schema each prompt asks the model to fill, so prompts can be compared.
PROMPT_SCHEMAS: dict[str, type[BaseModel]] = {
    "discussion_guide": DiscussionGuide,
    "persona": PersonaDetails,
    "ask": Answer,
    "rerank": Reranking,
    "theme_label": GroupLabels,
    "judge": Judgement,
    "planner": Plan,
    "critic": Critique,
}
