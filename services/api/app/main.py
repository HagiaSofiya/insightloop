import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import insights, research_jobs, workflow
from app.run_events import stream_events
from app.db import connect
from app.ask import DEFAULT_K, AskResponse, answer_question
from app.llm.client import LLMInvalidOutput, LLMRefusal, complete_structured
from app.prompts.registry import PromptRef, build_messages
from app.schemas import DiscussionGuide

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(title="Insightloop")
# The Next.js dev server runs on another port, so the browser needs permission to call this API (and open its event streams).
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:3100"], allow_methods=["GET", "POST"], allow_headers=["*"],
)

MODEL = "gpt-4o-mini"
GUIDE_PROMPT_NAME = "discussion_guide"
GUIDE_PROMPT_VERSION = "v1"


class BriefRequest(BaseModel):
    brief: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    study_id: int | None = None
    k: int = Field(default=DEFAULT_K, ge=1, le=20)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/briefs")
async def create_guide(request: BriefRequest) -> DiscussionGuide:
    inputs = {"brief": request.brief}
    messages = build_messages(GUIDE_PROMPT_NAME, GUIDE_PROMPT_VERSION, inputs)
    try:
        prompt = PromptRef(GUIDE_PROMPT_NAME, GUIDE_PROMPT_VERSION, inputs)
        return await complete_structured(messages, MODEL, DiscussionGuide, prompt)
    except LLMRefusal as refusal:
        raise HTTPException(status_code=422, detail=f"Model refused: {refusal}")
    except LLMInvalidOutput:
        raise HTTPException(
            status_code=502, detail="The model returned an invalid guide twice. Try again."
        )


@app.post("/ask")
async def ask_question(request: AskRequest) -> AskResponse:
    try:
        return await answer_question(request.question, request.study_id, request.k)
    except LLMRefusal as refusal:
        raise HTTPException(status_code=422, detail=f"Model refused: {refusal}")
    except LLMInvalidOutput:
        raise HTTPException(
            status_code=502, detail="The model returned an invalid answer twice. Try again."
        )


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.post("/research", status_code=202)
async def start_research(request: ResearchRequest) -> dict:
    """Queue a research question. The answer is produced by the worker; poll GET /research/{id}."""
    conn = await connect()
    try:
        return {"id": await research_jobs.submit(conn, request.question), "status": "queued"}
    finally:
        await conn.close()


@app.get("/research/{request_id}")
async def get_research(request_id: int) -> dict:
    conn = await connect()
    try:
        row = await research_jobs.get(conn, request_id)
    finally:
        await conn.close()
    if row is None:
        raise HTTPException(404, f"No research request with id {request_id}.")
    return {**dict(row), "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}


class WorkflowRequest(BaseModel):
    brief: str = Field(min_length=1, max_length=1000)
    respondents: int = Field(default=10, ge=workflow.MIN_RESPONDENTS, le=50)
    seed: int = 0
    review_guide: bool = Field(default=False, description="Pause after the guide is written so a person can edit it.")


class RetryRequest(BaseModel):
    from_step: str | None = Field(default=None, description=f"One of {workflow.STEPS}. Leave empty to continue after a failure.")


async def _workflow_call(call, *args) -> dict:
    """Run one control call on a fresh connection and return the run's new state. Wrong-state requests become 409."""
    conn = await connect()
    try:
        run_id = args[0]
        await call(conn, *args)
        return await workflow.describe(conn, run_id)
    except workflow.WorkflowNotFound as error:
        raise HTTPException(404, str(error))
    except workflow.WorkflowError as error:
        raise HTTPException(409, str(error))
    finally:
        await conn.close()


@app.post("/workflows", status_code=202)
async def start_workflow(request: WorkflowRequest) -> dict:
    """Start a full study run (guide, respondents, embed, themes, synthesize). The worker does the work."""
    conn = await connect()
    try:
        run_id = await workflow.start(conn, request.brief, request.respondents, request.seed, MODEL, request.review_guide)
        return {"id": run_id, "status": "pending"}
    finally:
        await conn.close()


@app.get("/workflows/{run_id}")
async def get_workflow(run_id: int) -> dict:
    conn = await connect()
    try:
        info = await workflow.describe(conn, run_id)
    finally:
        await conn.close()
    if info is None:
        raise HTTPException(404, f"No workflow run with id {run_id}.")
    return info


@app.post("/workflows/{run_id}/pause")
async def pause_workflow(run_id: int) -> dict:
    return await _workflow_call(workflow.pause, run_id)


@app.post("/workflows/{run_id}/resume", status_code=202)
async def resume_workflow(run_id: int) -> dict:
    return await _workflow_call(workflow.resume, run_id)


@app.post("/workflows/{run_id}/cancel")
async def cancel_workflow(run_id: int) -> dict:
    return await _workflow_call(workflow.cancel, run_id)


@app.post("/workflows/{run_id}/retry", status_code=202)
async def retry_workflow(run_id: int, request: RetryRequest | None = None) -> dict:
    return await _workflow_call(workflow.retry, run_id, request.from_step if request else None)


@app.get("/workflows/{run_id}/events")
async def workflow_events(run_id: int, request: Request, last_event_id: str | None = Header(default=None)) -> StreamingResponse:
    """Live progress as Server-Sent Events. A reconnecting browser sends Last-Event-ID automatically and
    receives only what it missed. The stream ends when the run is done, failed or cancelled."""
    try:
        last_id = int(last_event_id) if last_event_id else 0
    except ValueError:
        raise HTTPException(400, f"Last-Event-ID must be a whole number, got {last_event_id!r}.")
    conn = await connect()
    try:
        exists = await conn.fetchval("SELECT 1 FROM workflow_runs WHERE id = $1", run_id)
    finally:
        await conn.close()
    if not exists:
        raise HTTPException(404, f"No workflow run with id {run_id}.")
    return StreamingResponse(
        stream_events(connect, run_id, last_id, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no buffering by a proxy, or events arrive in lumps
    )


class GuideEdit(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=workflow.MAX_GUIDE_QUESTIONS)


@app.post("/workflows/{run_id}/guide", status_code=202)
async def approve_guide(run_id: int, edit: GuideEdit) -> dict:
    """Save the (possibly edited) guide questions and let a run that is waiting for review continue."""
    return await _workflow_call(workflow.approve_guide, run_id, edit.questions)


@app.get("/workflows")
async def list_workflows(limit: int = 50) -> list[dict]:
    conn = await connect()
    try:
        return await workflow.list_runs(conn, min(max(limit, 1), 200))
    finally:
        await conn.close()


async def _read(call, *args, missing: str) -> dict:
    conn = await connect()
    try:
        found = await call(conn, *args)
    finally:
        await conn.close()
    if found is None:
        raise HTTPException(404, missing)
    return found


@app.get("/studies/{study_id}/insights")
async def study_insights(study_id: int) -> dict:
    return await _read(insights.study_insights, study_id, missing=f"No study with id {study_id}.")


@app.get("/responses/{response_id}")
async def get_response(response_id: int) -> dict:
    """The source of a citation: one respondent's answer, with the question it answered."""
    return await _read(insights.source_response, response_id, missing=f"No response with id {response_id}.")


@app.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: int) -> dict:
    """An analyst's steps (tool calls and results) and its verifier outcome, for the reasoning panel."""
    return await _read(insights.agent_run, run_id, missing=f"No agent run with id {run_id}.")
