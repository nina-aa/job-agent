"""HTTP demo of the same graph the CLI drives.

    uvicorn job_agent.api:app            # real Gemini, needs GOOGLE_API_KEY
    LLM_PROVIDER=fake uvicorn job_agent.api:app   # offline

Each run is one LangGraph thread. Start it, then answer each interrupt the graph raises:
POST /runs -> shortlist (pick) -> POST /runs/{id}/resume -> letter (review) -> resume ... -> done.
The checkpointer is in memory: runs are lost on restart and this needs a single server process.
Run it locally only; every request can spend your Gemini quota.
"""
import os
import pathlib
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from langgraph.types import Command
from pydantic import BaseModel

from .graph import build_graph
from .llm import LLMError, get_llm
from .search import search_jobs
from .transcript import render_markdown


class RunRequest(BaseModel):
    search_term: str = "python"
    location: str = "Europe"


class ResumeRequest(BaseModel):
    """Answer to the pending interrupt: `chosen` for a pick, `action` (+ `feedback`) for a review."""

    chosen: list[int] | None = None
    action: Literal["approve", "reject", "skip"] | None = None
    feedback: str = ""


class RunResponse(BaseModel):
    run_id: str
    status: Literal["waiting", "done"]
    pending: dict | None  # the interrupt payload: {"type": "pick", ...} or {"type": "review", ...}
    saved: list[str]
    usage: dict


def default_graph():
    read = lambda name, default: pathlib.Path(os.getenv(name, default)).read_text(encoding="utf-8")
    return build_graph(
        get_llm(),
        read("PROFILE_FILE", "profile.txt"),
        read("CONTACT_FILE", "contact.txt"),
        lambda term, location: search_jobs(term, location, pages=3),
    )


def create_app(graph=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.graph is None:
            app.state.graph = default_graph()
        yield

    app = FastAPI(title="job-agent", description=__doc__, lifespan=lifespan)
    app.state.graph = graph

    def config_for(run_id):
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 100}

    def snapshot(run_id):
        snap = app.state.graph.get_state(config_for(run_id))
        if not snap.values:
            raise HTTPException(404, f"unknown run '{run_id}': use the run_id string returned by POST /runs (your answer, like [0], goes in the body)")
        return snap

    def pending_of(snap):
        return next((i.value for task in snap.tasks for i in task.interrupts), None)

    def respond(run_id):
        snap = snapshot(run_id)
        pending = pending_of(snap)
        calls = snap.values.get("calls", [])
        usage = {
            "llm_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "llm_seconds": round(sum(c["seconds"] for c in calls), 2),
            "models": sorted({c["model"] for c in calls}),
        }
        return RunResponse(run_id=run_id, status="waiting" if pending else "done", pending=pending,
                           saved=snap.values.get("saved", []), usage=usage)

    def run_graph(run_id, payload):
        try:
            app.state.graph.invoke(payload, config_for(run_id))
        except LLMError as e:
            raise HTTPException(502, str(e))

    # Plain `def` endpoints: FastAPI runs them in a thread pool, so slow LLM calls don't block the server.
    @app.post("/runs", response_model=RunResponse)
    def start_run(body: RunRequest):
        run_id = uuid.uuid4().hex[:12]
        run_graph(run_id, {"search_term": body.search_term, "location": body.location})
        return respond(run_id)

    @app.post("/runs/{run_id}/resume", response_model=RunResponse)
    def resume_run(run_id: str, body: ResumeRequest):
        pending = pending_of(snapshot(run_id))
        if pending is None:
            raise HTTPException(409, "this run is not waiting for an answer")
        if pending["type"] == "pick":
            if body.chosen is None:
                raise HTTPException(422, "this run is waiting for a pick: send `chosen`, e.g. [0]")
            answer = body.chosen
        else:
            if body.action is None:
                raise HTTPException(422, "this run is waiting for a review: send `action` (approve/reject/skip)")
            answer = {"action": body.action, "feedback": body.feedback}
        run_graph(run_id, Command(resume=answer))
        return respond(run_id)

    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str):
        return respond(run_id)

    @app.get("/runs/{run_id}/transcript")
    def get_transcript(run_id: str, format: Literal["json", "markdown"] = "json"):
        events = snapshot(run_id).values.get("transcript", [])
        if format == "markdown":
            return PlainTextResponse(render_markdown(events), media_type="text/markdown")
        return events

    return app


app = create_app()
