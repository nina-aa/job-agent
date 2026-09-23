"""The LangGraph workflow: search -> score -> human picks -> draft/verify loop -> human review -> save."""
import operator
import pathlib
import re
from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .models import FitList, Verdict, VerifierOutput
from .transcript import event
from .prompts import draft_prompt, score_prompt, verify_prompt

MAX_SCORED = 12  # one batched scoring call; keeps the prompt small and free-tier friendly
MAX_REVISIONS = 2  # automatic verify -> redraft rounds per letter before a human sees it anyway
WORDS_MIN, WORDS_MAX = 250, 420  # the prompt asks for 280-380; small tolerance avoids pointless retries


class State(TypedDict, total=False):
    """The schema every node reads and writes. A node function never gets the whole State
    object to mutate -- it receives a read-only snapshot and RETURNS a small dict of just
    the fields it's changing (a "delta"). LangGraph merges that delta into the persisted
    state after the node returns. TypedDict just declares the field names/types for the
    editor and for LangGraph to introspect; a dataclass or Pydantic model would also work.

    Default merge rule for a field with no annotation: the delta REPLACES the old value.
    That's correct for `letter`, `verdict`, `job` -- you only ever want the latest one.
    """

    search_term: str
    location: str
    candidates: list[dict]
    shortlist: list[dict]
    queue: list[dict]
    job: dict
    letter: str
    verdict: dict
    attempts: int
    feedback: str  # verifier notes for the current letter
    guidance: str  # the human's redraft requests for this job, kept across every redraft
    action: str
    # These three use a REDUCER instead of the default replace-on-write rule.
    # Annotated[list[dict], operator.add] tells LangGraph: when a node returns a delta for
    # this field (e.g. score() returns {"calls": [record]}, a ONE-item list, not the running
    # total), concatenate it onto the existing list instead of overwriting it. Without this,
    # each node's single-record delta would wipe out every earlier node's contribution --
    # a node has no way to .append() to "the real list" because it never holds a reference
    # to it, only ever a delta going out.
    saved: Annotated[list[str], operator.add]
    calls: Annotated[list[dict], operator.add]  # one record per LLM call: tokens, seconds, model
    transcript: Annotated[list[dict], operator.add]  # the agent/human conversation, see transcript.py


def check_letter(llm, profile, job, letter):
    """LLM fact-check plus a deterministic length check. Returns (Verdict, call record)."""
    output, record = llm.complete("verify", verify_prompt(profile, job, letter), VerifierOutput)
    verdict = Verdict(unsupported_claims=[c.claim for c in output.checks if c.unsupported])
    words = len(letter.split())
    if not WORDS_MIN <= words <= WORDS_MAX:
        verdict.length_issue = f"The letter has {words} words; it must be 280-380."
    return verdict, record


def save_letter(save_dir, job, letter):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    company = re.sub(r"[^a-zA-Z0-9_-]+", "-", job["company"])[:40]
    title = re.sub(r"[^a-zA-Z0-9_-]+", "-", job["title"])[:60]
    path = save_dir / f"{company}-{title}-{datetime.now():%Y%m%d}.txt"
    path.write_text(f"{job['title']} @ {job['company']}\n{job['url']}\n\n{letter}\n", encoding="utf-8")
    return str(path)


def build_graph(llm, profile, contact, search_fn, save_dir="applications"):
    def search(state):
        jobs, _mode = search_fn(state["search_term"], state["location"])
        candidates = jobs[:MAX_SCORED]
        log = event("search", term=state["search_term"], location=state["location"], found=len(jobs), scored=len(candidates))
        return {"candidates": candidates, "transcript": [log]}

    def score(state):
        fits, record = llm.complete("score", score_prompt(profile, state["candidates"], state["search_term"]), FitList)
        picked = []
        for fit in fits.fits:
            if fit.recommend and 0 <= fit.id < len(state["candidates"]):
                picked.append({**state["candidates"][fit.id], "score": fit.score, "reason": fit.reason})
        picked.sort(key=lambda j: j["score"], reverse=True)
        items = [{k: j[k] for k in ("title", "company", "score", "reason")} for j in picked]
        log = event("shortlist", scored=len(state["candidates"]), items=items)
        return {"shortlist": picked, "calls": [record], "transcript": [log]}

    def pick(state):
        # interrupt()/Command(resume=...): the human-in-the-loop mechanism.
        # First call: interrupt(...) raises internally. LangGraph catches it, checkpoints
        # state, and hands the payload back to the caller as a paused run -- nothing below
        # this line executes yet.
        # Resume call (graph.invoke(Command(resume=answer), config)): LangGraph does NOT
        # continue mid-function. It calls pick(state) again, FROM THE TOP. This time
        # interrupt() doesn't raise -- it returns `answer` -- and execution falls through
        # to the rest of the function for the first time.
        # Earlier nodes (search, score) are not re-run: their results are already
        # checkpointed. It's specifically the interrupted node that repeats its own top.
        # Consequence: anything with a side effect placed BEFORE interrupt() in this
        # function would fire again on every resume. That's why save_letter() lives in its
        # own `save` node, reached only after review()'s interrupt has already resolved --
        # never before an interrupt in the same function body.
        chosen = interrupt({"type": "pick", "shortlist": state["shortlist"]})
        shortlist = state["shortlist"]
        queue = [shortlist[i] for i in chosen if 0 <= i < len(shortlist)]
        return {"queue": queue, "transcript": [event("user_pick", chosen=[f"{j['title']} @ {j['company']}" for j in queue])]}

    def next_job(state):
        # queue[0] is always safe here even though it looks like it should be guarded:
        # next_job is only ever REACHED via a conditional edge (next_or_end, from pick or
        # from save) that already checked `queue` is non-empty before routing here. The
        # branch lives on the edge going INTO next_job, not inside it -- so by the time this
        # function runs, there is nothing left to check.
        queue = state["queue"]
        job = queue[0]
        return {
            "job": job, "queue": queue[1:], "letter": "", "attempts": 0, "feedback": "", "guidance": "",
            "transcript": [event("job_start", job=f"{job['title']} @ {job['company']}")],
        }

    def draft(state):
        prompt = draft_prompt(profile, contact, state["job"], state["letter"], state["feedback"], state.get("guidance", ""))
        letter, record = llm.complete("draft", prompt)
        log = event("draft", model=record["model"], words=len(letter.split()), letter=letter,
                    revision_notes=state["feedback"], guidance=state.get("guidance", ""))
        return {"letter": letter, "calls": [record], "transcript": [log]}

    def verify(state):
        verdict, record = check_letter(llm, profile, state["job"], state["letter"])
        return {
            "verdict": verdict.model_dump(),
            "feedback": verdict.feedback(),
            "attempts": state["attempts"] + 1,
            "calls": [record],
            "transcript": [event("verdict", **verdict.model_dump())],
        }

    def review(state):
        # Same interrupt/resume mechanics as pick() above: this function also re-runs from
        # its own top on every resume, with interrupt() returning the human's decision
        # instead of raising the second (and any later) time through.
        decision = interrupt(
            {"type": "review", "job": state["job"], "letter": state["letter"], "verdict": state["verdict"], "attempts": state["attempts"]}
        )
        log = event("user_review", action=decision["action"], feedback=decision.get("feedback", ""))
        if decision["action"] == "reject":
            guidance = "\n".join(filter(None, [state.get("guidance", ""), decision.get("feedback", "")]))
            return {"action": "reject", "guidance": guidance, "feedback": "", "attempts": 0, "transcript": [log]}
        return {"action": decision["action"], "transcript": [log]}

    def save(state):
        path = save_letter(save_dir, state["job"], state["letter"])
        return {"saved": [path], "transcript": [event("saved", path=path)]}

    def after_search(state):
        return "score" if state["candidates"] else END

    def after_score(state):
        return "pick" if state["shortlist"] else END

    def next_or_end(state):
        return "next_job" if state["queue"] else END

    def after_verify(state):
        if Verdict(**state["verdict"]).passed or state["attempts"] > MAX_REVISIONS:
            return "review"
        return "draft"

    def after_review(state):
        if state["action"] == "approve":
            return "save"
        return "draft" if state["action"] == "reject" else next_or_end(state)

    # Edge map, fixed vs. conditional:
    #   START->search                fixed
    #   search->score or END         conditional: after_search  (empty candidates -> stop)
    #   score->pick or END           conditional: after_score   (empty shortlist -> stop)
    #   pick->next_job or END        conditional: next_or_end   (nothing picked -> stop)
    #   next_job->draft              fixed (see comment in next_job)
    #   draft->verify                fixed
    #   verify->review or draft      conditional: after_verify  (passed/maxed-out -> review, else loop)
    #   review->save/draft/next_job/END  conditional: after_review (approve/reject/skip/nothing-left)
    #   save->next_job or END        conditional: next_or_end   (more jobs queued -> loop, else stop)
    # Only 3 LLM calls exist anywhere in this graph: score(), draft(), and check_letter()
    # (called from verify()). Everything else here is plain Python over already-computed state.
    g = StateGraph(State)
    for name, fn in [("search", search), ("score", score), ("pick", pick), ("next_job", next_job),
                     ("draft", draft), ("verify", verify), ("review", review), ("save", save)]:
        g.add_node(name, fn)
    g.add_edge(START, "search")
    g.add_conditional_edges("search", after_search, ["score", END])
    g.add_conditional_edges("score", after_score, ["pick", END])
    g.add_conditional_edges("pick", next_or_end, ["next_job", END])
    g.add_edge("next_job", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", after_verify, ["review", "draft"])
    g.add_conditional_edges("review", after_review, ["save", "draft", "next_job", END])
    g.add_conditional_edges("save", next_or_end, ["next_job", END])
    return g.compile(checkpointer=InMemorySaver())
