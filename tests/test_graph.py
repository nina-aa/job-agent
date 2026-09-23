import pytest
from langgraph.types import Command

from job_agent.graph import build_graph
from job_agent.llm import LLM, FakeLLM, LLMError
from job_agent.models import Job
from job_agent.search import matches_broad_fallback, matches_location, matches_search_term

PROFILE, CONTACT = "profile", "contact"

JOBS = [
    Job(source="T", title="Python Developer", company="Acme", location="Europe", url="u1", tags=["python"]).model_dump(),
    Job(source="T", title="Senior Java Architect", company="Big", location="Europe", url="u2", tags=["java"]).model_dump(),
    Job(source="T", title="Python Data Engineer", company="Data Co", location="Sweden", url="u3", tags=["python"]).model_dump(),
]


def make_graph(tmp_path, jobs=JOBS, llm=None):
    return build_graph(llm or FakeLLM(), PROFILE, CONTACT, lambda term, loc: (jobs, "exact"), save_dir=tmp_path)


def start(graph, thread="t"):
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 100}
    return config, graph.invoke({"search_term": "python", "location": "Europe"}, config)


def test_happy_path_with_revision_loop_and_two_interrupts(tmp_path):
    graph = make_graph(tmp_path)
    config, result = start(graph)

    pick = result["__interrupt__"][0].value
    assert pick["type"] == "pick"
    assert [j["company"] for j in pick["shortlist"]] == ["Acme", "Data Co"]  # Java role screened out

    result = graph.invoke(Command(resume=[0]), config)
    review = result["__interrupt__"][0].value
    assert review["type"] == "review"
    assert review["attempts"] == 2  # first draft had a fabricated claim, second passed
    assert review["verdict"] == {"unsupported_claims": [], "length_issue": ""}
    assert "PhD" not in review["letter"]

    graph.invoke(Command(resume={"action": "approve"}), config)
    state = graph.get_state(config).values
    assert len(state["saved"]) == 1 and "Acme" in state["saved"][0]
    assert [c["node"] for c in state["calls"]] == ["score", "draft", "verify", "draft", "verify"]


def test_reject_with_feedback_redrafts_and_skip_moves_on(tmp_path):
    graph = make_graph(tmp_path)
    config, _ = start(graph)
    graph.invoke(Command(resume=[0, 1]), config)
    result = graph.invoke(Command(resume={"action": "reject", "feedback": "shorter"}), config)
    assert result["__interrupt__"][0].value["type"] == "review"
    result = graph.invoke(Command(resume={"action": "skip"}), config)
    assert result["__interrupt__"][0].value["job"]["company"] == "Data Co"
    graph.invoke(Command(resume={"action": "skip"}), config)
    assert not graph.get_state(config).values.get("saved")


def test_stops_when_nothing_worth_a_letter(tmp_path):
    graph = make_graph(tmp_path, jobs=[JOBS[1]])
    _, result = start(graph)
    assert "__interrupt__" not in result and result["shortlist"] == []


def test_stops_when_search_is_empty(tmp_path):
    _, result = start(make_graph(tmp_path, jobs=[]))
    assert result["candidates"] == []


def test_human_can_pick_nothing(tmp_path):
    graph = make_graph(tmp_path)
    config, _ = start(graph)
    graph.invoke(Command(resume=[]), config)
    assert not graph.get_state(config).next


class Scripted(LLM):
    """Fails per model according to `plan`, otherwise returns 'ok'."""

    models = ["a", "b", "c"]

    def __init__(self, plan):
        super().__init__()
        self.plan, self.calls = plan, []

    def _call(self, model, prompt, node):
        self.calls.append(model)
        if model in self.plan:
            raise self.plan[model]
        return "ok", 10, 2

    def _classify(self, exc):
        return str(exc) if str(exc) in ("rate_limited", "unavailable") else super()._classify(exc)


def test_fallback_chain_cooldown_and_dead_models():
    llm = Scripted({"a": RuntimeError("rate_limited"), "b": RuntimeError("unavailable")})
    text, record = llm.complete("draft", "p")
    assert text == "ok" and record["model"] == "c" and len(record["failed_attempts"]) == 2
    llm.calls.clear()
    llm.complete("draft", "p")
    assert llm.calls == ["c"]  # a is cooling down, b is dead: neither is retried


def test_all_models_failing_raises():
    llm = Scripted({m: RuntimeError("boom") for m in "abc"})
    with pytest.raises(LLMError):
        llm.complete("draft", "p")


def test_search_filters():
    job = JOBS[0]
    assert matches_search_term(job, "PYTHON") and not matches_search_term(job, "rust")
    assert matches_location(job, "europe") and matches_broad_fallback(job)


class Recorder(FakeLLM):
    """Fabricates on the first redraft after the human's reject, forcing a verifier-driven round."""

    def __init__(self):
        super().__init__(flawed_first_draft=False)
        self.prompts = []

    def _draft(self, prompt):
        self.prompts.append(prompt)
        text = super()._draft(prompt)
        return text.replace("Dear Hiring Team,", "Dear Hiring Team, I hold a PhD.") if len(self.prompts) == 2 else text


def test_human_guidance_survives_verifier_driven_redrafts(tmp_path):
    llm = Recorder()
    graph = make_graph(tmp_path, llm=llm)
    config, _ = start(graph)
    graph.invoke(Command(resume=[0]), config)
    graph.invoke(Command(resume={"action": "reject", "feedback": "mention kubernetes as a basic skill"}), config)
    assert len(llm.prompts) == 3
    assert "mention kubernetes as a basic skill" in llm.prompts[1]
    assert "mention kubernetes as a basic skill" in llm.prompts[2]  # still there after the verifier's redraft
    assert "PhD" in llm.prompts[2]  # ...alongside the verifier's own note


def test_transcript_records_the_agent_human_exchange(tmp_path):
    from job_agent.transcript import render_markdown

    graph = make_graph(tmp_path)
    config, _ = start(graph)
    graph.invoke(Command(resume=[0]), config)
    graph.invoke(Command(resume={"action": "reject", "feedback": "shorter please"}), config)
    graph.invoke(Command(resume={"action": "approve"}), config)
    events = graph.get_state(config).values["transcript"]
    assert [e["type"] for e in events] == [
        "search", "shortlist", "user_pick", "job_start",
        "draft", "verdict", "draft", "verdict",  # flawed first draft, verifier-driven redraft
        "user_review", "draft", "verdict", "user_review", "saved",  # reject -> redraft -> approve
    ]
    assert events[8]["feedback"] == "shorter please"
    text = render_markdown(events)
    assert "shorter please" in text and "You** approve" in text
