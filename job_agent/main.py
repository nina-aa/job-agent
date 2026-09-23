"""CLI: python -m job_agent.main [--fake] [--search-term python] [--location Europe]"""
import argparse
import pathlib
import time
import uuid
from datetime import datetime

from langgraph.types import Command

from .graph import build_graph
from .llm import GeminiLLM, get_llm
from .metrics import summarize
from .search import search_jobs
from .transcript import render_markdown


def ask_pick(payload):
    print("\nThe agent recommends writing letters for:")
    for i, j in enumerate(payload["shortlist"]):
        print(f"  [{i}] {j['score']}/10  {j['title']} @ {j['company']} ({j['location']})")
        print(f"        {j['reason']}\n        {j['url']}")
    raw = input("Write letters for which? (e.g. 0,2 / all / none): ").strip().lower()
    if raw == "all":
        return list(range(len(payload["shortlist"])))
    return [int(x) for x in raw.replace(" ", "").split(",") if x.isdigit()]


def ask_review(payload):
    job, verdict = payload["job"], payload["verdict"]
    print(f"\n=== {job['title']} @ {job['company']} (draft round {payload['attempts']}) ===\n{payload['letter']}\n")
    problems = verdict["unsupported_claims"] + ([verdict["length_issue"]] if verdict["length_issue"] else [])
    if problems:
        print("Verifier still flags: " + "; ".join(problems))
    choice = input("[a]pprove / [r]eject with feedback / [s]kip: ").strip().lower()
    if choice.startswith("a"):
        return {"action": "approve"}
    if choice.startswith("r"):
        return {"action": "reject", "feedback": input("Feedback for the redraft: ").strip()}
    return {"action": "skip"}


def run(graph, initial):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 100}
    graph_seconds, payload = 0.0, initial
    while True:
        start = time.perf_counter()
        result = graph.invoke(payload, config)
        graph_seconds += time.perf_counter() - start
        if not result.get("__interrupt__"):
            return graph.get_state(config).values, graph_seconds
        request = result["__interrupt__"][0].value
        payload = Command(resume=ask_pick(request) if request["type"] == "pick" else ask_review(request))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-term", default="python")
    parser.add_argument("--location", default="Europe")
    parser.add_argument("--pages", type=int, default=5, help="Arbeitnow pages to scan")
    parser.add_argument("--strict", action="store_true", help="exact location matches only")
    parser.add_argument("--fake", action="store_true", help="offline fake LLM instead of Gemini")
    parser.add_argument("--profile", default="profile.txt")
    parser.add_argument("--contact", default="contact.txt")
    parser.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid and exit")
    parser.add_argument("--list-models", action="store_true", help="list Gemini models your key can use and exit")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(GeminiLLM().list_models()))
        return

    read = lambda p: pathlib.Path(p).read_text(encoding="utf-8")
    search_fn = lambda term, loc: search_jobs(term, loc, strict=args.strict, pages=args.pages)
    graph = build_graph(get_llm(fake=args.fake or args.mermaid), read(args.profile), read(args.contact), search_fn)

    if args.mermaid:
        print(graph.get_graph().draw_mermaid())
        return

    print(f"Searching for '{args.search_term}' in '{args.location}'...")
    state, graph_seconds = run(graph, {"search_term": args.search_term, "location": args.location})

    if not state.get("candidates"):
        print("No matching postings found. Try another location or search term.")
    elif not state.get("shortlist"):
        print("The agent found no postings worth a cover letter.")
    for path in state.get("saved", []):
        print(f"Saved: {path}")
    if state.get("transcript"):
        out = pathlib.Path("applications")
        out.mkdir(exist_ok=True)
        path = out / f"transcript-{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(render_markdown(state["transcript"]), encoding="utf-8")
        print(f"Transcript: {path}")
    print("\n" + summarize(state.get("calls", []), graph_seconds))


if __name__ == "__main__":
    main()
