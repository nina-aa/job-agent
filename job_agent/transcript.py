"""The conversation between the agent and the human, recorded as events in graph state."""
from datetime import datetime


def event(kind, **data):
    return {"type": kind, "at": datetime.now().isoformat(timespec="seconds"), **data}


def _quote(text):
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def render_markdown(events):
    out = ["# Run transcript", ""]
    for e in events:
        kind = e["type"]
        time = e["at"][11:]
        if kind == "search":
            out += [f"**[{time}] Search** `{e['term']}` in `{e['location']}`: {e['found']} postings, {e['scored']} sent to the scorer.", ""]
        elif kind == "shortlist":
            out += [f"**[{time}] Agent** recommends {len(e['items'])} of {e['scored']} postings:"]
            out += [f"- {i['score']}/10 **{i['title']}** @ {i['company']}: {i['reason']}" for i in e["items"]] or ["- (none)"]
            out += [""]
        elif kind == "user_pick":
            out += [f"**[{time}] You** chose: {', '.join(e['chosen']) or 'nothing'}", ""]
        elif kind == "job_start":
            out += [f"## {e['job']}", ""]
        elif kind == "draft":
            notes = "; ".join(filter(None, [e["guidance"] and f"your guidance: {e['guidance']}", e["revision_notes"] and f"verifier: {e['revision_notes']}"]))
            out += [f"**[{time}] Agent** draft ({e['words']} words, {e['model']})" + (f", revising for {notes}" if notes else ""), "", _quote(e["letter"]), ""]
        elif kind == "verdict":
            problems = e["unsupported_claims"] + ([e["length_issue"]] if e["length_issue"] else [])
            out += [f"**[{time}] Verifier** " + ("passed." if not problems else "flagged: " + "; ".join(problems)), ""]
        elif kind == "user_review":
            line = f"**[{time}] You** {e['action']}"
            out += [line + (f": {e['feedback']}" if e.get("feedback") else ""), ""]
        elif kind == "saved":
            out += [f"**[{time}] Saved** `{e['path']}`", ""]
    return "\n".join(out)
