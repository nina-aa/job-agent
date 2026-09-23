"""Verifier eval: can the fact-checker catch planted fabrications without crying wolf?

    python -m evals.run_evals          # real provider (Gemini chain)
    python -m evals.run_evals --fake   # harness check only; the fake verifier is a keyword rule
"""
import argparse
import json
import pathlib

from job_agent.graph import check_letter
from job_agent.llm import FakeLLM, get_llm
from job_agent.metrics import summarize

ROOT = pathlib.Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    data = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
    profile = (ROOT / "profile.txt").read_text(encoding="utf-8")
    llm = get_llm(fake=args.fake)
    if isinstance(llm, FakeLLM):
        print("NOTE: fake provider -- this only checks the harness, not verifier quality.\n")

    caught = false_alarms = planted = clean = 0
    calls = []
    for case in data["cases"]:
        verdict, record = check_letter(llm, profile, data["job"], case["letter"])
        calls.append(record)
        flagged = bool(verdict.unsupported_claims)  # length is not what this eval measures
        planted += case["fabricated"]
        clean += not case["fabricated"]
        caught += flagged and case["fabricated"]
        false_alarms += flagged and not case["fabricated"]
        ok = flagged == case["fabricated"]
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<16} flagged={verdict.unsupported_claims}")

    print(f"\ncaught fabrications: {caught}/{planted}   false alarms: {false_alarms}/{clean}\n")
    print(summarize(calls))


if __name__ == "__main__":
    main()
