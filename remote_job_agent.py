"""
Searches Remotive, Arbeitnow, and RemoteOK's public JSON APIs for remote jobs,
filtered by a location and a search term you provide, then drafts a tailored
cover letter for the top matches with Gemini. All three job sources are public,
unauthenticated, ToS-friendly aggregator APIs -- no scraping involved.

Usage:
    python remote_job_agent.py                 # prompts for location + search term
    python remote_job_agent.py --location Sweden --search-term developer
    python remote_job_agent.py --location Germany --strict
    python remote_job_agent.py --no-letters     # search only, skip Gemini
"""
import argparse
import os
import pathlib
import re
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv
import google.generativeai as genai

USER_AGENT = "remote-job-agent/1.0 (github portfolio project)"

BROAD_FALLBACK_TERMS = ["europe", "emea", "nordic", "worldwide", "anywhere", "global"]

MAX_LETTERS = 2
SAVE_DIR = pathlib.Path("applications")

# Edit these to match the real candidate before generating letters.
PROFILE_TEXT = (
    "Student data/ops analyst profile: final-year BSc (Information Systems).\n"
    "Core skills: Python, SQL, Excel, basic statistics, A/B testing, dashboards (Looker/Tableau),\n"
    "APIs & JSON, data cleaning, stakeholder comms.\n"
    "Projects: (1) Built a courier ETA predictor (MAE -14% vs baseline),\n"
    "(2) Automated weekly ops KPIs (2 hrs saved/week), (3) Classified support tickets (precision 0.86).\n"
    "Strengths: concise writing, structured thinking, bias-to-action.\n"
    "Availability: part-time now, full-time from June 2026. Location: Remote.\n"
)

CONTACT_BLOCK = (
    "Your Name | Your City, Country | your.email@example.com | +00 000 0000000 | github.com/you\n"
)


def normalize(source, title, company, location, url, tags):
    return {
        "source": source,
        "title": title or "(untitled)",
        "company": company or "Unknown company",
        "location": location or "Not specified",
        "url": url or "",
        "tags": tags or [],
    }


def fetch_remotive(search_term):
    """Remotive: single JSON endpoint, supports a free-text `search` param."""
    url = "https://remotive.com/api/remote-jobs"
    params = {"search": search_term} if search_term else {}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])
    except requests.RequestException as e:
        print(f"  Remotive fetch failed: {e}")
        return []

    return [
        normalize(
            "Remotive",
            j.get("title"),
            j.get("company_name"),
            j.get("candidate_required_location"),
            j.get("url"),
            j.get("tags"),
        )
        for j in jobs
    ]


def fetch_arbeitnow(max_pages):
    """Arbeitnow: paginated, no search param -- fetch pages and filter client-side."""
    all_jobs = []
    url = "https://www.arbeitnow.com/api/job-board-api"
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(url, params={"page": page}, headers={"User-Agent": USER_AGENT}, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as e:
            print(f"  Arbeitnow fetch failed on page {page}: {e}")
            break

        data = payload.get("data", [])
        if not data:
            break

        all_jobs.extend(
            normalize(
                "Arbeitnow",
                j.get("title"),
                j.get("company_name"),
                j.get("location"),
                j.get("url"),
                j.get("tags"),
            )
            for j in data
        )

        if not payload.get("links", {}).get("next"):
            break

    return all_jobs


def fetch_remoteok():
    """RemoteOK: single JSON array; first element is a legal notice, not a job."""
    url = "https://remoteok.com/api"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        jobs = resp.json()[1:]
    except requests.RequestException as e:
        print(f"  RemoteOK fetch failed: {e}")
        return []

    return [
        normalize(
            "RemoteOK",
            j.get("position"),
            j.get("company"),
            j.get("location"),
            j.get("url"),
            j.get("tags"),
        )
        for j in jobs
    ]


def matches_search_term(job, search_term):
    if not search_term:
        return True
    term = search_term.lower()
    haystack = job["title"].lower() + " " + " ".join(t.lower() for t in job["tags"])
    return term in haystack


def matches_location(job, location):
    return location.lower() in job["location"].lower()


def matches_broad_fallback(job):
    loc = job["location"].lower()
    return any(term in loc for term in BROAD_FALLBACK_TERMS)


def print_results(jobs, header):
    print(f"\n{header} ({len(jobs)})")
    print("-" * 70)
    for j in jobs:
        print(f"[{j['source']}] {j['title']} - {j['company']}")
        print(f"    Location: {j['location']}")
        print(f"    {j['url']}")
    print()


def init_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-2.5-flash")


def build_prompt(job):
    job_summary = (
        f"Company: {job['company']}\n"
        f"Title: {job['title']}\n"
        f"Location: {job['location']}\n"
        f"Source: {job['source']}\n"
        f"Link: {job['url']}\n"
        f"Tags: {', '.join(job['tags'])}"
    )
    return f"""
You are a careful assistant writing a concise cover letter.
Rules:
- 280-380 words, British English.
- Do not invent facts outside PROFILE.
- Use CONTACT exactly as given.
- Refer to the company and role naturally; avoid generic fluff and exclamation marks.
- End with a short, polite call-to-action and availability.

PROFILE:\n{PROFILE_TEXT}\n
CONTACT:\n{CONTACT_BLOCK}\n
JOB SUMMARY:\n{job_summary}\n
Write a tailored cover letter as plain text. No markdown, no headers.
"""


def draft_letter(model, job):
    try:
        resp = model.generate_content(build_prompt(job))
    except Exception as e:
        raise RuntimeError(f"Gemini generate_content failed: {e}")
    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = "(No text returned)"
    return text.strip()


def save_letter(job, letter_text):
    SAVE_DIR.mkdir(exist_ok=True)
    safe_company = re.sub(r"[^a-zA-Z0-9_-]+", "-", job["company"])[:40]
    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "-", job["title"])[:60]
    date_str = datetime.now().strftime("%Y%m%d")
    path = SAVE_DIR / f"{safe_company}-{safe_title}-{date_str}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{job['title']} @ {job['company']}\n{job['url']}\n\n{letter_text}\n")
    return path


def generate_letters(jobs):
    selected = jobs[:MAX_LETTERS]
    print(f"\nGenerate cover letters for these {len(selected)} job(s)?")
    for j in selected:
        print(f"  - {j['title']} @ {j['company']}")
    response = input("[Y/n]: ").strip().lower()
    if response and response != "y":
        print("Skipped cover letter generation.")
        return

    model = init_gemini()
    for idx, job in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] Drafting letter for {job['title']} @ {job['company']}...")
        letter = draft_letter(model, job)
        path = save_letter(job, letter)
        print(f"    Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default=None, help="Location to filter for, e.g. Sweden")
    parser.add_argument("--search-term", dest="search_term", default=None, help="Job title/keyword, e.g. developer")
    parser.add_argument("--pages", type=int, default=5, help="Arbeitnow pages to scan (100 jobs/page)")
    parser.add_argument("--strict", action="store_true", help="Only show exact location matches, skip broad fallback")
    parser.add_argument("--no-letters", action="store_true", help="Search only, skip Gemini cover letter generation")
    args = parser.parse_args()

    location = args.location
    if location is None:
        location = input("Which location are you searching for? [EU]: ").strip() or "EU"

    search_term = args.search_term
    if search_term is None:
        search_term = input("Search term / job keyword (leave blank for any): ").strip()

    print(f"\nSearching Remotive, Arbeitnow, and RemoteOK for '{search_term or 'any role'}' in '{location}'...")

    print("Fetching Remotive...")
    remotive_jobs = fetch_remotive(search_term)
    print("Fetching Arbeitnow...")
    arbeitnow_jobs = fetch_arbeitnow(args.pages)
    print("Fetching RemoteOK...")
    remoteok_jobs = fetch_remoteok()

    all_jobs = remotive_jobs + arbeitnow_jobs + remoteok_jobs
    print(f"\nFetched {len(all_jobs)} total postings across all three sources.")

    term_filtered = [j for j in all_jobs if matches_search_term(j, search_term)]
    exact = [j for j in term_filtered if matches_location(j, location)]

    if exact:
        print_results(exact, f"Exact '{location}' matches")
        if not args.no_letters:
            generate_letters(exact)
        return

    print(f"\nNo listings explicitly mention '{location}' in their location field right now.")

    if args.strict:
        print("(--strict set, not falling back to broader results.)")
        return

    broad = [j for j in term_filtered if matches_broad_fallback(j)]
    if broad:
        print_results(broad, "No exact matches - showing broader remote-eligible roles instead (Europe/Nordic/Worldwide)")
        if not args.no_letters:
            generate_letters(broad)
    else:
        print("No broader remote-eligible matches found either. Try a different location or search term.")


if __name__ == "__main__":
    sys.exit(main())
