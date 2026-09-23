"""Job sources: Remotive, Arbeitnow, RemoteOK and Jobicy public JSON APIs (no scraping).

Plain deterministic code on purpose -- the agent's judgement starts after this step.
"""
import html
import re
import sys

import requests

from .models import Job

USER_AGENT = "remote-job-agent/1.0 (github portfolio project)"
BROAD_FALLBACK_TERMS = ["europe", "emea", "nordic", "worldwide", "anywhere", "global"]
MAX_TAGS_MATCHED = 8


def _clean(text, limit=500):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def normalize(source, title, company, location, url, tags, description=""):
    return Job(
        source=source,
        title=title or "(untitled)",
        company=company or "Unknown company",
        location=location or "Not specified",
        url=url or "",
        tags=[str(t) for t in (tags or [])],
        description=_clean(description),
    ).model_dump()


def _get(url, **kwargs):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15, **kwargs)
    resp.raise_for_status()
    return resp.json()


def fetch_remotive(search_term):
    """Remotive: single endpoint, supports a free-text `search` param."""
    try:
        jobs = _get("https://remotive.com/api/remote-jobs", params={"search": search_term} if search_term else {}).get("jobs", [])
    except requests.RequestException as e:
        print(f"  Remotive fetch failed: {e}", file=sys.stderr)
        return []
    return [
        normalize("Remotive", j.get("title"), j.get("company_name"), j.get("candidate_required_location"), j.get("url"), j.get("tags"), j.get("description"))
        for j in jobs
    ]


def fetch_arbeitnow(max_pages):
    """Arbeitnow: paginated, no search param -- pages are fetched and filtered client-side."""
    all_jobs = []
    for page in range(1, max_pages + 1):
        try:
            payload = _get("https://www.arbeitnow.com/api/job-board-api", params={"page": page})
        except requests.RequestException as e:
            print(f"  Arbeitnow fetch failed on page {page}: {e}", file=sys.stderr)
            break
        data = payload.get("data", [])
        if not data:
            break
        all_jobs.extend(
            normalize("Arbeitnow", j.get("title"), j.get("company_name"), j.get("location"), j.get("url"), j.get("tags"), j.get("description"))
            for j in data
        )
        if not payload.get("links", {}).get("next"):
            break
    return all_jobs


def fetch_remoteok():
    """RemoteOK: single JSON array; the first element is a legal notice, not a job."""
    try:
        jobs = _get("https://remoteok.com/api")[1:]
    except requests.RequestException as e:
        print(f"  RemoteOK fetch failed: {e}", file=sys.stderr)
        return []
    return [
        normalize("RemoteOK", j.get("position"), j.get("company"), j.get("location"), j.get("url"), j.get("tags"), j.get("description"))
        for j in jobs
    ]


def fetch_jobicy(search_term):
    """Jobicy: tag filter; results carry a seniority level and geo like 'EMEA' or 'Anywhere'."""
    params = {"count": 50, **({"tag": search_term} if search_term else {})}
    try:
        jobs = _get("https://jobicy.com/api/v2/remote-jobs", params=params).get("jobs", [])
    except requests.RequestException as e:
        print(f"  Jobicy fetch failed: {e}", file=sys.stderr)
        return []
    level = lambda j: j.get("jobLevel") or j.get("seniority") or ""
    return [
        normalize("Jobicy", j.get("jobTitle"), j.get("companyName"), j.get("jobGeo"), j.get("url"),
                  [j.get("jobIndustry")] if isinstance(j.get("jobIndustry"), str) else j.get("jobIndustry"),
                  f"Level: {level(j)}. {j.get('jobExcerpt') or ''}")
        for j in jobs
    ]


def matches_search_term(job, search_term):
    """Title, description snippet and the first few tags. Some boards stuff 40+ tags into every
    posting, so scanning all of them would match any term."""
    if not search_term:
        return True
    haystack = " ".join([job["title"], job["description"], *job["tags"][:MAX_TAGS_MATCHED]]).lower()
    return search_term.lower() in haystack


def matches_location(job, location):
    return location.lower() in job["location"].lower()


def matches_broad_fallback(job):
    loc = job["location"].lower()
    return any(term in loc for term in BROAD_FALLBACK_TERMS)


def search_jobs(search_term, location, strict=False, pages=5):
    """Return (jobs, mode). Exact location matches come first; unless `strict`, broader
    remote-eligible ones (Europe/Worldwide/...) follow, so a few exact hits don't hide the rest.
    mode is 'exact', 'broad' (nothing exact) or 'none'."""
    all_jobs = fetch_remotive(search_term) + fetch_jobicy(search_term) + fetch_arbeitnow(pages) + fetch_remoteok()
    filtered = [j for j in all_jobs if matches_search_term(j, search_term)]
    exact = [j for j in filtered if matches_location(j, location)]
    if strict:
        return exact, ("exact" if exact else "none")
    broad = [j for j in filtered if matches_broad_fallback(j) and j not in exact]
    return exact + broad, ("exact" if exact else "broad" if broad else "none")
