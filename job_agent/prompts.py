def job_summary(job):
    return (
        f"Company: {job['company']}\n"
        f"Title: {job['title']}\n"
        f"Location: {job['location']}\n"
        f"Source: {job['source']}\n"
        f"Link: {job['url']}\n"
        f"Tags: {', '.join(job['tags'])}\n"
        f"Description: {job.get('description', '')}"
    )


def score_prompt(profile, jobs, search_term):
    listing = "\n".join(
        f"[{i}] {j['title']} @ {j['company']} | {j['location']} | tags: {', '.join(j['tags'][:8])} | {j.get('description', '')}"
        for i, j in enumerate(jobs)
    )
    return f"""
You screen job postings for a candidate. The candidate searched for: {search_term}.
For every posting decide whether it is a realistic match for the candidate's PROFILE
and whether writing a cover letter is worth the candidate's time.
Be strict about seniority: recommend only roles the candidate could plausibly get.

PROFILE:
{profile}

POSTINGS:
{listing}

Reply with JSON only, no markdown, in this shape:
{{"fits": [{{"id": <posting number>, "score": <0-10>, "reason": "<one sentence>", "recommend": <true|false>}}]}}
Include every posting exactly once.
"""


def draft_prompt(profile, contact, job, previous_letter="", feedback="", guidance=""):
    revision = ""
    if guidance:
        revision += f"\nREVIEWER GUIDANCE (from the human applicant, always follow, but never invent facts):\n{guidance}\n"
    if previous_letter and (feedback or guidance):
        revision += (
            f"\nREVISION NOTES (fix these in a new version of the previous letter):\n{feedback or 'Apply the reviewer guidance.'}\n"
            f"\nPREVIOUS LETTER:\n{previous_letter}\n"
        )
    return f"""
You are a careful assistant writing a concise cover letter.
Rules:
- 280-380 words, British English.
- Do not invent facts outside PROFILE.
- Use CONTACT exactly as given.
- Refer to the company and role naturally; avoid generic fluff and exclamation marks.
- End with a short, polite call-to-action and availability.

PROFILE:
{profile}

CONTACT:
{contact}

JOB SUMMARY:
{job_summary(job)}
{revision}
Write a tailored cover letter as plain text. No markdown, no headers.
"""


def verify_prompt(profile, job, letter):
    return f"""
You are a strict but fair fact-checker for cover letters.
Go through the LETTER and list each factual claim it makes about the CANDIDATE:
education, employers, projects, numbers, tools and skills, availability.
For every claim, quote the closest supporting text from PROFILE in "profile_evidence",
or write "none" if PROFILE contains nothing that supports it.

Rules:
- A paraphrase of something in PROFILE is supported (quote that PROFILE text).
- The role title, company name and anything about the job come from JOB SUMMARY;
  they are not claims about the candidate. Do not list them.
- Motivation, enthusiasm and willingness to learn are not factual claims. Do not list them.
- Do not comment on style or length.

PROFILE:
{profile}

JOB SUMMARY:
{job_summary(job)}

LETTER:
{letter}

Reply with JSON only, no markdown:
{{"checks": [{{"claim": "<short quote from the letter>", "profile_evidence": "<PROFILE text or none>"}}]}}
"""
