from pydantic import BaseModel, Field


class Job(BaseModel):
    source: str
    title: str = "(untitled)"
    company: str = "Unknown company"
    location: str = "Not specified"
    url: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""


class Fit(BaseModel):
    id: int
    score: int = Field(ge=0, le=10)
    reason: str
    recommend: bool


class FitList(BaseModel):
    fits: list[Fit]


class ClaimCheck(BaseModel):
    claim: str
    profile_evidence: str  # closest supporting PROFILE text, or "none"

    @property
    def unsupported(self) -> bool:
        return self.profile_evidence.strip().lower().startswith(("none", "n/a", "no "))  or not self.profile_evidence.strip()


class VerifierOutput(BaseModel):
    checks: list[ClaimCheck] = Field(default_factory=list)


class Verdict(BaseModel):
    """Verifier output. `unsupported_claims` comes from the LLM, `length_issue` from code."""

    unsupported_claims: list[str] = Field(default_factory=list)
    length_issue: str = ""

    @property
    def passed(self) -> bool:
        return not self.unsupported_claims and not self.length_issue

    def feedback(self) -> str:
        notes = [f"Remove or fix this claim not supported by PROFILE: {c}" for c in self.unsupported_claims]
        if self.length_issue:
            notes.append(self.length_issue)
        return "\n".join(notes)
