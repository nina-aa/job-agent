"""LLM layer: provider-agnostic fallback chain with per-call token/latency records."""
import os
import re
import time

from dotenv import load_dotenv

# Free-tier quotas are per model, so a long chain keeps the agent running when one is exhausted.
# Override with GEMINI_MODELS="a,b,c". Unknown names are skipped on 404, so stale entries are harmless.
DEFAULT_GEMINI_MODELS = [  # text models from ai.google.dev/gemini-api/docs/models, checked 2026-09-21
    # Cheapest first: older and lite models, then newer flash, with the expensive pro model last.
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-pro",
]
COOLDOWN_SECONDS = 60


class LLMError(RuntimeError):
    pass


def parse_json(text, schema):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return schema.model_validate_json(text[start : end + 1])


class LLM:
    """Subclasses set `models` and implement `_call`. `complete` walks the chain."""

    models: list[str] = []

    def __init__(self):
        self._cooldown = {}
        self._dead = set()

    def _call(self, model, prompt, node):
        """Return (text, input_tokens, output_tokens)."""
        raise NotImplementedError

    def _classify(self, exc):
        if isinstance(exc, ValueError):
            return "bad_output"
        return f"error({type(exc).__name__})"

    def complete(self, node, prompt, schema=None):
        """Return (result, record). `result` is a `schema` instance, or the stripped text."""
        failures = []
        for model in self.models:
            if model in self._dead:
                continue
            if self._cooldown.get(model, 0) > time.monotonic():
                failures.append(f"{model}: cooling down")
                continue
            start = time.perf_counter()
            try:
                text, tokens_in, tokens_out = self._call(model, prompt, node)
                if not text or not text.strip():
                    raise ValueError("empty reply")
                result = parse_json(text, schema) if schema else text.strip()
            except Exception as exc:
                kind = self._classify(exc)
                if kind == "rate_limited":
                    self._cooldown[model] = time.monotonic() + COOLDOWN_SECONDS
                elif kind == "unavailable":
                    self._dead.add(model)
                failures.append(f"{model}: {kind}")
                continue
            record = {
                "node": node,
                "model": model,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "seconds": round(time.perf_counter() - start, 3),
                "failed_attempts": failures,
            }
            return result, record
        raise LLMError("all models failed: " + "; ".join(failures))


class GeminiLLM(LLM):
    def __init__(self, api_key=None, models=None):
        super().__init__()
        from google import genai

        load_dotenv()
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY in .env (or use --fake)")
        self.client = genai.Client(api_key=api_key)
        env_models = os.getenv("GEMINI_MODELS")
        self.models = models or ([m.strip() for m in env_models.split(",") if m.strip()] if env_models else DEFAULT_GEMINI_MODELS)

    def _call(self, model, prompt, node):
        # This is the one line in the whole repo that makes an HTTP request to Gemini.
        # LLM.complete() (below) is what wraps it with the fallback chain: on failure it
        # tries the next model in self.models rather than calling this twice for one model.
        resp = self.client.models.generate_content(model=model, contents=prompt)
        usage = resp.usage_metadata
        return resp.text, (usage.prompt_token_count or 0), (usage.candidates_token_count or 0)

    def _classify(self, exc):
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            if exc.code == 429:
                return "rate_limited"
            if exc.code == 404:
                return "unavailable"
            return f"error({exc.code})"
        return super()._classify(exc)

    def list_models(self):
        return sorted(m.name.removeprefix("models/") for m in self.client.models.list())


class FakeLLM(LLM):
    """Deterministic offline provider. The first draft contains a fabricated claim so the
    verify -> revise loop is exercised; the fake verifier flags a fixed set of fabrications."""

    models = ["fake"]
    FABRICATIONS = ("PhD", "MSc", "Spotify", "40%")

    def __init__(self, flawed_first_draft=True):
        super().__init__()
        self.flawed_first_draft = flawed_first_draft

    def _call(self, model, prompt, node):
        text = getattr(self, f"_{node}")(prompt)
        return text, len(prompt) // 4, len(text) // 4

    def _score(self, prompt):
        fits = []
        for m in re.finditer(r"^\[(\d+)\] (.*)$", prompt, re.M):
            match = "python" in m.group(2).lower() and "senior" not in m.group(2).lower()
            fits.append(
                f'{{"id": {m.group(1)}, "score": {8 if match else 3}, '
                f'"reason": "fake scorer: {"Python role" if match else "no Python or too senior"}", '
                f'"recommend": {"true" if match else "false"}}}'
            )
        return '{"fits": [' + ", ".join(fits) + "]}"

    def _draft(self, prompt):
        company = re.search(r"^Company: (.*)$", prompt, re.M).group(1)
        title = re.search(r"^Title: (.*)$", prompt, re.M).group(1)
        sentences = [
            f"I am writing to apply for the {title} role at {company}.",
            "I am a final-year BSc student in Information Systems with a practical, data-focused background.",
            "I built a courier ETA predictor that improved MAE by 14% against the baseline.",
            "I also automated weekly operations KPIs, saving about two hours a week.",
            "Separately, I classified support tickets with a precision of 0.86, which sharpened my approach to data cleaning.",
            "My core toolkit is Python, SQL and Excel, and I am comfortable with basic statistics and A/B testing.",
            "I have also built dashboards in Looker and Tableau, and I work daily with APIs and JSON.",
            "Across these projects I have valued concise writing, structured thinking and a bias to action.",
            "I enjoy working with stakeholders to turn a loosely defined question into a clear, well-scoped analysis.",
            f"What draws me to {company} specifically is the chance to apply these skills to a real product rather than a classroom exercise.",
            "I learn new tools quickly when a project calls for them, and I would rather ask a clarifying question early than guess and redo work later.",
            "I hold myself to a high bar on correctness, particularly around anything that reaches a dashboard or a stakeholder-facing report.",
            "Outside coursework I keep a small habit of rebuilding public datasets into clean, documented tables, mostly to keep my SQL and data-cleaning instincts sharp between projects.",
            "I would welcome the chance to bring that same habit of checking my own work to a team that relies on its data being right.",
        ]
        if self.flawed_first_draft and "REVISION NOTES" not in prompt:
            sentences.append("I also hold a PhD in machine learning.")
        if "REVIEWER GUIDANCE" in prompt:
            sentences.append("I would also add that I am keen to build on this experience and grow into the role.")
        sentences.append("I am available part-time now and full-time from June 2026, and would welcome a conversation.")
        return "Dear Hiring Team,\n\n" + " ".join(sentences)

    def _verify(self, prompt):
        letter = prompt.split("LETTER:", 1)[1]
        items = ", ".join(
            f'{{"claim": "{f} appears in the letter", "profile_evidence": "none"}}' for f in self.FABRICATIONS if f in letter
        )
        return '{"checks": [' + items + "]}"


def get_llm(fake=False):
    load_dotenv()
    if fake or os.getenv("LLM_PROVIDER") == "fake":
        return FakeLLM()
    return GeminiLLM()
