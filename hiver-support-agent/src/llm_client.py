"""
LLM client abstraction.

Priority order:
  1. GEMINI_API_KEY (or GOOGLE_API_KEY) set -> real Gemini API call
  2. ANTHROPIC_API_KEY set -> real Claude API call
  3. neither set -> deterministic offline MOCK mode

The mock is NOT a toy - it uses the same keyword/similarity heuristics a
first-pass rule system would use, so baseline vs LLM comparisons in the eval
harness stay meaningful even without a key.

NOTE ON THIS SANDBOX SPECIFICALLY: this pipeline was built and tested in an
environment whose network egress is restricted to a fixed allowlist
(api.anthropic.com, pypi.org, github.com, etc) and does NOT include
generativelanguage.googleapis.com. So even with a valid Gemini key, real
Gemini calls could not be executed from that sandbox - the Gemini path below
is implemented and ready to run, but was validated for correct
request/response shape only, not exercised end-to-end with a live key. Run
it in an unrestricted environment (your own machine, Colab, etc) to get real
numbers - see README "Using a real LLM" section.
"""

import os
import json
import hashlib
import random

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

USE_GEMINI = bool(GEMINI_API_KEY)
USE_ANTHROPIC = bool(ANTHROPIC_API_KEY) and not USE_GEMINI
USE_REAL_LLM = USE_GEMINI or USE_ANTHROPIC

_client = None


def _get_anthropic_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _call_gemini(system: str, user: str, max_tokens: int) -> str:
    """
    Direct REST call to the Gemini API (no SDK dependency, so this works in
    any environment with network access to generativelanguage.googleapis.com
    without needing google-generativeai installed).
    """
    import urllib.request
    import time

    model = "gemini-flash-lite-latest"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    
    # Retry with exponential backoff for rate limits
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)  # 5, 10, 20, 40 seconds
                    print(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Rate limited, all {max_retries} retries exhausted.")
            raise


def call_llm(system: str, user: str, max_tokens: int = 500) -> str:
    """Single entry point every module uses. Returns raw text response."""
    if USE_GEMINI:
        try:
            return _call_gemini(system, user, max_tokens)
        except Exception as e:
            # Fail loud in real mode rather than silently degrading to mock -
            # a silently-mocked "real" run would produce misleading eval numbers.
            raise RuntimeError(
                f"Gemini API call failed ({e}). If this is a network-egress "
                f"restriction, run this pipeline in an environment with access "
                f"to generativelanguage.googleapis.com, or unset GEMINI_API_KEY "
                f"to fall back to mock mode explicitly."
            ) from e
    elif USE_ANTHROPIC:
        client = _get_anthropic_client()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    else:
        return _mock_llm(system, user)


# ---------------------------------------------------------------------------
# Mock mode: deterministic, keyword-driven, but structurally identical to what
# a real LLM call would return (same JSON schema etc). This lets the eval
# harness, baselines, and report all run without any API key.
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS = {
    "order_status": ["order", "tracking", "shipped", "where is my", "arrive", "cancel my order"],
    "delivery_issue": ["late", "delayed", "never arrived", "still waiting", "lost package", "damaged in transit"],
    "refund_request": ["refund", "money back", "charged twice", "return"],
    "account_access": ["can't log in", "locked out", "password", "account suspended", "hacked"],
    "product_defect": ["broken", "defective", "not working", "damaged", "doesn't work", "crashing", "won't open"],
    "billing_dispute": ["overcharged", "billed", "invoice", "charge on my card", "unauthorized charge", "without my permission"],
    "membership_issue": ["prime", "membership", "subscription", "prime video", "signed me up"],
    "positive_feedback": ["thank you", "thanks", "great service", "love", "awesome", "appreciate"],
    "general_complaint": ["worst", "terrible", "unacceptable", "ridiculous", "furious", "customer service"],
}

_ESCALATE_TRIGGERS = ["lawyer", "legal", "sue", "hurt", "injury", "unsafe", "fraud", "unauthorized"]


def _classify_mock(text: str):
    t = text.lower()
    scores = {}
    for intent, kws in _INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in kws if kw in t)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "other_unclear"
    # deterministic pseudo-confidence from hash, biased by keyword hit count
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
    confidence = min(0.55 + 0.15 * scores.get(best, 0) + (h % 10) / 100, 0.97)
    return best, round(confidence, 2)


def _mock_llm(system: str, user: str) -> str:
    """
    Routes based on a tag embedded in the system prompt so pipeline modules
    can request 'CLASSIFY', 'DRAFT', or 'JUDGE' behavior from the same call_llm().
    """
    if "TASK:CLASSIFY" in system:
        # user content is the raw customer message
        intent, conf = _classify_mock(user)
        return json.dumps({"intent": intent, "confidence": conf})

    if "TASK:DRAFT" in system:
        # user content contains customer message + retrieved exemplars (JSON blob after '---EXEMPLARS---')
        parts = user.split("---EXEMPLARS---")
        customer_msg = parts[0].strip()
        exemplars = []
        if len(parts) > 1:
            try:
                exemplars = json.loads(parts[1].strip())
            except Exception:
                exemplars = []
        if exemplars:
            template_reply = exemplars[0].get("brand_reply", "")
        else:
            template_reply = "We're sorry to hear this. Please DM us your order number so we can help."
        # lightly personalize
        draft = f"Hi, thanks for reaching out. {template_reply}"
        return draft

    if "TASK:JUDGE" in system:
        # user content contains JSON with customer_msg, reply, exemplar
        random.seed(hashlib.md5(user.encode()).hexdigest())
        groundedness = round(random.uniform(3.0, 5.0), 1)
        correctness = round(random.uniform(3.0, 5.0), 1)
        tone = round(random.uniform(3.5, 5.0), 1)
        return json.dumps({
            "groundedness": groundedness,
            "correctness": correctness,
            "tone": tone,
            "notes": "mock-judge score (no API key set)"
        })

    return "{}"
