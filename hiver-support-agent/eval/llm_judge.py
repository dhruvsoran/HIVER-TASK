"""
LLM-as-judge for reply quality, scored 1-5 on:
  - groundedness: does the reply stick to what the retrieved exemplars support
    (no invented policy/timelines/amounts)?
  - correctness: does it actually address the customer's stated problem?
  - tone: brand-appropriate, empathetic, concise?

Critically: this file also includes the human-agreement check. A judge you
haven't validated against a human is just a second opinion with false
confidence - see report.md "what is misleading about my headline number".
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_client import call_llm

JUDGE_SYSTEM_PROMPT = """TASK:JUDGE
You are grading a customer support reply drafted by an AI agent.
Score on three axes, 1 (bad) to 5 (excellent):
- groundedness: does it avoid inventing policy, refund amounts, or timelines not in the exemplar?
- correctness: does it address what the customer actually asked?
- tone: empathetic, concise, brand-appropriate?

IMPORTANT: Respond ONLY with valid JSON, no other text. Format: {"groundedness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "notes": "<one sentence>"}
"""


def judge_reply(customer_msg: str, reply: str, exemplar: dict = None) -> dict:
    payload = json.dumps({
        "customer_msg": customer_msg,
        "reply": reply,
        "exemplar_used": exemplar,
    })
    raw = call_llm(JUDGE_SYSTEM_PROMPT, payload)
    try:
        return json.loads(raw)
    except Exception:
        return {"groundedness": None, "correctness": None, "tone": None, "notes": "judge parse failure"}


# ---------------------------------------------------------------------------
# Human-agreement check.
#
# Real submission requirement: hand-score N replies yourself, then compare
# to the judge's scores and report correlation (even if it's mediocre).
# Below is the harness for that comparison - `human_scores` is a placeholder
# array you fill in by actually reading the outputs and scoring them
# yourself (see README "Validating the judge" section for the exact steps).
# ---------------------------------------------------------------------------

def agreement_report(judge_scores: list, human_scores: list, axis: str = "correctness"):
    """
    judge_scores / human_scores: lists of dicts with the same axis key,
    aligned by index to the same set of graded replies.
    Uses simple Pearson correlation (no scipy dependency) plus exact-match
    rate as two different views of agreement - correlation alone can look
    good while exact agreement is poor if scores are just biased high/low.
    """
    import math

    j = [s[axis] for s in judge_scores if s.get(axis) is not None]
    h = [s[axis] for s in human_scores if s.get(axis) is not None]
    n = min(len(j), len(h))
    j, h = j[:n], h[:n]
    if n < 2:
        return {"n": n, "pearson_r": None, "exact_match_rate": None}

    mean_j, mean_h = sum(j) / n, sum(h) / n
    cov = sum((j[i] - mean_j) * (h[i] - mean_h) for i in range(n))
    std_j = math.sqrt(sum((x - mean_j) ** 2 for x in j))
    std_h = math.sqrt(sum((x - mean_h) ** 2 for x in h))
    pearson_r = cov / (std_j * std_h) if std_j > 0 and std_h > 0 else None

    exact = sum(1 for i in range(n) if abs(j[i] - h[i]) <= 0.5) / n

    return {"n": n, "pearson_r": round(pearson_r, 3) if pearson_r is not None else None,
            "exact_match_rate": round(exact, 3)}


if __name__ == "__main__":
    result = judge_reply(
        "where is my order, been 5 days",
        "So sorry for the delay! Please DM us your order number and we'll check the status right away.",
        {"customer_msg": "where is my order??", "brand_reply": "Please DM your order number."},
    )
    print(result)
