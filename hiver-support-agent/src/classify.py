"""
Stage 1: intent classification.

Taxonomy V2 - revised from the ORIGINAL plan after actually running TF-IDF +
KMeans clustering on 5,000 real AmazonHelp-directed customer tweets
(scripts/build_taxonomy.py). Two real findings from that process:

  1. "membership_issue" was NOT in the original hand-designed taxonomy. It
     emerged as its own clean cluster (n=234 of 5,000, ~4.7%) - Prime-specific
     complaints (delivery-speed promise unmet, unauthorized sign-up, app
     access) that don't cleanly fit "billing_dispute" or "delivery_issue".
     Added because the data said so, not because it was anticipated.

  2. "product_defect", "positive_feedback", and "billing_dispute" did NOT
     form their own clean top-level clusters at k=12 - they were folded into
     one large generic catch-all cluster (2,332 of 5,000 - nearly half the
     sample) alongside off-topic, non-English, and fragment-only tweets.
     They still exist in the data (confirmed by hand-reading catch-all
     samples - see eval/golden_set_real.csv), just not as separable
     clusters at this k with TF-IDF features. This is reported as a real
     limitation in report/REPORT.md, not hidden.

Final taxonomy:
  order_status         - where is my order / tracking / cancel/change order
  delivery_issue        - late, lost, misdelivered, or damaged-in-transit packages
  refund_request         - explicit ask for money back / duplicate charge
  account_access         - login, lockout, suspension, hacked account
  product_defect          - item (physical or digital/app) broken or malfunctioning
  billing_dispute          - unauthorized/incorrect charges, invoice questions
  membership_issue          - Prime-specific: signup, cancellation, unmet shipping promise
  positive_feedback          - compliments, thanks, no action needed
  general_complaint           - frustration/anger without one clear actionable ask
  other_unclear                 - fragment, off-topic, or ambiguous; always escalated
"""

import json
from llm_client import call_llm

INTENTS = [
    "order_status", "delivery_issue", "refund_request", "account_access",
    "product_defect", "billing_dispute", "membership_issue",
    "positive_feedback", "general_complaint", "other_unclear",
]

SYSTEM_PROMPT = f"""TASK:CLASSIFY
You are an intent classifier for customer support tweets to Amazon's Twitter support account.
Classify the customer's message into exactly one of these intents:
{", ".join(INTENTS)}

Respond ONLY with JSON: {{"intent": "<intent>", "confidence": <float 0-1>}}
If the message doesn't clearly fit any intent (fragment, off-topic, non-English, or otherwise
ambiguous), use "other_unclear" with low confidence rather than forcing a fit.
"""


def classify_intent(text: str):
    raw = call_llm(SYSTEM_PROMPT, text)
    try:
        data = json.loads(raw)
        intent = data.get("intent", "other_unclear")
        confidence = float(data.get("confidence", 0.5))
        if intent not in INTENTS:
            intent = "other_unclear"
        return intent, confidence
    except Exception:
        return "other_unclear", 0.3


if __name__ == "__main__":
    tests = [
        "where is my order?? it was supposed to arrive 3 days ago",
        "thanks for the quick help, appreciate it!",
        "my account got hacked, someone placed orders I never made",
    ]
    for t in tests:
        print(t, "->", classify_intent(t))
