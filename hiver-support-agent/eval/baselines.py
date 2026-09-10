"""
Two baselines the agent must beat:

1. TRIVIAL: always predict the most frequent intent in the golden set,
   always auto-handle (never escalate).
2. SIMPLE: keyword/regex intent classifier (same keyword table the mock LLM
   uses internally - representative of what a rule-based v0 would look like)
   + template reply by intent + rule-based escalation (hard triggers only,
   no confidence gating since there's no model confidence to gate on).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter
from llm_client import _INTENT_KEYWORDS, _ESCALATE_TRIGGERS  # reuse keyword tables


TEMPLATE_REPLIES = {
    "order_status": "Please DM your order number and we'll check the status.",
    "delivery_issue": "Sorry about that - please DM your order number so we can investigate with the carrier.",
    "refund_request": "Please DM your order ID and we'll look into a refund.",
    "account_access": "For account security please contact us via the app to verify your identity.",
    "product_defect": "Sorry to hear that - please DM your order number and photos.",
    "billing_dispute": "Please DM your account email and we'll review the billing history.",
    "membership_issue": "Please DM your account details so we can look into your Prime membership.",
    "positive_feedback": "Thanks so much, glad we could help!",
    "general_complaint": "We're sorry for the frustration - please DM your order number so we can help.",
    "other_unclear": "Thanks for reaching out - please DM us more details so we can help.",
}


def trivial_baseline(golden_msgs, golden_intents):
    most_common_intent = Counter(golden_intents).most_common(1)[0][0]
    results = []
    for msg in golden_msgs:
        results.append({
            "intent": most_common_intent,
            "escalate": False,
            "reply": TEMPLATE_REPLIES.get(most_common_intent, "Please DM us for help."),
        })
    return results


def simple_keyword_classify(text):
    t = text.lower()
    scores = {intent: sum(1 for kw in kws if kw in t) for intent, kws in _INTENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other_unclear"
    return best


def simple_baseline(golden_msgs):
    results = []
    for msg in golden_msgs:
        intent = simple_keyword_classify(msg)
        escalate = any(trig in msg.lower() for trig in _ESCALATE_TRIGGERS) or intent == "other_unclear" or intent == "account_access"
        results.append({
            "intent": intent,
            "escalate": escalate,
            "reply": TEMPLATE_REPLIES.get(intent, "Please DM us for help."),
        })
    return results
