"""
Stage 3: escalate-or-auto-handle decision.

Deliberately a HYBRID of hard rules + model confidence, not a pure LLM call -
see report.md "what good means for this brand": missed escalations (a message
that needed a human but got auto-handled) are the costly error, so recall on
escalation is prioritized over precision, and rules act as a recall floor
that no confidence score can override.
"""

import re

CONFIDENCE_THRESHOLD = 0.65

ALWAYS_ESCALATE_INTENTS = {"account_access", "other_unclear"}

HARD_TRIGGER_PHRASES = [
    "lawyer", "legal action", "sue", "sued", "hurt", "injury", "injured",
    "unsafe", "fraud", "unauthorized", "hacked", "hack",
]

REPEAT_CONTACT_MARKERS = [
    "contacted you", "called twice", "called 3 times", "second time",
    "again", "still no", "already told you",
]


def _word_match(phrase: str, text: str) -> bool:
    """
    Word-boundary match instead of substring match. Substring matching was
    firing on 'sue' inside 'issues', 'hack' inside 'backpack', etc - a real
    bug found during evaluation on the actual Kaggle dataset (see report.md
    failure analysis). Multi-word phrases still match as a substring since
    \\b doesn't apply cleanly across spaces in the same way single words need it.
    """
    if " " in phrase:
        return phrase in text
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def should_escalate(customer_msg: str, intent: str, intent_confidence: float):
    text = customer_msg.lower()
    reasons = []

    for phrase in HARD_TRIGGER_PHRASES:
        if _word_match(phrase, text):
            reasons.append(f"hard-trigger phrase matched: '{phrase}'")

    if intent in ALWAYS_ESCALATE_INTENTS:
        reasons.append(f"intent '{intent}' is always-escalate by policy")

    if intent_confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"classifier confidence {intent_confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}")

    for marker in REPEAT_CONTACT_MARKERS:
        if marker in text:
            reasons.append(f"repeat-contact language detected: '{marker}'")

    escalate = len(reasons) > 0
    reason_str = "; ".join(reasons) if reasons else "no escalation triggers; routine intent with high confidence"
    return escalate, reason_str


if __name__ == "__main__":
    print(should_escalate("account got hacked, orders I never made", "account_access", 0.9))
    print(should_escalate("where's my order, tracking says delivered yesterday", "order_status", 0.88))
    print(should_escalate("thanks so much, love the fast shipping!", "positive_feedback", 0.91))
