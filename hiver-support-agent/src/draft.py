"""
Stage 2: draft a reply grounded in retrieved historical resolutions for the
same intent. Kept as a separate LLM call from classification deliberately -
see decision log item on error attribution.
"""

import json
from llm_client import call_llm

SYSTEM_PROMPT = """TASK:DRAFT
You are drafting a customer support reply on behalf of a brand's Twitter support account.
You will be given the customer's message and 1-3 examples of how the brand has
historically resolved similar issues (real past replies).

Rules:
- Match the brand's tone and structure from the examples.
- Never invent policy, refund amounts, or timelines not supported by the examples.
- If the examples don't cover the specifics, keep the reply generic and direct
  the customer to DM order details rather than guessing.
- Keep it under 280 characters, Twitter-reply style.
- Do not promise anything the exemplars don't support.
"""


def draft_reply(customer_msg: str, exemplars: list):
    user_content = f"{customer_msg}\n---EXEMPLARS---\n{json.dumps(exemplars)}"
    return call_llm(SYSTEM_PROMPT, user_content, max_tokens=200)


if __name__ == "__main__":
    from thread_builder import load_pairs
    from kb_builder import build_kb, retrieve_exemplars
    from classify import classify_intent

    pairs = load_pairs("/home/claude/hiver-support-agent/data/sample_tweets.csv", "AmazonHelp")
    kb = build_kb(pairs)

    query = "hey where is my package, it's been 5 days and tracking hasn't updated"
    intent, conf = classify_intent(query)
    exemplars = retrieve_exemplars(kb, query, intent, k=2)
    reply = draft_reply(query, exemplars)
    print("Intent:", intent, conf)
    print("Exemplars used:", [e["customer_msg"] for e in exemplars])
    print("Draft reply:", reply)
