"""
Builds the "resolution KB": every historical (customer_msg, brand_reply) pair
tagged with an intent label, so draft.py can retrieve similar past resolutions
for a new incoming message.

Retrieval uses TF-IDF cosine similarity (via tfidf_utils.py), restricted to
same-intent history before ranking - a real improvement over a naive
character-overlap proxy, fit once per KB build rather than per query.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from classify import classify_intent
from tfidf_utils import fit_vectorizer


def build_kb(pairs_df: pd.DataFrame, sample_n: int = None, random_state: int = 42) -> pd.DataFrame:
    """
    sample_n: if set, build the KB from a random sample of this many pairs
    instead of the full set. Necessary at real-dataset scale (AmazonHelp
    alone has ~150K historical pairs) - classifying every KB row to tag it
    with an intent is one LLM call per row, so an unsampled KB against a
    real API key would be slow and expensive to rebuild. See decision log.
    """
    kb = pairs_df if sample_n is None or len(pairs_df) <= sample_n else pairs_df.sample(
        n=sample_n, random_state=random_state
    ).reset_index(drop=True)
    kb = kb.copy()
    intents, confidences = [], []
    for msg in kb["customer_msg"]:
        intent, conf = classify_intent(msg)
        intents.append(intent)
        confidences.append(conf)
    kb["intent"] = intents
    kb["intent_confidence"] = confidences
    return kb


class RetrievalIndex:
    """Fits TF-IDF once over the KB; reused across many retrieve() calls."""

    def __init__(self, kb: pd.DataFrame):
        self.kb = kb.reset_index(drop=True)
        self.vectorizer, self.matrix = fit_vectorizer(self.kb["customer_msg"].tolist())

    def retrieve(self, query: str, intent: str, k: int = 3):
        mask = (self.kb["intent"] == intent).values
        if not mask.any():
            mask = np.ones(len(self.kb), dtype=bool)  # fallback: no same-intent history

        from tfidf_utils import clean_text
        q_vec = self.vectorizer.transform([clean_text(query)])
        sims = cosine_similarity(q_vec, self.matrix[mask]).flatten()

        sub = self.kb[mask].reset_index(drop=True)
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            results.append({
                "customer_msg": sub.loc[i, "customer_msg"],
                "brand_reply": sub.loc[i, "brand_reply"],
                "sim": float(sims[i]),
            })
        return results


def retrieve_exemplars(kb: pd.DataFrame, query: str, intent: str, k: int = 3):
    """
    Convenience one-shot wrapper (fits a fresh index every call - fine for a
    single lookup or a script, but SupportAgent in pipeline.py builds a
    RetrievalIndex once and reuses it for efficiency across many messages).
    """
    return RetrievalIndex(kb).retrieve(query, intent, k)


if __name__ == "__main__":
    from thread_builder import load_pairs
    pairs = load_pairs("/home/claude/hiver-support-agent/data/sample_tweets.csv", "AmazonHelp")
    kb = build_kb(pairs)
    print(kb[["customer_msg", "intent", "intent_confidence"]].to_string())
