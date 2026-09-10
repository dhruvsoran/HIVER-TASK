"""
End-to-end agent: message in -> {intent, draft reply, escalate decision+reason} out.

Usage:
    python pipeline.py "where is my order?? been 5 days"          # demo data (zero setup)
    python pipeline.py                                              # demo batch of 5
    python pipeline.py --csv /path/to/twcs.csv --kb-sample-n 5000    # real data
"""

import sys
import os
import json
import argparse

from thread_builder import load_pairs
from kb_builder import build_kb, RetrievalIndex
from classify import classify_intent
from draft import draft_reply
from escalate import should_escalate


class SupportAgent:
    def __init__(self, csv_path: str, brand_handle: str, k_exemplars: int = 3, kb_sample_n: int = 3000):
        pairs = load_pairs(csv_path, brand_handle)
        if pairs.empty:
            raise ValueError(f"No (customer, brand_reply) pairs found for brand '{brand_handle}' in {csv_path}")
        # kb_sample_n keeps KB-building tractable at real-dataset scale
        # (AmazonHelp alone has ~150K historical pairs - see decision log)
        self.kb = build_kb(pairs, sample_n=kb_sample_n)
        self.index = RetrievalIndex(self.kb)  # fit TF-IDF once, reuse for every message
        self.k_exemplars = k_exemplars

    def handle(self, customer_msg: str) -> dict:
        intent, confidence = classify_intent(customer_msg)
        exemplars = self.index.retrieve(customer_msg, intent, k=self.k_exemplars)
        reply = draft_reply(customer_msg, exemplars)
        escalate, reason = should_escalate(customer_msg, intent, confidence)
        return {
            "customer_msg": customer_msg,
            "intent": intent,
            "intent_confidence": confidence,
            "draft_reply": reply,
            "escalate": escalate,
            "escalate_reason": reason,
            "exemplars_used": [e["customer_msg"] for e in exemplars],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?", help="single customer message to process")
    default_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_tweets.csv")
    parser.add_argument("--csv", default=default_csv,
                         help="Raw Kaggle-format CSV. Defaults to the small synthetic demo file. "
                              "Point at the real twcs.csv for real numbers.")
    parser.add_argument("--brand", default="AmazonHelp")
    parser.add_argument("--kb-sample-n", type=int, default=3000,
                         help="Cap on historical pairs used to build the KB (real dataset has ~150K for AmazonHelp)")
    args = parser.parse_args()

    agent = SupportAgent(args.csv, args.brand, kb_sample_n=args.kb_sample_n)

    if args.message:
        result = agent.handle(args.message)
        print(json.dumps(result, indent=2))
    else:
        demo_msgs = [
            "where is my order, it's been 5 days and tracking hasn't updated",
            "I've been charged twice for the same purchase, need this fixed now",
            "someone hacked my account and placed orders I never made!",
            "just wanted to say thanks, delivery was super fast this time",
            "this is the third time I've contacted you about my broken blender, unacceptable",
        ]
        for msg in demo_msgs:
            print(json.dumps(agent.handle(msg), indent=2))
            print("-" * 60)


if __name__ == "__main__":
    main()
