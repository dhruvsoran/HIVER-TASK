"""
Runs the agent AND both baselines against the golden set, computes:
  - intent classification: accuracy, macro-F1, confusion matrix
  - escalation decision: precision/recall/F1 on escalate=True (recall is the
    headline number per the report's framing - missed escalations are the
    costly error)
  - reply quality: LLM-judge scores (groundedness/correctness/tone), averaged

Run against the demo data (zero setup):
    python eval_harness.py

Run against the real dataset for real numbers:
    python eval_harness.py --csv /mnt/user-data/uploads/twcs.csv --kb-sample-n 5000
"""

import sys
import os
import csv
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import SupportAgent
from baselines import trivial_baseline, simple_baseline
from llm_judge import judge_reply

BRAND = "AmazonHelp"


def load_golden(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["gold_escalate"] = row["gold_escalate"].strip().lower() == "true"
            rows.append(row)
    return rows


def classification_metrics(golden_intents, pred_intents):
    labels = sorted(set(golden_intents) | set(pred_intents))
    correct = sum(1 for g, p in zip(golden_intents, pred_intents) if g == p)
    accuracy = correct / len(golden_intents)

    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(golden_intents, pred_intents) if g == label and p == label)
        fp = sum(1 for g, p in zip(golden_intents, pred_intents) if g != label and p == label)
        fn = sum(1 for g, p in zip(golden_intents, pred_intents) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    confusion = defaultdict(lambda: defaultdict(int))
    for g, p in zip(golden_intents, pred_intents):
        confusion[g][p] += 1

    return {"accuracy": round(accuracy, 3), "macro_f1": round(macro_f1, 3), "confusion": confusion}


def escalation_metrics(gold_escalate, pred_escalate):
    tp = sum(1 for g, p in zip(gold_escalate, pred_escalate) if g and p)
    fp = sum(1 for g, p in zip(gold_escalate, pred_escalate) if not g and p)
    fn = sum(1 for g, p in zip(gold_escalate, pred_escalate) if g and not p)
    tn = sum(1 for g, p in zip(gold_escalate, pred_escalate) if not g and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def print_top_confusions(confusion, n=8):
    pairs = []
    for gold, preds in confusion.items():
        for pred, count in preds.items():
            if gold != pred:
                pairs.append((count, gold, pred))
    pairs.sort(reverse=True)
    print(f"Top {n} confusions (gold -> predicted):")
    for count, gold, pred in pairs[:n]:
        print(f"  {gold:22s} -> {pred:22s}  x{count}")


def run_eval(golden_path, csv_path, kb_sample_n, sample_judge_n=15):
    golden = load_golden(golden_path)
    golden_msgs = [r["customer_msg"] for r in golden]
    golden_intents = [r["gold_intent"] for r in golden]
    gold_escalate = [r["gold_escalate"] for r in golden]

    print(f"Golden set: {golden_path}  ({len(golden)} examples)")
    print(f"Data source: {csv_path}\n")

    agent = SupportAgent(csv_path, BRAND, kb_sample_n=kb_sample_n)
    agent_results = [agent.handle(msg) for msg in golden_msgs]
    agent_intents = [r["intent"] for r in agent_results]
    agent_escalate = [r["escalate"] for r in agent_results]

    trivial_results = trivial_baseline(golden_msgs, golden_intents)
    simple_results = simple_baseline(golden_msgs)

    print("=" * 70)
    print("INTENT CLASSIFICATION")
    print("=" * 70)
    agent_metrics = None
    for name, preds in [
        ("Trivial baseline", [r["intent"] for r in trivial_results]),
        ("Simple keyword baseline", [r["intent"] for r in simple_results]),
        ("Agent (LLM classifier)", agent_intents),
    ]:
        m = classification_metrics(golden_intents, preds)
        if name.startswith("Agent"):
            agent_metrics = m
        print(f"{name:28s} accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")
    print()
    print_top_confusions(agent_metrics["confusion"])

    print()
    print("=" * 70)
    print("ESCALATION DECISION (recall on escalate=True is the headline metric)")
    print("=" * 70)
    for name, preds in [
        ("Trivial baseline", [r["escalate"] for r in trivial_results]),
        ("Simple keyword baseline", [r["escalate"] for r in simple_results]),
        ("Agent (hybrid rules+confidence)", agent_escalate),
    ]:
        m = escalation_metrics(gold_escalate, preds)
        print(f"{name:32s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})")

    print()
    print("=" * 70)
    print(f"REPLY QUALITY (LLM-judge, sampled n={sample_judge_n} of {len(golden)} for cost)")
    print("=" * 70)
    judge_scores = []
    for r in agent_results[:sample_judge_n]:
        exemplar = r["exemplars_used"][0] if r["exemplars_used"] else None
        score = judge_reply(r["customer_msg"], r["draft_reply"], {"customer_msg": exemplar} if exemplar else None)
        judge_scores.append(score)

    def avg(key):
        vals = [s[key] for s in judge_scores if s.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    print(f"avg groundedness={avg('groundedness')}  avg correctness={avg('correctness')}  avg tone={avg('tone')}")

    print()
    print("NOTE: full confusion matrix and per-example agent outputs are in eval_full_output.json")

    out = {
        "golden": golden,
        "agent_results": agent_results,
        "trivial_results": trivial_results,
        "simple_results": simple_results,
        "judge_scores": judge_scores,
    }
    out_path = os.path.join(os.path.dirname(__file__), "eval_full_output.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_golden = os.path.join(os.path.dirname(__file__), "golden_set_real.csv")
    default_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_tweets.csv")
    ap.add_argument("--golden", default=default_golden)
    ap.add_argument("--csv", default=default_csv,
                     help="Point at the real twcs.csv for real numbers; defaults to the small demo file.")
    ap.add_argument("--kb-sample-n", type=int, default=3000)
    ap.add_argument("--judge-n", type=int, default=15)
    args = ap.parse_args()
    run_eval(args.golden, args.csv, args.kb_sample_n, args.judge_n)
