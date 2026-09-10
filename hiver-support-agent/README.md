# Hiver SDE Intern Take-Home — AI Support Agent (AmazonHelp)

An AI support agent for **AmazonHelp** — confirmed the highest-volume brand
in the dataset (169,840 replies) — that classifies incoming customer tweets
into intents, drafts a reply grounded in how the brand has historically
resolved similar issues, and decides auto-handle vs. escalate with a stated
reason.

This has been run against the **real Kaggle dataset** (153,038 real
`(customer, brand reply)` pairs extracted from the full 3M-row `twcs.csv`)
and a **168-example golden set sampled and labeled from real tweets**. See
`report/REPORT.md` for the real results and a full accounting of what's
real vs. what's still mock in this specific environment.

## ⚠️ One confirmed, disclosed limitation: Gemini network access

I was given a Gemini API key. **A live call from this sandbox returns
`HTTP 403 Forbidden`** — its network egress is restricted to a fixed
allowlist that doesn't include `generativelanguage.googleapis.com`. This
is verified, not assumed (see `report/REPORT.md` Section 0). The Gemini
integration in `src/llm_client.py` is fully implemented and will work in
any environment with normal network access — it **fails loudly** with a
clear error rather than silently substituting mock output, so a real key
either gets you real results or an obvious error, never a result that
looks real but isn't.

Every eval number in `report/REPORT.md` was produced by explicitly *not*
setting `GEMINI_API_KEY` (mock mode) against the **real dataset** — so the
data/taxonomy/retrieval/golden-set are real, the LLM calls are not. Set the
key in an unrestricted environment to get the real classification/drafting
numbers with zero code changes.

## Quickstart (zero setup, mock mode)

```bash
pip install -r requirements.txt
cd src
python pipeline.py "where is my order?? been 5 days"
python pipeline.py                                    # demo batch of 5
```

## Run against the real dataset

```bash
cd src
python pipeline.py --csv /path/to/twcs.csv --kb-sample-n 5000 "where is my order"
```

`thread_builder.py` extracts real `(customer_msg, brand_reply)` pairs
directly from the raw Kaggle export in under 30 seconds (vectorized pandas,
not a row-by-row loop).

## Reproduce the headline eval results (~2-3 min on real data)

```bash
cd eval
python eval_harness.py                                                        # demo data
python eval_harness.py --csv /path/to/twcs.csv --kb-sample-n 5000             # real data, real numbers
```

Prints intent-classification accuracy/macro-F1 + confusion matrix,
escalation precision/recall/F1, and LLM-judge reply-quality scores; dumps
full per-example detail to `eval_full_output.json`.

## Using a real LLM

```bash
export GEMINI_API_KEY=your-key-here     # or ANTHROPIC_API_KEY
python pipeline.py "where is my order?"
```

No code changes needed — `src/llm_client.py` auto-detects the key.
Anthropic is tried first if both are set; Gemini next; otherwise mock.

## Regenerating the real-data artifacts from scratch

```bash
cd scripts
python build_real_dataset.py --csv /path/to/twcs.csv --brand AmazonHelp \
    --kb-size 20000 --eval-pool-size 5000
python build_taxonomy.py --csv ../data/amazonhelp_eval_pool.csv --n-clusters 12
```

The first script extracts real pairs and splits them into a disjoint
KB-pool and eval-pool (so golden-set questions aren't drawn from the same
pool as the retrieval KB). The second runs TF-IDF + KMeans clustering and
prints each cluster's top terms + sample tweets for taxonomy derivation —
this is exactly how `membership_issue` was discovered as a real, distinct
intent not in the original hand-designed taxonomy (see `decision_log.md`).

## Repo structure

```
src/
  llm_client.py      # LLM abstraction: Gemini/Anthropic if key set (fail-loud on error), else mock
  thread_builder.py   # vectorized raw-tweets -> (customer_msg, brand_reply) pairs, scales to 3M rows
  tfidf_utils.py        # shared TF-IDF vectorizer utilities (clustering + retrieval)
  kb_builder.py           # tags historical pairs with intent; TF-IDF cosine retrieval index
  classify.py               # stage 1: intent classification (10-intent taxonomy, revised from real clustering)
  draft.py                    # stage 2: RAG-grounded reply drafting
  escalate.py                   # stage 3: hybrid rule+confidence escalation decision
  pipeline.py                     # end-to-end entrypoint
scripts/
  build_real_dataset.py  # extracts real AmazonHelp pairs from twcs.csv, splits KB/eval pools
  build_taxonomy.py       # TF-IDF + KMeans clustering to derive/validate the taxonomy from real data
eval/
  golden_set_real.csv               # 168 real examples, per-row label_method column, spot-check corrected
  golden_set_excluded_non_english.csv # 7 real non-English examples, documented as out of scope
  golden_set.csv                     # original 30-example synthetic demo set (kept for reference)
  baselines.py                        # trivial + simple keyword baselines
  llm_judge.py                          # LLM-as-judge + human-agreement harness (wired, not yet run for real)
  eval_harness.py                        # runs everything, prints headline metrics + confusion matrix
data/
  sample_tweets.csv           # tiny synthetic demo file (Kaggle export format) for zero-setup runs
  amazonhelp_kb_pool.csv        # 20,000 real pairs (KB source pool, disjoint from eval pool)
  amazonhelp_eval_pool.csv        # 5,000 real pairs (golden-set source pool)
  amazonhelp_eval_pool_clustered.csv # eval pool + TF-IDF/KMeans cluster assignments
report/
  REPORT.md   # real results, honest confusion-matrix-level failure analysis, misleading-number section
decision_log.md
```

## Why the pipeline is shaped this way (short version — see decision_log.md)

- Classification and drafting are **separate LLM calls**, so eval can
  attribute an error to "wrong intent" vs. "right intent, bad reply."
- Escalation is a **hybrid** of hard keyword rules + intent policy +
  classifier confidence — recall on `escalate=True` is the metric that
  matters, and rules act as a floor no confidence score overrides.
- Retrieval uses **TF-IDF cosine similarity** restricted to same-intent
  history, fit once per KB build and reused — real embeddings are the
  next upgrade, flagged explicitly in the report.
- The taxonomy was **derived from real clustering, not designed up front**
  — and kept the result even where it contradicted the original plan (see
  decision log #19).
