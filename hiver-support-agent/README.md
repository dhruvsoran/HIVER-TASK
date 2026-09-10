# Hiver SDE Intern Take-Home — AI Support Agent (AmazonHelp)

An AI support agent for **AmazonHelp** — confirmed the highest-volume brand
in the dataset (169,840 replies) — that classifies incoming customer tweets
into intents, drafts a reply grounded in how the brand has historically
resolved similar issues, and decides auto-handle vs. escalate with a stated
reason.

This has been run against the **real Kaggle dataset** (153,038 real
`(customer, brand reply)` pairs extracted from the full 3M-row `twcs.csv`)
and a **250-example golden set sampled and labeled from real tweets** (150
hand-labelled, 100 cluster-derived). See `report/REPORT.md` for the real
results and a full accounting of what's real vs. what's still mock in this
specific environment.

## Quickstart (zero setup, mock mode)

```bash
pip install -r requirements.txt
cd src
python pipeline.py "where is my order?? been 5 days"
python pipeline.py                                    # demo batch of 5
```

## Run against the real dataset

The repository includes pre-built data files extracted from the real Kaggle
dataset, so you can run immediately without downloading the full 3M-row file:

```bash
cd src
python pipeline.py --csv ../data/amazonhelp_kb_pool.csv "where is my order"
```

If you want to regenerate from scratch using the full Kaggle dataset:

1. Download `twcs.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
2. Place it in the project root or specify the path:

```bash
cd scripts
python build_real_dataset.py --csv /path/to/twcs.csv --brand AmazonHelp \
    --kb-size 20000 --eval-pool-size 5000
```

## Reproduce the headline eval results (~2-3 min)

Using the pre-built data (no download needed):

```bash
cd eval
python eval_harness.py                                                        # demo data
python eval_harness.py --csv ../data/amazonhelp_kb_pool.csv --kb-sample-n 5000  # real data
```

Using the full Kaggle dataset:

```bash
cd eval
python eval_harness.py --csv /path/to/twcs.csv --kb-sample-n 5000
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

**Note on rate limits:** The free-tier Gemini API has strict rate limits.
For full eval runs (250+ examples), you may need to:
- Use a paid API tier with higher rate limits
- Run with `--mock-kb` flag to use mock classifier for KB building:
  ```bash
  python eval/eval_harness.py --csv data/amazonhelp_kb_pool.csv --kb-sample-n 500 --mock-kb
  ```
- Or run in an environment with higher rate limits (your own machine, Colab, a VM)

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
  golden_set_real.csv               # 250 real examples (150 hand-labelled, 100 cluster-derived)
  golden_set_excluded_non_english.csv # 7 real non-English examples, documented as out of scope
  golden_set.csv                     # original 30-example synthetic demo set (kept for reference)
  baselines.py                        # trivial + simple keyword baselines
  llm_judge.py                          # LLM-as-judge + human-agreement harness (validated with 5 examples)
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
