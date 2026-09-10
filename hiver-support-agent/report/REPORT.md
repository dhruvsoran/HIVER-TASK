# Report — AmazonHelp Support Agent

**This version reflects a run against the real Kaggle dataset** (`twcs.csv`,
~3M rows, filtered to 153,038 real AmazonHelp `(customer, brand reply)`
pairs) and a **168-example golden set drawn from real tweets**, not the
earlier synthetic demo. Classification/drafting/judging still ran in mock
(keyword-based) mode for this specific run — see Section 0 — but the data,
taxonomy, retrieval, and golden set underneath are now real.

## 0. What actually ran, and what didn't

I was given a Gemini API key and the real dataset. The dataset processing,
clustering, retrieval, and eval harness below all ran for real against real
data. **The classification/drafting/judging LLM calls did not** — this
sandbox's network egress is restricted to a fixed allowlist that does not
include `generativelanguage.googleapis.com` (confirmed directly: a live
call with the provided key returns `HTTP 403: Forbidden`, host not in
allowlist). `src/llm_client.py` is wired to call Gemini for real and **fails
loudly** rather than silently substituting the mock when a key is present —
I chose fail-loud specifically so a run never looks real while quietly being
mock. The numbers in this report were produced by explicitly *not* setting
`GEMINI_API_KEY`, so they're honestly-labeled mock results, not partially-real
ones. Run `eval_harness.py` with the key set in an unrestricted environment
(your own machine, Colab, a VM) to get the real LLM numbers — everything
else in the pipeline is unaffected by that constraint.

## 1. Problem framing: what "good" means for this brand

Unchanged from the original framing: never invent a policy/refund/timeline
the brand hasn't actually offered before; never auto-handle real
account-security, safety, or legal-risk content; catch the routine 70-80%
so a human's time goes to the harder cases; optimize escalation for recall,
not precision.

**What changed after touching the real data:**
- **The taxonomy itself changed.** Running TF-IDF + KMeans clustering
  (k=12) on 5,000 real AmazonHelp-directed tweets surfaced `membership_issue`
  (Prime-specific complaints — unmet shipping promises, unauthorized
  sign-ups, app access) as its own clean cluster of 234/5,000 (4.7%) that
  wasn't in the original hand-designed taxonomy. Added because the data
  said so.
- **Three assumed intents didn't survive contact with real clustering.**
  `product_defect`, `positive_feedback`, and `billing_dispute` did not form
  clean top-level clusters at k=12 — they were folded into one large
  generic catch-all cluster (2,332 of 5,000 real tweets, ~47%) along with
  off-topic content, non-English tweets, and context-free fragments. They
  still exist in the data (confirmed by hand-reading 40 catch-all samples),
  just not as separable clusters with these features at this k. This is a
  genuine limitation of TF-IDF + KMeans on short, noisy tweet text, not
  something I'm papering over — see Section 4.
- **Real data is dramatically messier than the demo data in specific,
  concrete ways**: multiple languages mixed into the "single brand"
  stream (French, German, Spanish, Japanese, Hindi/Hinglish — none of
  which were in my synthetic demo), and a large fraction of single-turn
  customer tweets that are genuinely uninterpretable without the rest of
  the thread ("Yeah I tried that", "I'm waiting", "Two orders in a row.") —
  this dataset's `(customer_msg, brand_reply)` pairing only keeps one turn,
  so mid-thread replies like these lose their context entirely. Both are
  now explicit, out-of-scope categories rather than silently mishandled:
  non-English tweets are pulled into a separate documented file
  (`eval/golden_set_excluded_non_english.csv`, 7 examples), and fragments
  route to `other_unclear` → always escalate.

**Still out of scope, unchanged:** multi-turn conversation state, a
fine-tuned classifier, non-English support.

## 2. Results vs. two baselines (real data, real golden set, n=168)

Run via `eval/eval_harness.py --csv /mnt/user-data/uploads/twcs.csv --kb-sample-n 5000`.

| System | Intent accuracy | Intent macro-F1 | Escalation recall | Escalation precision | Escalation F1 |
|---|---|---|---|---|---|
| Trivial (majority class, never escalate) | 0.292 | 0.045 | 0.000 | 0.000 | 0.000 |
| Simple keyword baseline | 0.274 | 0.271 | 0.628 | 0.351 | 0.450 |
| **Agent (this system, mock-LLM mode)** | 0.274 | 0.271 | **0.721** | 0.392 | 0.508 |

Reply quality (LLM-judge, 1-5 scale, n=15 sampled): groundedness=4.21,
correctness=4.04, tone=4.47 — **these are mock-judge scores** (deterministic
placeholder, not a real quality read — see Section 0).

**Three honest observations about this table:**

1. **Intent accuracy dropped from 0.467 (synthetic demo) to 0.274 (real
   data)** — real customer language is genuinely harder for a keyword-based
   classifier than anything I wrote by hand. This is exactly the kind of
   gap a synthetic-only eval would have hidden.
2. **The trivial baseline's accuracy went UP on real data (0.167 → 0.292)**
   because the real golden set is more skewed toward `delivery_issue`
   (49/168 = 29%) than my synthetic set was. A more imbalanced real
   distribution makes "always guess the majority class" look artificially
   competitive — another reason macro-F1, not accuracy, is the number to
   trust (trivial's macro-F1 is still 0.045).
3. **Escalation recall improved further on real data** (0.721 vs. 0.692
   synthetic) — the hybrid rule layer generalizes better than raw keyword
   matching precisely because real messages contain more of the repeat-contact
   and hard-trigger language it's built to catch (real anger is messier
   but the trigger phrases are common: "again", "hacked", "unauthorized").

## 3. Failure analysis — top 5 failure modes, now with real examples

Pulled directly from `eval/eval_full_output.json` (real run) and the
confusion matrix printed by `eval_harness.py`:

**1. Massive over-prediction of `other_unclear` on real data (the #1 real
failure mode).** The confusion matrix shows `delivery_issue → other_unclear`
(27 cases), `general_complaint → other_unclear` (14 cases),
`refund_request → other_unclear` (10 cases), `account_access →
other_unclear` (7 cases) — 58 of 168 golden examples, over a third, got
mapped to the escalate-everything bucket. **Hypothesis**: the keyword
classifier's vocabulary was built from a small set of synthetic examples
and doesn't cover the actual phrasing diversity of 153,038 real tweets —
this is the single clearest piece of evidence that a real LLM (not the
keyword mock) is not optional for this system, it's load-bearing. It also
means the escalation recall numbers above are partly inflated by this
over-triggering: `other_unclear` is always-escalate by policy, so
misclassifying a routine `delivery_issue` as `other_unclear` accidentally
"correctly" escalates a gold-negative example only when the gold label was
also True — but for gold=False delivery_issue examples that got
misclassified this way, it shows up as an unnecessary escalation
(contributing to the fp=48 in the escalation table). The recall number is
real; some of what's driving it is a symptom, not a feature.

**2. `general_complaint → positive_feedback` confusion (10 cases).** Sarcastic
or backhanded phrasing containing thank-adjacent words gets misread as
genuinely positive. Real example from the catch-all cluster: a tweet
thanking the brand "for the careless delivery" is structurally a complaint
wearing a compliment's vocabulary. **Hypothesis**: any classifier that
weights keyword presence over structure/tone will make this mistake; this
is a strong argument for testing whether a real LLM actually handles
sarcasm here, rather than assuming it will.

**3. Single-turn dataset structure loses necessary context.** Roughly a
fifth of the catch-all cluster's hand-read examples were fragments that
only make sense as a reply to something earlier in the thread — "Yeah I
tried that," "I'm waiting," "Two orders in a row." **Hypothesis**: this
isn't a classifier failure, it's a data-structure limitation — the
`(customer_msg, brand_reply)` pairing this v1 scope uses discards the
preceding thread. Correctly routing these to `other_unclear`/escalate is
the right behavior given the scope, but it also means a meaningful chunk of
real traffic can never be auto-handled without multi-turn context, which
changes the realistic ceiling on auto-handle rate.

**4. Cluster impurity caused real gold-label errors that a spot-check caught.**
Of 25 spot-checked cluster-derived labels (~18.5% of that portion of the
golden set), 4 were wrong because the cluster they were drawn from was
itself impure — e.g. a delivery-location tweet landed in the
"account_access" cluster because it happened to share vocabulary
("account"), and got manually corrected during review. **This ~16%
spot-check error rate is the most important number in this whole report for
judging label quality** — it implies a meaningful fraction of the
un-spot-checked 143 examples likely carry similar noise. Documented, not
hidden — see `eval/golden_set_real.csv`'s `label_method` column, which
marks every row as either `hand-read (catch-all cluster)` or
`cluster-derived + rule-based escalation` so this uncertainty is traceable
per-example, not just asserted in prose.

**5. Multilingual traffic is a bigger real chunk than expected.** 2 of 12
clusters (Cluster 4, Cluster 5) at k=12 were dominated by non-English text
(Japanese in one, a French/Spanish/Portuguese mix in the other) — meaning
roughly a sixth of the 5,000-tweet sample used for clustering wasn't
English. **Hypothesis**: for a brand this global, "non-English support is
out of scope" (my Section 1 framing) removes a much bigger slice of real
traffic than I assumed when I wrote that scoping decision before looking at
the data.

## 4. What's misleading about my headline number

*(mandatory section — being blunt on purpose, updated for the real run)*

- **The escalation-recall headline (0.721) is inflated by failure mode #1.**
  A meaningful chunk of the "correctly escalated" cases are `other_unclear`
  misclassifications riding the always-escalate policy, not the hybrid rule
  layer actually recognizing risk. The true signal — does the system catch
  *correctly-classified* high-risk messages — needs the confusion matrix
  read alongside the recall number, not the recall number alone.
- **Intent accuracy (0.274) is a mock-classifier number, not an LLM
  number.** I could not get a real Gemini call through from this
  environment (confirmed 403, not just assumed) — every "Agent" result in
  Section 2 was produced by the same keyword logic as the "simple
  baseline," by design (see decision log), so the identical
  accuracy/macro-F1 between them is expected, not a finding that the LLM
  approach doesn't help. The real test of that claim hasn't been run.
- **The golden set's label quality has a directly-measured, non-trivial
  error rate (~16% on spot-check), not just theoretical noise.** Any single
  metric above should be read with that in mind — a 168-example set with
  ~16% label noise means point estimates could plausibly shift several
  points with a full second-labeler pass.
- **The KB was built from a 5,000-pair random sample of the 153,038 real
  pairs** (`--kb-sample-n 5000`, capped for classify-every-KB-row cost/time —
  see decision log), not the full historical set. Retrieval quality is
  bounded by what's in that sample; a rare-but-real historical resolution
  pattern not in the 5,000-pair sample simply isn't retrievable, and that's
  not visible in any of the numbers above.
- **7 non-English examples were excluded from the 168-example golden set
  and moved to a separate file** rather than folded into `other_unclear`'s
  denominator — meaning the reported accuracy/recall numbers implicitly
  assume a non-English-free traffic stream, which Section 3's finding #5
  says is not realistic for this brand.
- **LLM-judge reply-quality scores (groundedness=4.21 etc.) are pure mock
  placeholders** in this run — random-but-deterministic numbers, explicitly
  not a real quality assessment. The `agreement_report()` human-validation
  function in `eval/llm_judge.py` is wired and ready but has never been run
  against real human-assigned scores, because that requires a real drafting
  pass first.

## 5. What I'd do next with one more week

1. **Run everything in an environment where Gemini's endpoint isn't
   blocked** — this is now the single concrete, verified blocker (not a
   hypothetical), and every number in Section 2 changes once it's lifted.
2. **Do a full second-labeler pass on the golden set**, given the
   directly-measured ~16% spot-check error rate — this is no longer
   theoretical, it's the highest-leverage fix to the eval's own reliability.
3. **Re-cluster at higher k (or with sentence embeddings instead of
   TF-IDF)** specifically to try to split the 47% catch-all cluster into
   real sub-intents — `product_defect`, `positive_feedback`, and
   `billing_dispute` are known to be in there (I hand-confirmed it) but
   unsupervised structure at k=12 with TF-IDF features can't find them.
4. **Rebuild the KB from the full 153,038 pairs** (or a much larger sample)
   instead of 5,000, once a real embedding/classification budget exists —
   the current sample-size cap is a real, disclosed constraint on retrieval
   coverage.
5. **Decide what to do about the non-English fraction** — Section 3
   finding #5 suggests it's a bigger slice of real AmazonHelp traffic than
   the original scoping assumed; worth quantifying precisely (what % of
   all 153,038 real pairs are non-English) before deciding whether to keep
   excluding it or build a translation step.
6. **Extend to multi-turn threads** — failure mode #3 shows single-turn
   framing has a real, measurable cost in un-interpretable fragments, not
   just a theoretical one.
7. **Run the human-agreement check in `llm_judge.py` for real** once a real
   drafting pass exists to grade.
