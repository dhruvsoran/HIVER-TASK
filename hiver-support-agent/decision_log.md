# Decision Log

Plain list of non-obvious decisions and why.

1. **Chose AmazonHelp over an airline brand.** Airlines introduce
   flight-specific regulatory edge cases (EU261, weather delays, rebooking
   rules) that would balloon the intent taxonomy and escalation rules beyond
   what a one-week project can responsibly cover.

2. **Split classification and drafting into two separate LLM calls**, not
   one combined prompt. This lets the eval harness attribute failures to
   "wrong intent" vs. "right intent but bad reply" instead of one opaque
   end-to-end score, which matters far more for debugging than the extra
   latency/cost of a second call.

3. **Escalation is a hybrid of hard rules + intent policy + classifier
   confidence, not a pure LLM judgment call.** A pure LLM escalation
   decision is exactly the kind of decision I don't want silently drifting
   between prompt versions — hard-coded trigger phrases (legal threats,
   safety language, account-hack language) act as a floor no model
   confidence score can override.

4. **Escalation optimizes for recall over precision on `escalate=True`.**
   The cost asymmetry is real: an unnecessary escalation costs a human a few
   minutes of triage; a missed escalation on an account-security or
   safety-risk message costs trust and potentially real harm. Stated
   explicitly rather than left implicit in a single F1 number.

5. **`other_unclear` is an explicit, always-escalate bucket** rather than
   forcing every message into one of the 8 "real" intents. Refusing to
   force-fit ambiguous input is a taxonomy-design choice, not a cop-out —
   forcing a fit would silently corrupt the retrieval KB with mislabeled
   exemplars.

6. **Retrieval is restricted to same-intent history before ranking by
   similarity**, not global similarity search. A reply grounded in a
   textually-similar but wrong-*kind*-of-problem exemplar is worse than one
   grounded in an on-topic but less textually similar exemplar.

7. **Used macro-F1 alongside accuracy for intent classification**, not
   accuracy alone, because the golden set (and almost certainly real
   traffic) is imbalanced toward order-status/delivery-issue style
   complaints — accuracy alone would hide poor performance on rarer intents
   like account_access.

8. **Built an offline deterministic mock-LLM mode** rather than requiring an
   API key to run anything. A take-home that only works with a paid key the
   grader may not have configured is a bad experience; the mock mode is
   explicitly and repeatedly flagged as a stand-in, not a real result.

9. **Kept the mock classifier keyword-based on purpose**, matching what a
   naive rule-based v0 would look like, so that even in offline mode the
   "agent vs. simple baseline" comparison in the eval harness stays
   structurally meaningful (it correctly shows they tie on intent accuracy
   in mock mode — see report Section 4 — rather than faking a fake win).

10. **Golden set is 30 hand-written synthetic examples, not the full
    150-250**, and this is disclosed prominently in the README and report
    rather than padded to look complete. Sampling method: I wrote messages
    covering all 9 intents plus known-hard edge cases (sarcasm-adjacent
    phrasing, ambiguous fragments, safety/legal trigger language,
    repeat-contact language), rather than randomly sampling nonexistent real
    traffic.

11. **No second labeler for the golden set.** I'm the only labeler; the
    report states this as a limitation rather than reporting a fabricated
    inter-annotator agreement number.

12. **LLM-judge validation harness is wired but not executed** against real
    human scores, because doing so honestly requires a real drafting pass
    with a real model first — running it against mock-mode outputs would
    produce a validation number that validates nothing.

13. **Used simple text-overlap (`difflib`) for retrieval similarity instead
    of embeddings**, to keep the whole pipeline runnable with zero extra
    dependencies and zero API calls for retrieval specifically. Flagged
    explicitly as the single highest-value upgrade for reply quality.

14. **Dropped multi-turn thread handling from v1 scope.** `thread_builder.py`
    only keeps one customer message → one direct brand reply per pair. Real
    threads can run 3-5 turns; handling that properly needs conversation
    state design that didn't fit a one-week first pass.

15. **Kept `TEMPLATE_REPLIES` in baselines.py deliberately dumb** (one fixed
    string per intent, no personalization) so the comparison against the
    agent's grounded, retrieval-conditioned draft is a fair "templated vs.
    grounded" test, not a strawman.

## Additions after processing the real dataset

16. **Chose fail-loud over silent-fallback when a real API key is present
    but the call fails.** `llm_client.py` raises a clear `RuntimeError`
    rather than quietly substituting mock output — a silently-mocked "real"
    run would produce eval numbers that look real but aren't, with no
    signal to the person running it. Verified directly in this environment:
    a real Gemini call with the provided key returns `HTTP 403 Forbidden`
    (host not in this sandbox's network allowlist), and the pipeline
    correctly surfaces that instead of hiding it.

17. **Split real data into disjoint KB-pool and eval-pool before sampling
    the golden set**, so golden-set customer messages aren't drawn from the
    same pool used to build the retrieval KB. Avoids a trivial "the exact
    exemplar for this test question is sitting in the KB" scenario.

18. **Capped the KB build at a random sample of the real pairs
    (`--kb-sample-n`, default 3000-5000) rather than using all 153,038.**
    Classifying every KB row to tag it with an intent is one LLM call per
    row; at real-API cost and the 15-minute reproducibility target, an
    unsampled KB is not tractable. Disclosed explicitly as a bound on
    retrieval coverage, not hidden as if the full history were used.

19. **Derived the taxonomy from real TF-IDF+KMeans clustering, not just
    read off the original plan** — and kept the result even where it
    contradicted the original design (added `membership_issue`; found that
    `product_defect`/`positive_feedback`/`billing_dispute` don't separate
    cleanly from a large catch-all cluster at k=12). Reporting the
    taxonomy that survived contact with data, not the one drafted before
    seeing it.

20. **Golden set labeling method is per-example traceable, not a single
    blanket claim.** Every row in `golden_set_real.csv` has a `label_method`
    column: `hand-read (catch-all cluster)` for 40 examples I read and
    labeled individually, or `cluster-derived + rule-based escalation` for
    135 examples where the intent came from the cluster's assigned name and
    escalation came from a first-pass rule application. A stratified
    18.5% spot-check of the second group found a ~16% error rate from
    cluster impurity, which was corrected where found and is reported as a
    property of the whole set, not swept into a vague "may contain noise"
    disclaimer.

21. **Excluded non-English examples into a separate file
    (`golden_set_excluded_non_english.csv`) rather than dropping them
    silently or force-labeling them `other_unclear`.** Keeps the scoping
    decision (no non-English support) auditable and quantifiable rather
    than invisible.

22. **Replaced `difflib` retrieval with TF-IDF cosine similarity
    (`RetrievalIndex` in `kb_builder.py`), fit once per KB build and reused
    across every subsequent message** rather than re-fit per query. A real,
    verified improvement over the earlier character-overlap proxy, without
    taking on a heavy embedding-model dependency — flagged in the report as
    still a step below real sentence embeddings, which remain the next
    upgrade if there's time/budget for the dependency.

23. **Rewrote `thread_builder.py`'s pair extraction to be fully vectorized**
    (pandas merge, not `iterrows()`) after confirming the original loop-based
    version would not finish in reasonable time against the real 3M-row
    file. Verified: full extraction for AmazonHelp (153,038 pairs) completes
    in under 30 seconds.
