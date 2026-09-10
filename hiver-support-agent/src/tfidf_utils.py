"""
Shared TF-IDF vectorizer utilities. Used for both:
  (a) unsupervised clustering to derive the intent taxonomy from real data
  (b) retrieval similarity in kb_builder.py (replaces the old difflib
      text-overlap proxy - see decision log for why TF-IDF over full neural
      embeddings: zero heavy dependencies, runs fast on 20k+ docs, still a
      real improvement over character-overlap matching).
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"@\w+", " ", t)          # strip @mentions (always present, no signal)
    t = re.sub(r"https?://\S+", " ", t)  # strip URLs
    t = re.sub(r"[^a-z0-9'\s]", " ", t)  # strip punctuation/emoji
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fit_vectorizer(texts, max_features=5000):
    cleaned = [clean_text(t) for t in texts]
    vec = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(cleaned)
    return vec, matrix


def top_terms_per_cluster(vec, matrix, labels, n_clusters, top_n=12):
    import numpy as np
    terms = vec.get_feature_names_out()
    result = {}
    for c in range(n_clusters):
        idx = labels == c
        if idx.sum() == 0:
            result[c] = []
            continue
        mean_tfidf = matrix[idx].mean(axis=0)
        mean_tfidf = np.asarray(mean_tfidf).flatten()
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        result[c] = [terms[i] for i in top_idx]
    return result
