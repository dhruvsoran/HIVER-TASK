"""
Derives the intent taxonomy FROM the real data via TF-IDF + KMeans clustering,
per the methodology in the original plan: embed, cluster, read cluster
samples, name clusters, collapse to a small taxonomy.

This prints each cluster's top TF-IDF terms and a handful of real example
messages so a human (me, acting as the labeler) can assign an intent name to
each cluster - a genuine data-driven step, not a taxonomy invented up front
and imposed on the data.

Run:
    python build_taxonomy.py --csv ../data/amazonhelp_eval_pool.csv --n-clusters 12
"""

import argparse
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tfidf_utils import fit_vectorizer, top_terms_per_cluster
from sklearn.cluster import KMeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--samples-per-cluster", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    texts = df["customer_msg"].astype(str).tolist()

    vec, matrix = fit_vectorizer(texts)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(matrix)
    df["cluster"] = labels

    top_terms = top_terms_per_cluster(vec, matrix, labels, args.n_clusters)

    for c in range(args.n_clusters):
        cluster_df = df[df["cluster"] == c]
        print("=" * 70)
        print(f"CLUSTER {c}  (n={len(cluster_df)})")
        print(f"Top terms: {', '.join(top_terms[c])}")
        print("-" * 70)
        sample = cluster_df["customer_msg"].sample(
            min(args.samples_per_cluster, len(cluster_df)), random_state=args.seed
        )
        for s in sample:
            print(f"  - {s[:140]}")
        print()

    out_path = args.csv.replace(".csv", "_clustered.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote cluster assignments -> {out_path}")


if __name__ == "__main__":
    main()
