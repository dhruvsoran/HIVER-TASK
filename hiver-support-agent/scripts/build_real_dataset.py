"""
Extracts real (customer_msg, brand_reply) pairs for AmazonHelp from the full
twcs.csv (~3M rows), using vectorized pandas ops rather than iterrows() -
the original thread_builder.py loop is fine for the ~15-row demo file but
would take hours on 3M rows.

Splits into two disjoint pools so the golden eval set isn't drawn from the
same pairs used as the retrieval KB (avoids trivially "grounding" a reply in
the exact exemplar the eval question came from):
  - kb_pool: used to build the retrieval knowledge base
  - eval_pool: source pool for the golden set sample

Run:
    python build_real_dataset.py --csv /mnt/user-data/uploads/twcs.csv \
        --brand AmazonHelp --kb-size 20000 --eval-pool-size 5000
"""

import argparse
import pandas as pd


def load_real_pairs(csv_path: str, brand: str) -> pd.DataFrame:
    # ids as strings from the start - avoids the float "2.0" vs "2" bug entirely
    dtypes = {
        "tweet_id": str, "author_id": str, "text": str,
        "response_tweet_id": str, "in_response_to_tweet_id": str,
    }
    usecols = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id"]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, low_memory=False)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand)]
    print(f"Brand reply rows for {brand}: {len(brand_replies):,}")
    brand_reply_ids = set(brand_replies["tweet_id"])

    cust = df[df["inbound"] == True].copy()
    cust["first_reply_id"] = cust["response_tweet_id"].str.split(",").str[0].str.strip()

    # vectorized filter: keep only customer rows whose first reply is a brand reply
    cust_to_brand = cust[cust["first_reply_id"].isin(brand_reply_ids)]
    print(f"Customer rows replying to {brand}: {len(cust_to_brand):,}")

    merged = cust_to_brand.merge(
        brand_replies[["tweet_id", "text"]].rename(columns={"tweet_id": "first_reply_id", "text": "brand_reply"}),
        on="first_reply_id", how="inner",
    )
    out = merged[["tweet_id", "text", "brand_reply", "created_at"]].rename(
        columns={"tweet_id": "customer_tweet_id", "text": "customer_msg"}
    )
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--kb-size", type=int, default=20000)
    ap.add_argument("--eval-pool-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="../data")
    args = ap.parse_args()

    pairs = load_real_pairs(args.csv, args.brand)
    print(f"Total real pairs for {args.brand}: {len(pairs):,}")

    pairs = pairs.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    eval_pool = pairs.iloc[: args.eval_pool_size]
    kb_pool = pairs.iloc[args.eval_pool_size: args.eval_pool_size + args.kb_size]

    kb_path = f"{args.out_dir}/amazonhelp_kb_pool.csv"
    eval_path = f"{args.out_dir}/amazonhelp_eval_pool.csv"
    kb_pool.to_csv(kb_path, index=False)
    eval_pool.to_csv(eval_path, index=False)
    print(f"Wrote KB pool ({len(kb_pool):,} pairs) -> {kb_path}")
    print(f"Wrote eval pool ({len(eval_pool):,} pairs) -> {eval_path}")


if __name__ == "__main__":
    main()
