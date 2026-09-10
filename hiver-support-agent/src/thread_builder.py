"""
Reconstructs (customer_message, brand_reply) pairs from the raw Kaggle
Customer Support on Twitter export format.

Real dataset columns: tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id

Vectorized (pandas merge, not iterrows) so this scales to the full ~3M-row
real dataset in under 30 seconds, not just the small demo file - the same
function is used by pipeline.py (small demo file, zero setup) and
scripts/build_real_dataset.py (full real file).

We only keep pairs where:
  - the customer tweet is inbound (inbound == True)
  - it has at least one brand reply (response_tweet_id resolves to a row
    authored by brand_handle)
This intentionally drops: customer tweets that got no reply (can't ground a
draft on a nonexistent resolution), and multi-hop threads beyond one
customer->brand turn (v1 scope - see decision log).
"""

import pandas as pd


def load_pairs(csv_path: str, brand_handle: str) -> pd.DataFrame:
    # Check if this is already a pre-built pairs file
    df_sample = pd.read_csv(csv_path, nrows=5)
    if "customer_msg" in df_sample.columns and "brand_reply" in df_sample.columns:
        # Pre-built format: customer_tweet_id, customer_msg, brand_reply, created_at
        df = pd.read_csv(csv_path, dtype=str)
        return df[["customer_tweet_id", "customer_msg", "brand_reply", "created_at"]].reset_index(drop=True)
    
    # Raw Kaggle format
    dtypes = {
        "tweet_id": str, "author_id": str, "text": str,
        "response_tweet_id": str, "in_response_to_tweet_id": str,
    }
    usecols = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id"]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, low_memory=False)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand_handle)]
    brand_reply_ids = set(brand_replies["tweet_id"])

    cust = df[df["inbound"] == True].copy()
    cust["first_reply_id"] = cust["response_tweet_id"].str.split(",").str[0].str.strip()
    cust_to_brand = cust[cust["first_reply_id"].isin(brand_reply_ids)]

    merged = cust_to_brand.merge(
        brand_replies[["tweet_id", "text"]].rename(columns={"tweet_id": "first_reply_id", "text": "brand_reply"}),
        on="first_reply_id", how="inner",
    )
    out = merged[["tweet_id", "text", "brand_reply", "created_at"]].rename(
        columns={"tweet_id": "customer_tweet_id", "text": "customer_msg"}
    )
    return out.reset_index(drop=True)


if __name__ == "__main__":
    pairs = load_pairs("/home/claude/hiver-support-agent/data/sample_tweets.csv", "AmazonHelp")
    print(f"Built {len(pairs)} (customer_msg, brand_reply) pairs")
    print(pairs.head(3).to_string())
