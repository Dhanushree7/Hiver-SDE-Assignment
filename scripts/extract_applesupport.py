import pandas as pd
from pathlib import Path

INPUT = "data/raw/twcs.csv"
OUTPUT = "data/processed/applesupport_tweets.csv"

print("Loading AppleSupport tweets...")

df = pd.read_csv(
    INPUT,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    dtype={
        "tweet_id": "int64",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce",
)

# Keep AppleSupport's own replies plus the customer tweets
# that those replies directly respond to.
brand_replies = df[
    (df["author_id"] == "AppleSupport") &
    (df["inbound"] == False)
].copy()

parent_ids = pd.to_numeric(
    brand_replies["in_response_to_tweet_id"],
    errors="coerce"
).dropna().astype("int64")

customer_messages = df[
    (df["tweet_id"].isin(parent_ids)) &
    (df["inbound"] == True)
].copy()

# Combine the two sides and remove accidental duplicates.
apple = pd.concat(
    [customer_messages, brand_replies],
    ignore_index=True
).drop_duplicates(subset=["tweet_id"])

apple = apple.sort_values("created_at")

Path("data/processed").mkdir(parents=True, exist_ok=True)

apple.to_csv(OUTPUT, index=False)

print()
print("AppleSupport extraction complete.")
print(f"Brand replies:       {len(brand_replies):,}")
print(f"Customer messages:   {len(customer_messages):,}")
print(f"Combined tweets:     {len(apple):,}")
print(f"Saved to:            {OUTPUT}")
print()
print("Date range:")
print(f"  {apple['created_at'].min()}")
print(f"  {apple['created_at'].max()}")
