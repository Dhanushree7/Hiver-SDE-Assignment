import pandas as pd
from pathlib import Path

CSV_PATH = "data/raw/twcs.csv"

BRANDS = [
    "AppleSupport",
    "AmazonHelp",
    "SpotifyCares",
    "Uber_Support",
    "AirAsiaSupport",
    "comcastcares",
]

print("Loading dataset...")
print("This may take a little while because the dataset is ~516 MB.")

df = pd.read_csv(
    CSV_PATH,
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

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

print(f"Loaded {len(df):,} tweets.\n")

results = []

for brand in BRANDS:
    print(f"Analyzing {brand}...")

    brand_mask = (df["author_id"] == brand) & (df["inbound"] == False)
    brand_replies = df[brand_mask]

    # Customer tweets that received a direct reply from this brand.
    replies_with_parent = brand_replies[
        brand_replies["in_response_to_tweet_id"].notna()
    ]

    customer_ids = pd.to_numeric(
        replies_with_parent["in_response_to_tweet_id"],
        errors="coerce"
    ).dropna().astype("int64")

    customer_tweets = df[df["tweet_id"].isin(customer_ids)]

    # Only count cases where the parent tweet was actually inbound.
    customer_tweets = customer_tweets[customer_tweets["inbound"] == True]

    # Exact duplicate rate among brand replies.
    if len(brand_replies) > 0:
        duplicate_rate = (
            1 - brand_replies["text"].nunique(dropna=True) / len(brand_replies)
        )
    else:
        duplicate_rate = 0

    # Very simple proxy for account-specific issues.
    account_words = (
        r"order|refund|charge|account|password|login|cancel|subscription|"
        r"payment|invoice|card|booking|reservation|ticket|billing"
    )

    self_service_words = (
        r"how do i|how to|not working|won't|wont|error|update|install|"
        r"setup|reset|slow|crash|bug|problem|issue"
    )

    customer_text = customer_tweets["text"].fillna("")

    account_mask = customer_text.str.contains(
        account_words, case=False, regex=True
    )

    self_service_mask = customer_text.str.contains(
        self_service_words, case=False, regex=True
    )

    account_only = account_mask & ~self_service_mask
    self_service_only = self_service_mask & ~account_mask
    mixed = account_mask & self_service_mask
    unclear = ~account_mask & ~self_service_mask

    total_customers = len(customer_tweets)

    def pct(series):
        if total_customers == 0:
            return 0.0
        return round(series.sum() / total_customers * 100, 1)

    results.append({
        "brand": brand,
        "brand_replies": len(brand_replies),
        "customer_brand_pairs": len(customer_tweets),
        "pair_rate_pct": round(
            len(customer_tweets) / max(len(brand_replies), 1) * 100, 1
        ),
        "unique_customer_messages": customer_tweets["tweet_id"].nunique(),
        "duplicate_reply_pct": round(duplicate_rate * 100, 1),
        "pct_account_specific": pct(account_only),
        "pct_self_service": pct(self_service_only),
        "pct_mixed": pct(mixed),
        "pct_unclear": pct(unclear),
        "date_start": brand_replies["created_at"].min(),
        "date_end": brand_replies["created_at"].max(),
    })

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "customer_brand_pairs",
    ascending=False
)

Path("data/processed").mkdir(parents=True, exist_ok=True)

result_df.to_csv(
    "data/processed/brand_comparison.csv",
    index=False
)

print("\n" + "=" * 100)
print("BRAND COMPARISON")
print("=" * 100)
print(result_df.to_string(index=False))
print("=" * 100)

print(
    "\nSaved to: data/processed/brand_comparison.csv"
)
