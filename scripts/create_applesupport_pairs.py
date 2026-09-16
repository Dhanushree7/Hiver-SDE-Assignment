import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/twcs.csv")
OUTPUT = Path("data/processed/applesupport_pairs.csv")

BRAND = "AppleSupport"

print("Loading TWCS dataset...")
df = pd.read_csv(INPUT)

print(f"Total rows: {len(df):,}")

# Normalize tweet IDs
df["tweet_id_num"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df["parent_id_num"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

# Create lookup BEFORE changing the index
tweet_lookup = df[
    [
        "tweet_id_num",
        "author_id",
        "inbound",
        "created_at",
        "text"
    ]
].dropna(subset=["tweet_id_num"])

tweet_lookup = tweet_lookup.set_index("tweet_id_num")

# AppleSupport responses
apple = df[df["author_id"] == BRAND].copy()

print(f"AppleSupport replies: {len(apple):,}")

pairs = []

for _, reply in apple.iterrows():

    parent_id = reply["parent_id_num"]

    if pd.isna(parent_id):
        continue

    if parent_id not in tweet_lookup.index:
        continue

    customer = tweet_lookup.loc[parent_id]

    # Only keep inbound/customer messages
    if customer["inbound"] is not True and customer["inbound"] != True:
        continue

    pairs.append({
        "customer_tweet_id": int(parent_id),
        "response_tweet_id": int(reply["tweet_id_num"]),
        "created_at_customer": customer["created_at"],
        "created_at_response": reply["created_at"],
        "customer_text": customer["text"],
        "response_text": reply["text"],
    })

pairs_df = pd.DataFrame(pairs)

if pairs_df.empty:
    print("ERROR: No customer-response pairs found.")
    raise SystemExit(1)

# Remove duplicate pairs
pairs_df = pairs_df.drop_duplicates(
    subset=["customer_tweet_id", "response_tweet_id"]
)

# Parse dates
pairs_df["created_at_customer"] = pd.to_datetime(
    pairs_df["created_at_customer"],
    utc=True,
    format="mixed"
)

pairs_df["created_at_response"] = pd.to_datetime(
    pairs_df["created_at_response"],
    utc=True,
    format="mixed"
)

# Sort chronologically
pairs_df = pairs_df.sort_values("created_at_customer")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pairs_df.to_csv(OUTPUT, index=False)

print()
print("=" * 70)
print("APPLE SUPPORT CUSTOMER -> RESPONSE PAIRS")
print("=" * 70)

print(f"Saved file:             {OUTPUT}")
print(f"Pairs:                  {len(pairs_df):,}")
print(f"Unique customer tweets: {pairs_df['customer_tweet_id'].nunique():,}")
print(f"Unique Apple replies:   {pairs_df['response_tweet_id'].nunique():,}")

print()
print("Date range:")
print(f"  Start: {pairs_df['created_at_customer'].min()}")
print(f"  End:   {pairs_df['created_at_customer'].max()}")

print()
print("=" * 70)
print("SAMPLE PAIRS")
print("=" * 70)

for i, (_, row) in enumerate(pairs_df.head(10).iterrows(), 1):
    print()
    print(f"--- Pair {i} ---")
    print(f"[CUSTOMER] {row['customer_text']}")
    print(f"[APPLE]    {row['response_text']}")

print()
print("Done.")
