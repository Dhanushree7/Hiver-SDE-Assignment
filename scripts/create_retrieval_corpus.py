import pandas as pd


SOURCE = "data/processed/applesupport_pairs.csv"
OUTPUT = "data/processed/retrieval_corpus.csv"

MAX_PAIRS = 5000


df = pd.read_csv(SOURCE)

print(f"Total pairs available: {len(df)}")


# Keep only usable customer/reply pairs
df["customer_text"] = df["customer_text"].fillna("").astype(str)
df["response_text"] = df["response_text"].fillna("").astype(str)

df = df[
    (df["customer_text"].str.strip() != "") &
    (df["response_text"].str.strip() != "")
].copy()


# Remove exact duplicate customer/reply pairs
df = df.drop_duplicates(
    subset=["customer_text", "response_text"]
)


# Reproducible sample
if len(df) > MAX_PAIRS:
    df = df.sample(
        n=MAX_PAIRS,
        random_state=42
    )


# Sort chronologically when available
if "created_at" in df.columns:
    df = df.sort_values("created_at")


df.to_csv(
    OUTPUT,
    index=False
)


print("=" * 60)
print("RETRIEVAL CORPUS")
print("=" * 60)

print(f"Pairs selected: {len(df)}")
print(f"Saved: {OUTPUT}")

print("\nColumns:")
print(list(df.columns))

print("\nSample:")
print(
    df[
        ["customer_text", "response_text"]
    ].head(5).to_string(index=False)
)