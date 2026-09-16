import pandas as pd
import re
from pathlib import Path

INPUT = Path("data/processed/applesupport_pairs.csv")
OUTPUT = Path("data/processed/applesupport_clean_text.csv")

df = pd.read_csv(INPUT)

def clean_text(text):
    text = str(text)

    # HTML entities
    text = text.replace("&amp;", "and")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')

    # URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Twitter handles
    text = re.sub(r"@\w+", " ", text)

    # Numeric Twitter/user IDs that occur in the dataset
    text = re.sub(r"\b\d{4,}\b", " ", text)

    # Normalize apostrophes
    text = text.replace("’", "'")

    # Keep letters/numbers and useful punctuation
    text = re.sub(r"[^a-zA-Z0-9\s'\-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

df["clean_customer_text"] = df["customer_text"].fillna("").apply(clean_text)

# Remove completely empty messages
df = df[df["clean_customer_text"].str.len() > 2].copy()

df.to_csv(OUTPUT, index=False)

print("=" * 70)
print("APPLE SUPPORT TEXT CLEANING")
print("=" * 70)
print(f"Original pairs:       106,646")
print(f"Remaining messages:   {len(df):,}")
print(f"Saved:                {OUTPUT}")

print()
print("=" * 70)
print("SAMPLE CLEANED TEXT")
print("=" * 70)

for i, (_, row) in enumerate(
    df[["customer_text", "clean_customer_text"]].sample(
        15, random_state=42
    ).iterrows(), 1
):
    print()
    print(f"--- Example {i} ---")
    print("ORIGINAL:", row["customer_text"])
    print("CLEAN:   ", row["clean_customer_text"])

print()
print("Done.")
