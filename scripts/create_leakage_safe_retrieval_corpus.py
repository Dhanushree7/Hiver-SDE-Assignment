import pandas as pd
from pathlib import Path


PAIRS_FILE = Path(
    "data/processed/applesupport_pairs.csv"
)

GOLDEN_FILE = Path(
    "data/golden/intent_labeling.csv"
)

OUTPUT_FILE = Path(
    "data/processed/leakage_safe_retrieval_corpus.csv"
)

TARGET_SIZE = 5000


def main():

    print("Loading golden set...")

    golden = pd.read_csv(
        GOLDEN_FILE,
        usecols=[
            "customer_tweet_id",
            "intent_label"
        ]
    )

    golden = golden[
        golden["intent_label"].notna()
    ]

    golden_ids = set(
        golden["customer_tweet_id"]
        .astype(str)
    )

    print(
        f"Golden examples: {len(golden_ids)}"
    )

    print("\nLoading AppleSupport pairs...")

    pairs = pd.read_csv(
        PAIRS_FILE
    )

    print(
        f"Available pairs: {len(pairs)}"
    )

    # ---------------------------------------------------------
    # Normalize IDs
    # ---------------------------------------------------------

    pairs["customer_tweet_id"] = (
        pairs["customer_tweet_id"]
        .astype(str)
    )

    pairs["response_tweet_id"] = (
        pairs["response_tweet_id"]
        .astype(str)
    )

    # ---------------------------------------------------------
    # Remove ALL golden customer examples
    # ---------------------------------------------------------

    before = len(pairs)

    pairs = pairs[
        ~pairs["customer_tweet_id"].isin(golden_ids)
    ].copy()

    print(
        f"After golden exclusion: {len(pairs)}"
    )

    print(
        f"Excluded: {before - len(pairs)}"
    )

    # ---------------------------------------------------------
    # Remove exact duplicate customer/reply pairs
    # ---------------------------------------------------------

    pairs = pairs.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "response_tweet_id"
        ]
    )

    # ---------------------------------------------------------
    # Remove duplicate customer texts
    #
    # Keep one historical resolution per exact customer text.
    # ---------------------------------------------------------

    pairs = pairs.drop_duplicates(
        subset=["customer_text"]
    )

    print(
        f"After duplicate-text removal: {len(pairs)}"
    )

    # ---------------------------------------------------------
    # Reproducible sample
    # ---------------------------------------------------------

    sample_size = min(
        TARGET_SIZE,
        len(pairs)
    )

    corpus = pairs.sample(
        n=sample_size,
        random_state=42
    ).copy()

    corpus = corpus.reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "customer_tweet_id",
        "response_tweet_id",
        "created_at_customer",
        "created_at_response",
        "customer_text",
        "response_text",
    ]

    corpus[columns].to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    golden_overlap = (
        corpus["customer_tweet_id"]
        .isin(golden_ids)
        .sum()
    )

    duplicate_texts = (
        corpus["customer_text"]
        .duplicated()
        .sum()
    )

    print("\n" + "=" * 70)
    print("LEAKAGE-SAFE RETRIEVAL CORPUS")
    print("=" * 70)

    print(
        f"Rows: {len(corpus)}"
    )

    print(
        f"Golden customer IDs present: "
        f"{golden_overlap}"
    )

    print(
        f"Duplicate customer texts: "
        f"{duplicate_texts}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()