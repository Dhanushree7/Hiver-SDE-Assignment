import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import Retriever


GOLDEN_FILE = Path(
    "data/golden/intent_labeling.csv"
)

OUTPUT_FILE = Path(
    "data/processed/leakage_safe_retrieval_evaluation.csv"
)


def main():

    golden = pd.read_csv(
        GOLDEN_FILE
    )

    golden = golden[
        golden["intent_label"].notna()
    ].copy()

    print(
        f"Golden examples: {len(golden)}"
    )

    retriever = Retriever()

    rows = []

    print("\nEvaluating retrieval...")

    for i, row in golden.iterrows():

        query = str(
            row["customer_text"]
        )

        results = retriever.search(
            query,
            top_k=3
        )

        top1 = results[0]["score"]
        top3_mean = np.mean(
            [r["score"] for r in results]
        )

        rows.append({
            "customer_tweet_id":
                row["customer_tweet_id"],
            "customer_text":
                query,
            "gold_intent":
                row["intent_label"],
            "top1_similarity":
                top1,
            "top3_mean_similarity":
                top3_mean,
        })

        if len(rows) % 25 == 0:
            print(
                f"Processed {len(rows)}/"
                f"{len(golden)}"
            )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -------------------------------------------------
    # Summary statistics
    # -------------------------------------------------

    top1 = results_df[
        "top1_similarity"
    ]

    top3 = results_df[
        "top3_mean_similarity"
    ]

    print("\n" + "=" * 70)
    print("LEAKAGE-SAFE RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        f"Examples: {len(results_df)}"
    )

    print(
        f"Mean Top-1 similarity: "
        f"{top1.mean():.4f}"
    )

    print(
        f"Median Top-1 similarity: "
        f"{top1.median():.4f}"
    )

    print(
        f"Mean Top-3 similarity: "
        f"{top3.mean():.4f}"
    )

    print("\nTop-1 similarity thresholds:")

    for threshold in [
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        percentage = (
            (top1 >= threshold).mean()
            * 100
        )

        print(
            f"  >= {threshold:.2f}: "
            f"{percentage:.1f}%"
        )

    # -------------------------------------------------
    # Weakest examples
    # -------------------------------------------------

    print(
        "\nWeakest 10 retrievals:"
    )

    weakest = results_df.nsmallest(
        10,
        "top1_similarity"
    )

    for _, row in weakest.iterrows():

        print(
            f"\n{row['top1_similarity']:.3f}"
            f" | {row['gold_intent']}"
        )

        print(
            row["customer_text"]
        )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()