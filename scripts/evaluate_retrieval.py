import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


GOLDEN_PATH = "data/golden/intent_labeling.csv"
CORPUS_PATH = "data/processed/retrieval_corpus.csv"
EMBEDDINGS_PATH = "data/processed/retrieval_embeddings.npy"

OUTPUT_PATH = "data/processed/retrieval_evaluation.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)
golden = golden[golden["intent_label"].notna()].copy()

corpus = pd.read_csv(CORPUS_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

print(f"Golden examples: {len(golden)}")
print(f"Retrieval corpus: {len(corpus)}")


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Embed golden messages
# ---------------------------------------------------------

queries = golden["customer_text"].astype(str).tolist()

print("Embedding golden examples...")

query_embeddings = model.encode(
    queries,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ---------------------------------------------------------
# Retrieve top-5 historical cases
# ---------------------------------------------------------

results = []

for i, query_embedding in enumerate(query_embeddings):

    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:5]

    top_scores = scores[top_indices]

    top1_idx = top_indices[0]

    results.append({
        "golden_index": i,
        "customer_text": golden.iloc[i]["customer_text"],
        "intent_label": golden.iloc[i]["intent_label"],

        "top1_score": float(top_scores[0]),
        "top3_avg_score": float(np.mean(top_scores[:3])),
        "top5_avg_score": float(np.mean(top_scores)),

        "top1_customer_text":
            corpus.iloc[top1_idx]["customer_text"],

        "top1_response_text":
            corpus.iloc[top1_idx]["response_text"],

        "top1_customer_tweet_id":
            corpus.iloc[top1_idx]["customer_tweet_id"],
    })


results_df = pd.DataFrame(results)


# ---------------------------------------------------------
# Retrieval statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

print(f"\nExamples evaluated: {len(results_df)}")

print(
    f"Mean Top-1 similarity: "
    f"{results_df['top1_score'].mean():.4f}"
)

print(
    f"Median Top-1 similarity: "
    f"{results_df['top1_score'].median():.4f}"
)

print(
    f"Mean Top-3 similarity: "
    f"{results_df['top3_avg_score'].mean():.4f}"
)


# ---------------------------------------------------------
# Evidence-found rates at similarity thresholds
# ---------------------------------------------------------

for threshold in [0.50, 0.60, 0.65, 0.70, 0.75]:

    rate = (
        results_df["top1_score"] >= threshold
    ).mean()

    print(
        f"Top-1 >= {threshold:.2f}: "
        f"{rate:.1%}"
    )


# ---------------------------------------------------------
# Show weakest retrievals
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("10 WEAKEST RETRIEVALS")
print("=" * 70)

weakest = results_df.sort_values(
    "top1_score"
).head(10)

for _, row in weakest.iterrows():

    print("\n" + "-" * 70)

    print(
        f"Score: {row['top1_score']:.3f}"
    )

    print(
        f"Intent: {row['intent_label']}"
    )

    print(
        f"Query: {row['customer_text']}"
    )

    print(
        f"Retrieved: {row['top1_customer_text']}"
    )

    print(
        f"Historical response: "
        f"{row['top1_response_text']}"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved: {OUTPUT_PATH}"
)