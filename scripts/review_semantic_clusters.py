import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT = "data/processed/semantic_intent_samples.csv"

df = pd.read_csv(INPUT)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("=" * 70)
print("SEMANTIC CLUSTER REVIEW")
print("=" * 70)

for cluster_id in sorted(df["cluster"].unique()):

    cluster_df = df[df["cluster"] == cluster_id].copy()

    print()
    print("=" * 70)
    print(f"CLUSTER {cluster_id} | {len(cluster_df):,} messages")
    print("=" * 70)

    texts = (
        cluster_df["clean_customer_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Mean embedding = cluster centroid
    centroid = embeddings.mean(axis=0)

    # Normalize centroid
    centroid = centroid / np.linalg.norm(centroid)

    # Cosine similarity because embeddings are normalized
    scores = embeddings @ centroid

    cluster_df["representative_score"] = scores

    representatives = cluster_df.sort_values(
        "representative_score",
        ascending=False
    ).head(8)

    for i, (_, row) in enumerate(representatives.iterrows(), 1):
        print(f"{i}. {row['customer_text']}")

print()
print("=" * 70)
print("Review complete.")
print("=" * 70)
