import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

INPUT = "data/processed/applesupport_clean_text.csv"

print("Loading data...")
df = pd.read_csv(INPUT)

# Use a representative sample for intent discovery
sample = df.sample(
    n=min(30000, len(df)),
    random_state=42
).reset_index(drop=True)

texts = sample["clean_customer_text"].fillna("").astype(str).tolist()

print(f"Messages selected: {len(texts):,}")
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating semantic embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Embedding shape: {embeddings.shape}")

# Try 10 semantic clusters
print()
print("Clustering...")
km = MiniBatchKMeans(
    n_clusters=10,
    random_state=42,
    n_init=10,
    batch_size=1024
)

labels = km.fit_predict(embeddings)
sample["cluster"] = labels

print()
print("=" * 70)
print("SEMANTIC INTENT CLUSTERS")
print("=" * 70)

for cluster_id in range(10):

    cluster_df = sample[sample["cluster"] == cluster_id]

    print()
    print(f"CLUSTER {cluster_id} ({len(cluster_df):,} messages)")
    print("-" * 70)

    examples = cluster_df.sample(
        n=min(5, len(cluster_df)),
        random_state=cluster_id
    )

    for _, row in examples.iterrows():
        print(f"- {row['customer_text']}")

sample.to_csv(
    "data/processed/semantic_intent_samples.csv",
    index=False
)

np.save(
    "data/processed/semantic_intent_embeddings.npy",
    embeddings
)

print()
print("=" * 70)
print("Saved:")
print("  data/processed/semantic_intent_samples.csv")
print("  data/processed/semantic_intent_embeddings.npy")
print("=" * 70)
