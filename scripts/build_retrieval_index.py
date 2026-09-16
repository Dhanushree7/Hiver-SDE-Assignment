import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CORPUS_PATH = "data/processed/retrieval_corpus.csv"
EMBEDDINGS_PATH = "data/processed/retrieval_embeddings.npy"


df = pd.read_csv(CORPUS_PATH)

texts = df["customer_text"].fillna("").astype(str).tolist()

print(f"Retrieval examples: {len(texts)}")

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating retrieval embeddings...")

embeddings = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings)

print(f"\nEmbedding shape: {embeddings.shape}")

np.save(
    EMBEDDINGS_PATH,
    embeddings
)

print("=" * 60)
print("RETRIEVAL INDEX CREATED")
print("=" * 60)

print(f"Embeddings: {EMBEDDINGS_PATH}")
print(f"Shape: {embeddings.shape}")