import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path


CORPUS_FILE = Path(
    "data/processed/leakage_safe_retrieval_corpus.csv"
)

OUTPUT_FILE = Path(
    "data/processed/leakage_safe_retrieval_embeddings.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():

    print("Loading retrieval corpus...")

    df = pd.read_csv(CORPUS_FILE)

    print(f"Corpus rows: {len(df)}")

    texts = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        f"\nEmbedding shape: "
        f"{embeddings.shape}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()