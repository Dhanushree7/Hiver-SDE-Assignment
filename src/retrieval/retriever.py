import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

CORPUS_FILE = Path(
    "data/processed/leakage_safe_retrieval_corpus.csv"
)

EMBEDDINGS_FILE = Path(
    "data/processed/leakage_safe_retrieval_embeddings.npy"
)


class Retriever:

    def __init__(self):

        print("Loading retrieval corpus...")

        self.corpus = pd.read_csv(
            CORPUS_FILE
        )

        print(
            f"Retrieval corpus size: "
            f"{len(self.corpus)}"
        )

        print("Loading retrieval embeddings...")

        self.embeddings = np.load(
            EMBEDDINGS_FILE
        )

        if len(self.corpus) != len(
            self.embeddings
        ):
            raise ValueError(
                "Corpus and embedding count do not match."
            )

        print(
            f"Embedding matrix: "
            f"{self.embeddings.shape}"
        )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def search(
        self,
        query,
        top_k=5
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Since both query and corpus embeddings are
        # normalized, dot product = cosine similarity.
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            row = self.corpus.iloc[index]

            results.append({
                "customer_tweet_id": str(
                    row["customer_tweet_id"]
                ),
                "response_tweet_id": str(
                    row["response_tweet_id"]
                ),
                "created_at_customer":
                    row["created_at_customer"],
                "created_at_response":
                    row["created_at_response"],
                "customer_text":
                    row["customer_text"],
                "response_text":
                    row["response_text"],
                "score": float(
                    scores[index]
                ),
            })

        return results


if __name__ == "__main__":

    retriever = Retriever()

    query = (
        "My iPhone battery is draining "
        "very quickly after an iOS update."
    )

    results = retriever.search(
        query,
        top_k=3
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[{i}] "
            f"similarity={result['score']:.3f}"
        )

        print(
            "Customer:",
            result["customer_text"]
        )

        print(
            "Response:",
            result["response_text"]
        )