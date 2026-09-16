import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


JUDGE_SET = "data/golden/reply_judge_set.csv"
CORPUS = "data/processed/leakage_safe_retrieval_corpus.csv"
EMBEDDINGS = "data/processed/leakage_safe_retrieval_embeddings.npy"
OUTPUT = "data/golden/reply_judge_set_with_evidence.csv"

TOP_K = 3
MODEL_NAME = "all-MiniLM-L6-v2"


def main():

    judge = pd.read_csv(JUDGE_SET, dtype=str)
    corpus = pd.read_csv(CORPUS, dtype=str)
    embeddings = np.load(EMBEDDINGS)

    print("Judge examples:", len(judge))
    print("Retrieval corpus:", len(corpus))
    print("Embeddings:", embeddings.shape)

    if len(corpus) != len(embeddings):
        raise RuntimeError(
            "Corpus and embedding count do not match."
        )

    model = SentenceTransformer(MODEL_NAME)

    queries = judge["customer_text"].fillna("").tolist()

    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    corpus_embeddings = embeddings.copy()

    # Ensure normalized embeddings for cosine similarity.
    norms = np.linalg.norm(
        corpus_embeddings,
        axis=1,
        keepdims=True,
    )

    corpus_embeddings = (
        corpus_embeddings / np.maximum(norms, 1e-12)
    )

    evidence_strings = []

    for i, query_vector in enumerate(query_embeddings):

        scores = corpus_embeddings @ query_vector

        # Prevent the judge example from retrieving itself,
        # although the retrieval corpus was already constructed
        # to exclude the golden examples.
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        blocks = []

        for rank, idx in enumerate(top_indices, start=1):

            row = corpus.iloc[idx]

            blocks.append(
                f"""
HISTORICAL CASE {rank}

Customer:
{row["customer_text"]}

AppleSupport response:
{row["response_text"]}

Similarity:
{scores[idx]:.4f}
""".strip()
            )

        evidence_strings.append(
            "\n\n".join(blocks)
        )

    judge["retrieved_evidence"] = evidence_strings

    Path(OUTPUT).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    judge.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 80)
    print("JUDGE DATASET PREPARED")
    print("=" * 80)
    print("Examples:", len(judge))
    print("Top-K evidence:", TOP_K)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()