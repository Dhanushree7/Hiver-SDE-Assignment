import pandas as pd

human = pd.read_csv("data/golden/reply_human_labels.csv")
llm = pd.read_csv("data/processed/llm_judge_results_groq.csv")

metrics = [
    "correctness",
    "relevance",
    "groundedness",
    "completeness",
    "tone",
    "hallucination",
    "escalation_appropriate"
]

human["eval_id"] = human["eval_id"].astype(str)
llm["example_id"] = llm["example_id"].astype(str)

merged = human.merge(
    llm,
    left_on="eval_id",
    right_on="example_id",
    suffixes=("_human", "_llm")
)

print("Matched examples:", len(merged))
print("\nHuman vs LLM mean scores:")

for metric in metrics:
    h = merged[f"{metric}_human"]
    l = merged[f"{metric}_llm"]

    agreement = (h == l).mean() * 100

    print(
        f"{metric}: "
        f"Human={h.mean():.2f}, "
        f"LLM={l.mean():.2f}, "
        f"Exact agreement={agreement:.1f}%"
    )

print("\nMean absolute difference:")

for metric in metrics:
    difference = (
        merged[f"{metric}_human"] -
        merged[f"{metric}_llm"]
    ).abs().mean()

    print(f"{metric}: {difference:.2f}")

merged.to_csv(
    "data/processed/human_llm_agreement.csv",
    index=False
)

print("\nSaved: data/processed/human_llm_agreement.csv")
