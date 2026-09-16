import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

df = pd.read_csv("data/processed/human_llm_agreement.csv")

metrics = [
    "correctness",
    "relevance",
    "groundedness",
    "completeness",
    "tone",
    "hallucination",
    "escalation_appropriate"
]

print("Human–LLM Agreement Analysis")
print("=" * 40)

for metric in metrics:
    human = df[f"{metric}_human"]
    llm = df[f"{metric}_llm"]

    correlation, _ = spearmanr(human, llm)

    print(f"\n{metric}")
    print(f"Spearman correlation: {correlation:.3f}")

    if metric in ["hallucination", "escalation_appropriate"]:
        kappa = cohen_kappa_score(human, llm)
        print(f"Cohen's kappa: {kappa:.3f}")

print("\nAnalysis completed.")
