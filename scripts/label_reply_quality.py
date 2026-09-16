import pandas as pd

INPUT = "data/golden/reply_judge_set.csv"
OUTPUT = "data/golden/reply_human_labels.csv"

df = pd.read_csv(INPUT, dtype=str)

# Create output columns if they don't exist
columns = {
    "correctness": "",
    "relevance": "",
    "groundedness": "",
    "completeness": "",
    "tone": "",
    "hallucination": "",
    "escalation_appropriate": "",
}

for col, default in columns.items():
    if col not in df.columns:
        df[col] = default

print("=" * 80)
print("BLIND HUMAN REPLY QUALITY EVALUATION")
print("=" * 80)
print()
print("Rate the reply using ONLY the customer message and agent reply.")
print()
print("1-5 scale:")
print("  1 = Very poor")
print("  2 = Poor")
print("  3 = Acceptable")
print("  4 = Good")
print("  5 = Excellent")
print()
print("Binary:")
print("  hallucination: 1 = unsupported/hallucinated claim, 0 = no")
print("  escalation_appropriate: 1 = appropriate, 0 = inappropriate")
print("=" * 80)

for i, row in df.iterrows():

    # Skip already completed rows
    if all(str(row[col]).strip() in ["0", "1", "2", "3", "4", "5"]
           for col in columns):
        continue

    print(f"\nExample {i + 1}/{len(df)}")
    print("-" * 80)
    print("CUSTOMER:")
    print(row["customer_text"])
    print()
    print("AGENT REPLY:")
    print(row["reply"])
    print("-" * 80)

    prompts = [
        ("correctness", "Correctness [1-5]: ", ["1","2","3","4","5"]),
        ("relevance", "Relevance [1-5]: ", ["1","2","3","4","5"]),
        ("groundedness", "Groundedness [1-5]: ", ["1","2","3","4","5"]),
        ("completeness", "Completeness [1-5]: ", ["1","2","3","4","5"]),
        ("tone", "Tone [1-5]: ", ["1","2","3","4","5"]),
        ("hallucination", "Hallucination [1=yes, 0=no]: ", ["0","1"]),
        ("escalation_appropriate", "Escalation appropriate [1=yes, 0=no]: ", ["0","1"]),
    ]

    for col, prompt, valid in prompts:
        while True:
            value = input(prompt).strip()
            if value in valid:
                df.at[i, col] = value
                df.to_csv(OUTPUT, index=False)
                break
            print("Invalid input. Please enter:", ", ".join(valid))

print("\n" + "=" * 80)
print("HUMAN REPLY EVALUATION COMPLETED")
print("=" * 80)
print("Saved:", OUTPUT)