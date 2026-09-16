import pandas as pd

INPUT = "data/golden/escalation_labeling_annotator2.csv"

df = pd.read_csv(INPUT, dtype=str)

print("=" * 80)
print("BLIND SECOND ESCALATION LABELING")
print("=" * 80)
print()
print("For each example:")
print("  1 = ESCALATE")
print("  0 = AUTO-HANDLE")
print()
print("Make your decision using ONLY the customer message.")
print("=" * 80)

for i, row in df.iterrows():

    if str(row.get("human_escalate_2", "")).strip() in ["0", "1"]:
        continue

    print(f"\nExample {i + 1}/{len(df)}")
    print("-" * 80)
    print(row["customer_text"])
    print("-" * 80)

    while True:
        label = input("Your decision [1=ESCALATE, 0=AUTO-HANDLE]: ").strip()

        if label in ["0", "1"]:
            df.at[i, "human_escalate_2"] = label
            df.to_csv(INPUT, index=False)
            break

        print("Please enter only 1 or 0.")

print("\n" + "=" * 80)
print("BLIND SECOND PASS COMPLETED")
print("=" * 80)

print(
    df["human_escalate_2"]
    .value_counts(dropna=False)
)