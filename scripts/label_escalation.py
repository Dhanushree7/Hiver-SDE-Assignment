import pandas as pd

INPUT_PATH = "data/golden/escalation_labeling.csv"
OUTPUT_PATH = "data/golden/escalation_labeling.csv"

df = pd.read_csv(INPUT_PATH, dtype=str)

# Create label column if it doesn't exist
if "human_escalate" not in df.columns:
    df["human_escalate"] = ""

# Resume safely if interrupted
for i in range(len(df)):

    # Skip already labelled examples
    if pd.notna(df.loc[i, "human_escalate"]) and str(
        df.loc[i, "human_escalate"]
    ).strip() in {"1", "2"}:
        continue

    print("\n" + "=" * 80)
    print(f"EXAMPLE {i + 1}/{len(df)}")
    print("=" * 80)

    print("\nCustomer message:")
    print(df.loc[i, "customer_text"])

    print("\nShould this be escalated to a human?")

    print("1 = ESCALATE")
    print("2 = AUTO-HANDLE")

    while True:

        choice = input("\nYour choice (1/2, or Q): ").strip().upper()

        if choice == "Q":
            df.to_csv(OUTPUT_PATH, index=False)
            print("\nProgress saved.")
            print("You can run the script again to continue.")
            raise SystemExit

        if choice in {"1", "2"}:
            break

        print("Please enter 1, 2, or Q.")

    df.loc[i, "human_escalate"] = choice

    # Save after every example
    df.to_csv(OUTPUT_PATH, index=False)

    label = "ESCALATE" if choice == "1" else "AUTO-HANDLE"

    print(f"Saved: {label}")


print("\n" + "=" * 80)
print("50 ESCALATION EXAMPLES COMPLETED")
print("=" * 80)

print("\nLabel distribution:")
print(df["human_escalate"].value_counts(dropna=False))