import pandas as pd
from pathlib import Path

FILE = Path("data/golden/intent_labeling.csv")

LABELS = [
    "ios_update_issues",
    "apps_services",
    "battery_power",
    "device_performance",
    "hardware_accessories",
    "keyboard_input",
    "connectivity",
    "photos_data",
    "apple_id_account",
    "purchases_orders_refunds",
    "other_unclear"
]

df = pd.read_csv(FILE, dtype=str).fillna("")

# Make sure the label column is text, not float.
df["intent_label"] = df["intent_label"].astype("string")

# Find first unlabeled row.
unlabeled = df.index[
    df["intent_label"].str.strip() == ""
]

if len(unlabeled) == 0:
    print("All examples are already labeled.")
    raise SystemExit

start = unlabeled[0]

print("=" * 80)
print("APPLE SUPPORT INTENT LABELER")
print("=" * 80)
print(f"Resuming at example {start + 1}")
print("Type the number of the best matching intent.")
print("Type Q at any time to save and quit.")
print()

for i in range(start, min(200, len(df))):

    print("-" * 80)
    print(f"EXAMPLE {i + 1} / 200")
    print("-" * 80)
    print()
    print("CUSTOMER:")
    print(df.loc[i, "customer_text"])
    print()
    print("INTENT OPTIONS:")

    for n, label in enumerate(LABELS, 1):
        print(f"  {n:2}. {label}")

    print()

    while True:
        choice = input("Your choice (1-11, or Q): ").strip().lower()

        if choice == "q":
            df.to_csv(FILE, index=False, encoding="utf-8-sig")
            labeled = (
                df["intent_label"].str.strip().ne("").sum()
            )
            print()
            print(f"Progress saved.")
            print(f"Total labeled: {labeled}")
            raise SystemExit

        if choice.isdigit() and 1 <= int(choice) <= 11:
            label = LABELS[int(choice) - 1]
            df.loc[i, "intent_label"] = label
            df.to_csv(FILE, index=False, encoding="utf-8-sig")
            print(f"Saved: {label}")
            break

        print("Please enter a number from 1 to 11, or Q.")

print()
print("=" * 80)
print("200 EXAMPLES COMPLETED")
print("=" * 80)
