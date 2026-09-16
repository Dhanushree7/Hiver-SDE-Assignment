import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/applesupport_clean_text.csv")
OUTPUT = Path("data/golden/intent_labeling.csv")

df = pd.read_csv(INPUT)

# Stratified-ish sampling using keyword candidate signals.
# This gives us a diverse starting pool rather than 250 random tweets.
patterns = {
    "ios_update_issues": r"\bios\b|\bupdate\b|\bupdated\b|\bupdat(e|ing)\b|\bupgrade\b",
    "apps_services": r"\bapp store\b|\bitunes\b|\bapple music\b|\bsiri\b|\bmail\b|\bapp\b",
    "battery_power": r"\bbattery\b|\bcharging\b|\bcharger\b|\bcharge\b|\boverheat|\bhot\b",
    "device_performance": r"\bfreez|\blag\b|\bslow\b|\bcrash|\bhanging\b|\bunresponsive\b",
    "hardware_accessories": r"\bscreen\b|\btouchscreen\b|\bcamera\b|\bearpods?\b|\bcable\b|\bheadphone",
    "keyboard_input": r"\bkeyboard\b|\bautocorrect\b|\btyping\b|\btype\b|\bquestion mark\b",
    "connectivity": r"\bwi[- ]?fi\b|\bbluetooth\b|\bsignal\b|\bsim\b|\bcellular\b|\bnetwork\b",
    "photos_data": r"\bphoto|\bpictures?\b|\bcontacts?\b|\bbackup\b|\bdata\b",
    "apple_id_account": r"\bapple id\b|\bicloud\b|\bpassword\b|\blocked\b|\bverification\b|\baccount\b",
    "purchases_orders_refunds": r"\border\b|\brefund\b|\bpurchase\b|\bbought\b|\bsubscription\b|\bcheckout\b|\bpreorder\b"
}

# Collect up to 30 examples per candidate intent.
samples = []

for intent, pattern in patterns.items():

    matches = df[
        df["clean_customer_text"]
        .fillna("")
        .str.contains(pattern, regex=True, case=False, na=False)
    ]

    n = min(30, len(matches))

    if n > 0:
        samples.append(
            matches.sample(n=n, random_state=42)
        )

# Add 50 completely random examples for discovering messages
# that don't fit the candidate taxonomy.
random_sample = df.sample(
    n=min(50, len(df)),
    random_state=123
)

samples.append(random_sample)

label_df = pd.concat(samples, ignore_index=True)

# Remove duplicate customer tweets
label_df = label_df.drop_duplicates(
    subset=["customer_tweet_id"]
)

# Shuffle
label_df = label_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Keep a manageable initial manual labeling set.
label_df = label_df.head(300).copy()

# Empty column for YOUR manual label.
label_df["intent_label"] = ""

# Optional notes for difficult cases.
label_df["label_notes"] = ""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

label_df[
    [
        "customer_tweet_id",
        "customer_text",
        "response_text",
        "intent_label",
        "label_notes"
    ]
].to_csv(OUTPUT, index=False)

print("=" * 70)
print("MANUAL INTENT LABELING SET")
print("=" * 70)

print(f"Total examples: {len(label_df):,}")
print(f"Saved to:       {OUTPUT}")

print()
print("Candidate labels:")
for intent in patterns:
    print(f"  - {intent}")

print("  - other_unclear")

print()
print("IMPORTANT:")
print("Open the CSV and fill ONLY the intent_label column.")
print("Use other_unclear when none of the candidate intents fits.")
print("Use label_notes for ambiguous examples.")

print()
print("Done.")
