import pandas as pd
import re


SOURCE = "data/processed/applesupport_clean_text.csv"
GOLDEN = "data/golden/intent_labeling.csv"
OUTPUT = "data/processed/intent_train.csv"


df = pd.read_csv(SOURCE)
gold = pd.read_csv(GOLDEN)

# Remove anything used in the golden set
gold_texts = set(
    gold.loc[gold["intent_label"].notna(), "customer_text"]
    .astype(str)
)

df = df[
    ~df["customer_text"].astype(str).isin(gold_texts)
].copy()


RULES = {
    "ios_update_issues": [
        r"\bios\b.*\b(update|upgrade|downgrade)\b",
        r"\b(update|upgraded|updating)\b.*\bios\b",
        r"\bios\s*\d+",
        r"\bhigh sierra\b",
        r"\bsoftware update\b",
    ],

    "battery_power": [
        r"\bbattery\b",
        r"\bcharging\b",
        r"\bcharge\b",
        r"\bdrain(s|ed|ing)?\b",
        r"\bbattery life\b",
    ],

    "connectivity": [
        r"\bwi[- ]?fi\b",
        r"\bbluetooth\b",
        r"\bcellular\b",
        r"\bsim\b",
        r"\bnetwork\b",
        r"\bsignal\b",
    ],

    "keyboard_input": [
        r"\bkeyboard\b",
        r"\bautocorrect\b",
        r"\btyping\b",
        r"\btype\b",
        r"\bquestion mark\b",
        r"\btyping\b",
    ],

    "apple_id_account": [
        r"\bapple id\b",
        r"\bicloud account\b",
        r"\bpassword\b",
        r"\baccount locked\b",
        r"\blogin\b",
        r"\bsign in\b",
        r"\bverification\b",
    ],

    "purchases_orders_refunds": [
        r"\brefund\b",
        r"\bpurchase\b",
        r"\bpurchased\b",
        r"\border\b",
        r"\bpre[- ]?order\b",
        r"\bsubscription\b",
        r"\bbilling\b",
        r"\bpayment\b",
        r"\bcharged\b",
    ],

    "photos_data": [
        r"\bphotos?\b",
        r"\bpictures?\b",
        r"\bcontacts?\b",
        r"\bbackup\b",
        r"\bdata loss\b",
        r"\bmissing data\b",
    ],

    "hardware_accessories": [
        r"\bheadphones?\b",
        r"\bairpods?\b",
        r"\bcamera\b",
        r"\bscreen\b",
        r"\bdisplay\b",
        r"\bcharger\b",
        r"\bcable\b",
        r"\bspeaker\b",
        r"\btouch bar\b",
        r"\b3d touch\b",
    ],

    "apps_services": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bsiri\b",
        r"\bapp store\b",
        r"\bpodcasts?\b",
        r"\bmessages?\b",
        r"\bmail\b",
        r"\bvoicemail\b",
    ],

    "device_performance": [
        r"\bfreez(e|es|ing)\b",
        r"\blag(ging)?\b",
        r"\bcrash(es|ed|ing)?\b",
        r"\breboot(s|ed|ing)?\b",
        r"\brestart(s|ed|ing)?\b",
        r"\bslow\b",
        r"\bunresponsive\b",
    ],
}


def classify(text):
    text = str(text).lower()

    matches = []

    for intent, patterns in RULES.items():
        score = sum(
            1 for pattern in patterns
            if re.search(pattern, text)
        )

        if score > 0:
            matches.append((intent, score))

    if not matches:
        return None, 0

    matches.sort(key=lambda x: x[1], reverse=True)

    # Only keep high-confidence single-intent examples.
    if len(matches) == 1:
        return matches[0][0], matches[0][1]

    # If one intent clearly dominates, accept it.
    if matches[0][1] >= matches[1][1] + 2:
        return matches[0][0], matches[0][1]

    return None, 0


labels = []
confidences = []

for text in df["customer_text"]:
    label, confidence = classify(text)
    labels.append(label)
    confidences.append(confidence)

df["intent_label"] = labels
df["rule_confidence"] = confidences

# Keep only confidently weak-labelled examples.
train = df[
    df["intent_label"].notna()
].copy()

# Limit each class so the training set is not dominated by large intents.
MAX_PER_CLASS = 2500

train = (
    train.groupby("intent_label", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), MAX_PER_CLASS),
            random_state=42
        )
    )
    .reset_index(drop=True)
)

train.to_csv(OUTPUT, index=False)

print("=" * 60)
print("INTENT TRAINING SET")
print("=" * 60)

print(f"Source examples: {len(df)}")
print(f"Training examples: {len(train)}")

print("\nExamples per intent:")
print(train["intent_label"].value_counts().sort_index())

print(f"\nSaved: {OUTPUT}")