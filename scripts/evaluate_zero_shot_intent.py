import sys
sys.path.insert(0, ".")

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.intent.classifier import IntentClassifier


GOLDEN_PATH = "data/golden/intent_labeling.csv"
OUTPUT_PATH = "data/processed/zero_shot_intent_predictions.csv"


df = pd.read_csv(GOLDEN_PATH)

# Keep only the manually labeled golden examples
df = df[df["intent_label"].notna()].copy()
df["intent_label"] = df["intent_label"].astype(str)

print(f"Golden examples: {len(df)}")
classifier = IntentClassifier()

predictions = []

for i, row in df.iterrows():
    text = str(row["customer_text"])

    result = classifier.predict(text, top_k=1)[0]
    predictions.append(result["intent"])

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)}")


df["predicted_intent"] = predictions

y_true = df["intent_label"]
y_pred = df["predicted_intent"]


print("\n" + "=" * 70)
print("ZERO-SHOT INTENT CLASSIFIER RESULTS")
print("=" * 70)

print(f"\nExamples evaluated: {len(df)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

labels = sorted(df["intent_label"].dropna().unique())

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
))

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")