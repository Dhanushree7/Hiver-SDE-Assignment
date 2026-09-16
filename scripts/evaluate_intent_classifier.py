import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


GOLDEN_PATH = "data/golden/intent_labeling.csv"
MODEL_PATH = "data/processed/intent_classifier.joblib"
LABEL_PATH = "data/processed/intent_labels.npy"

OUTPUT_PATH = "data/processed/learned_intent_predictions.csv"


# Load golden evaluation set
df = pd.read_csv(GOLDEN_PATH)

df = df[df["intent_label"].notna()].copy()
df["intent_label"] = df["intent_label"].astype(str)

print(f"Golden examples: {len(df)}")


# Load classifier
classifier = joblib.load(MODEL_PATH)
labels = np.load(LABEL_PATH, allow_pickle=True)


# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Generate embeddings
texts = df["customer_text"].astype(str).tolist()

print("Generating golden-set embeddings...")

X = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)


# Predict
print("Running predictions...")

y_pred_ids = classifier.predict(X)
y_pred = labels[y_pred_ids]

y_true = df["intent_label"].values


# Results
print("\n" + "=" * 70)
print("LEARNED INTENT CLASSIFIER RESULTS")
print("=" * 70)

print(f"\nExamples evaluated: {len(df)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)


# Save predictions
df["predicted_intent"] = y_pred

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved: {OUTPUT_PATH}")