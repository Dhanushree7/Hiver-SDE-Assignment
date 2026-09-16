import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


TRAIN_PATH = "data/processed/intent_train.csv"
MODEL_PATH = "data/processed/intent_classifier.joblib"
LABEL_PATH = "data/processed/intent_labels.npy"


df = pd.read_csv(TRAIN_PATH)

texts = df["customer_text"].astype(str).tolist()
labels = df["intent_label"].astype(str).tolist()

print(f"Training examples: {len(texts)}")
print(f"Number of intents: {len(set(labels))}")

print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating embeddings...")

X = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

X = np.asarray(X)

print(f"\nEmbedding shape: {X.shape}")

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=1000,
    C=3.0,
    class_weight="balanced",
    random_state=42
)

classifier.fit(X, y)

joblib.dump(classifier, MODEL_PATH)
np.save(LABEL_PATH, encoder.classes_)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Model saved: {MODEL_PATH}")
print(f"Labels saved: {LABEL_PATH}")

print("\nClasses:")
for i, label in enumerate(encoder.classes_):
    print(f"{i}: {label}")