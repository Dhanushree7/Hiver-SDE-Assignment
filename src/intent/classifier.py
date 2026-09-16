import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INTENTS = {
    "ios_update_issues": "iOS or macOS software updates causing problems, update failures, update bugs, or requests about updating/downgrading",
    "apps_services": "Apple apps or services not working, crashing, syncing, or behaving incorrectly, such as Music, Messages, Mail, iTunes, Siri, or App Store",
    "battery_power": "battery draining quickly, charging problems, overheating related to battery, or poor battery life",
    "device_performance": "device freezing, crashing, lagging, rebooting, slow performance, or general device instability when no specific update cause is central",
    "hardware_accessories": "physical hardware or accessories such as screens, cameras, speakers, headphones, chargers, cables, keyboards, or SSDs",
    "keyboard_input": "keyboard, typing, autocorrect, character rendering, or incorrect letters/symbols",
    "connectivity": "Wi-Fi, Bluetooth, cellular, SIM, or network connection problems",
    "photos_data": "photos, contacts, backups, iCloud data, missing data, data loss, or transferring personal data",
    "apple_id_account": "Apple ID, iCloud account access, passwords, verification, account recovery, or account security",
    "purchases_orders_refunds": "purchases, orders, subscriptions, billing, refunds, payments, or shipping",
    "other_unclear": "unclear or insufficient information to determine a specific customer support intent",
}


class IntentClassifier:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.intent_names = list(INTENTS.keys())
        descriptions = list(INTENTS.values())

        self.intent_embeddings = self.model.encode(
            descriptions,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def predict(self, text, top_k=3):
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = np.dot(self.intent_embeddings, embedding)
        ranking = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranking:
            results.append({
                "intent": self.intent_names[idx],
                "score": float(scores[idx]),
            })

        return results


if __name__ == "__main__":
    classifier = IntentClassifier()

    examples = [
        "My iPhone battery drains really quickly",
        "Bluetooth keeps disconnecting",
        "The keyboard changes I into a question mark",
        "Apple Music keeps crashing",
        "My iPhone freezes constantly",
        "I need to cancel my iPhone order",
    ]

    for text in examples:
        result = classifier.predict(text, top_k=1)[0]
        print(f"\n{text}")
        print(f" -> {result['intent']} ({result['score']:.3f})")