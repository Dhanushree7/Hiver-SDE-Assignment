import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "intent_labeling.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agent_evaluation.csv"
)


INTENTS = [
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
    "other_unclear",
]


# ============================================================
# Load agent
# ============================================================

from src.agent import AppleSupportAgent


# ============================================================
# Load golden set
# ============================================================

def load_golden():

    df = pd.read_csv(
        GOLDEN_FILE,
        dtype=str
    )

    # Only evaluate manually labelled examples.
    df = df[
        df["intent_label"].notna()
    ].copy()

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
    )

    df["intent_label"] = (
        df["intent_label"]
        .astype(str)
    )

    return df


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("FULL AGENT EVALUATION")
    print("=" * 70)

    df = load_golden()

    print(
        f"Golden examples: {len(df)}"
    )

    # --------------------------------------------------------
    # Initialize agent
    # --------------------------------------------------------

    print("\nInitializing AppleSupport agent...")

    agent = AppleSupportAgent()

    # --------------------------------------------------------
    # Resume existing evaluation if present
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        existing = pd.read_csv(
            OUTPUT_FILE,
            dtype=str
        )

        completed_ids = set(
            existing[
                "customer_tweet_id"
            ].astype(str)
        )

        print(
            f"Existing completed examples: "
            f"{len(completed_ids)}"
        )

    else:

        existing = pd.DataFrame()

        completed_ids = set()

        print(
            "No previous evaluation found. "
            "Starting from scratch."
        )

    results = []

    # --------------------------------------------------------
    # Evaluate each example
    # --------------------------------------------------------

    for index, row in df.iterrows():

        tweet_id = str(
            row["customer_tweet_id"]
        )

        customer_text = str(
            row["customer_text"]
        )

        gold_intent = str(
            row["intent_label"]
        )

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if tweet_id in completed_ids:

            continue

        print(
            f"\n[{index + 1}/{len(df)}]"
        )

        print(
            f"Customer: {customer_text[:180]}"
        )

        # ----------------------------------------------------
        # Run agent
        # ----------------------------------------------------

        try:

            result = agent.run(
                customer_text
            )

            predicted_intent = result.get(
                "intent",
                "ERROR"
            )

            reply = result.get(
                "reply",
                ""
            )

            intent_confidence = result.get(
                "intent_confidence",
                0.0
            )

            intent_margin = result.get(
                "intent_margin",
                0.0
            )

            retrieval_similarity = result.get(
                "retrieval_top1_similarity",
                0.0
            )

            should_escalate = result.get(
                "should_escalate",
                False
            )

            escalation_reason = result.get(
                "escalation_reason",
                ""
            )

            llm_escalation = result.get(
                "llm_escalation_suggestion",
                False
            )

            evidence = result.get(
                "evidence",
                []
            )

            evidence_used = ""

            # The generator returns evidence_used internally,
            # but the current agent exposes the retrieved evidence.
            if evidence:

                evidence_used = json.dumps(
                    [
                        {
                            "customer_text":
                                item.get(
                                    "customer_text",
                                    ""
                                ),
                            "response_text":
                                item.get(
                                    "response_text",
                                    ""),
                            "score":
                                item.get(
                                    "score",
                                    0.0
                                ),
                        }
                        for item in evidence
                    ],
                    ensure_ascii=False
                )

            generation_status = "success"

        except Exception as exc:

            # ------------------------------------------------
            # Do NOT stop the entire evaluation because of
            # one failed generation.
            # ------------------------------------------------

            predicted_intent = "ERROR"

            reply = ""

            intent_confidence = 0.0

            intent_margin = 0.0

            retrieval_similarity = 0.0

            should_escalate = True

            escalation_reason = (
                f"Generation error: {str(exc)}"
            )

            llm_escalation = True

            evidence_used = ""

            generation_status = "error"

            print(
                f"ERROR: {exc}"
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        record = {
            "customer_tweet_id":
                tweet_id,

            "customer_text":
                customer_text,

            "gold_intent":
                gold_intent,

            "predicted_intent":
                predicted_intent,

            "intent_correct":
                predicted_intent == gold_intent,

            "intent_confidence":
                intent_confidence,

            "intent_margin":
                intent_margin,

            "retrieval_top1_similarity":
                retrieval_similarity,

            "reply":
                reply,

            "should_escalate":
                should_escalate,

            "escalation_reason":
                escalation_reason,

            "llm_escalation_suggestion":
                llm_escalation,

            "evidence":
                evidence_used,

            "generation_status":
                generation_status,
        }

        results.append(record)

        # ----------------------------------------------------
        # Save incrementally every 10 examples
        # ----------------------------------------------------

        if (
            len(results) % 10 == 0
            or index == df.index[-1]
        ):

            new_df = pd.DataFrame(
                results
            )

            if not existing.empty:

                combined = pd.concat(
                    [
                        existing,
                        new_df
                    ],
                    ignore_index=True
                )

            else:

                combined = new_df

            # Remove duplicates if the script is resumed.
            combined = combined.drop_duplicates(
                subset=[
                    "customer_tweet_id"
                ],
                keep="last"
            )

            combined.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"\nSaved progress: "
                f"{len(combined)}/{len(df)}"
            )

            results = []

    # --------------------------------------------------------
    # Load final results
    # --------------------------------------------------------

    final_df = pd.read_csv(
        OUTPUT_FILE,
        dtype=str
    )

    # Convert relevant columns.
    final_df["gold_intent"] = (
        final_df["gold_intent"]
        .astype(str)
    )

    final_df["predicted_intent"] = (
        final_df["predicted_intent"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Intent evaluation
    # --------------------------------------------------------

    valid = final_df[
        final_df["predicted_intent"] != "ERROR"
    ].copy()

    y_true = valid[
        "gold_intent"
    ]

    y_pred = valid[
        "predicted_intent"
    ]

    print("\n")
    print("=" * 70)
    print("FINAL INTENT RESULTS")
    print("=" * 70)

    print(
        f"Successfully evaluated: "
        f"{len(valid)}/{len(final_df)}"
    )

    print(
        f"Generation failures: "
        f"{len(final_df) - len(valid)}"
    )

    if len(valid) > 0:

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=INTENTS,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            labels=INTENTS,
            average="weighted",
            zero_division=0
        )

        print(
            f"\nAccuracy: {accuracy:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        print(
            f"Weighted F1: {weighted_f1:.4f}"
        )

        print("\nClassification report:")

        print(
            classification_report(
                y_true,
                y_pred,
                labels=INTENTS,
                zero_division=0
            )
        )

    # --------------------------------------------------------
    # Retrieval statistics
    # --------------------------------------------------------

    retrieval_scores = pd.to_numeric(
        final_df[
            "retrieval_top1_similarity"
        ],
        errors="coerce"
    ).dropna()

    if len(retrieval_scores) > 0:

        print("\n")
        print("=" * 70)
        print("RETRIEVAL RESULTS")
        print("=" * 70)

        print(
            f"Mean Top-1 similarity: "
            f"{retrieval_scores.mean():.4f}"
        )

        print(
            f"Median Top-1 similarity: "
            f"{retrieval_scores.median():.4f}"
        )

        print(
            f"Top-1 >= 0.50: "
            f"{(retrieval_scores >= 0.50).mean():.2%}"
        )

        print(
            f"Top-1 >= 0.60: "
            f"{(retrieval_scores >= 0.60).mean():.2%}"
        )

        print(
            f"Top-1 >= 0.70: "
            f"{(retrieval_scores >= 0.70).mean():.2%}"
        )

    # --------------------------------------------------------
    # Generation status
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATION STATUS")
    print("=" * 70)

    print(
        final_df[
            "generation_status"
        ].value_counts()
    )

    print("\n")
    print("=" * 70)

    print(
        "Evaluation saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()