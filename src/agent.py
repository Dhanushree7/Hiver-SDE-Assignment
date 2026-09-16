import sys
from pathlib import Path

# Make project root importable when running:
# python src\agent.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.intent.classifier import IntentClassifier
from src.retrieval.retriever import Retriever
from src.generation.generator import ResponseGenerator
from src.escalation.policy import EscalationPolicy


class AppleSupportAgent:

    def __init__(self):
        print("Loading intent classifier...")
        self.classifier = IntentClassifier()

        print("Loading retriever...")
        self.retriever = Retriever()

        print("Loading response generator...")
        self.generator = ResponseGenerator()

        self.escalation_policy = EscalationPolicy()

    def run(self, customer_text):

        # -------------------------------------------------
        # 1. Intent classification
        # -------------------------------------------------

        predictions = self.classifier.predict(
            customer_text,
            top_k=2
        )

        top_intent = predictions[0]["intent"]
        intent_confidence = predictions[0]["score"]

        if len(predictions) > 1:
            second_confidence = predictions[1]["score"]
            intent_margin = (
                intent_confidence - second_confidence
            )
        else:
            intent_margin = intent_confidence

        # -------------------------------------------------
        # 2. Historical retrieval
        # -------------------------------------------------

        evidence = self.retriever.search(
            customer_text,
            top_k=3
        )

        if evidence:
            retrieval_top1 = evidence[0].get(
                "score",
                0.0
            )
        else:
            retrieval_top1 = 0.0

        # -------------------------------------------------
        # 3. Generate grounded response
        # -------------------------------------------------

        generation = self.generator.generate(
            customer_text=customer_text,
            intent=top_intent,
            evidence=evidence
        )

        # -------------------------------------------------
        # 4. Deterministic escalation policy
        # -------------------------------------------------

        escalation = self.escalation_policy.decide(
            intent=top_intent,
            intent_confidence=intent_confidence,
            intent_margin=intent_margin,
            retrieval_top1_similarity=retrieval_top1,
            llm_should_escalate=generation[
                "should_escalate"
            ],
            llm_escalation_reason=generation[
                "escalation_reason"
            ],
        )

        # -------------------------------------------------
        # 5. Return complete result
        # -------------------------------------------------

        return {
            "customer_text": customer_text,
            "intent": top_intent,
            "intent_confidence": round(
                intent_confidence,
                4
            ),
            "intent_margin": round(
                intent_margin,
                4
            ),
            "retrieval_top1_similarity": round(
                retrieval_top1,
                4
            ),
            "evidence": evidence,
            "reply": generation["reply"],
            "should_escalate": escalation.should_escalate,
            "escalation_reason": escalation.reason,
            "llm_escalation_suggestion": generation[
                "should_escalate"
            ],
        }


def print_result(result):

    print("\n" + "=" * 70)
    print("APPLE SUPPORT AGENT")
    print("=" * 70)

    print("\nCUSTOMER:")
    print(result["customer_text"])

    print("\nINTENT:")
    print(
        f'{result["intent"]} '
        f'(confidence={result["intent_confidence"]})'
    )

    print(
        f'Intent margin: {result["intent_margin"]}'
    )

    print(
        f'Retrieval top-1 similarity: '
        f'{result["retrieval_top1_similarity"]}'
    )

    print("\nHISTORICAL EVIDENCE:")

    for i, item in enumerate(
        result["evidence"],
        start=1
    ):
        print(
            f"\n[{i}] similarity="
            f'{item.get("score", 0):.3f}'
        )

        print(
            f'Customer: {item["customer_text"]}'
        )

        print(
            f'Response: {item["response_text"]}'
        )

    print("\nGENERATED REPLY:")
    print(result["reply"])

    print("\nDECISION:")

    if result["should_escalate"]:
        print("ESCALATE")
    else:
        print("AUTO-HANDLE")

    print(
        f'Reason: {result["escalation_reason"]}'
    )

    print(
        "\nLLM escalation suggestion: "
        f'{result["llm_escalation_suggestion"]}'
    )

    print("=" * 70)


if __name__ == "__main__":

    agent = AppleSupportAgent()

    customer_message = (
        "My iPhone battery is draining really quickly "
        "after the latest iOS update. What can I do?"
    )

    result = agent.run(customer_message)

    print_result(result)