from dataclasses import dataclass


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: str


class EscalationPolicy:
    """
    Conservative escalation policy for the AppleSupport agent.

    The LLM can suggest escalation, but deterministic safety and
    confidence rules can override that suggestion.
    """

    # Intents where automated handling is risky because the issue may
    # require account-specific or transactional information.
    SENSITIVE_INTENTS = {
        "apple_id_account",
        "purchases_orders_refunds",
    }

    def __init__(
        self,
        min_intent_confidence=0.45,
        min_retrieval_similarity=0.55,
        min_intent_margin=0.05,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity
        self.min_intent_margin = min_intent_margin

    def decide(
        self,
        intent,
        intent_confidence,
        intent_margin,
        retrieval_top1_similarity,
        llm_should_escalate=False,
        llm_escalation_reason="",
    ):
        # -------------------------------------------------
        # Rule 1: Insufficient information
        # -------------------------------------------------

        if intent == "other_unclear":
            return EscalationDecision(
                True,
                "The message does not contain enough information "
                "to assign a reliable support intent."
            )

        # -------------------------------------------------
        # Rule 2: Low classifier confidence
        # -------------------------------------------------

        if intent_confidence < self.min_intent_confidence:
            return EscalationDecision(
                True,
                f"Low intent confidence "
                f"({intent_confidence:.3f})."
            )

        # -------------------------------------------------
        # Rule 3: Ambiguous intent
        # -------------------------------------------------

        if intent_margin < self.min_intent_margin:
            return EscalationDecision(
                True,
                f"Ambiguous intent prediction; top-two "
                f"intent margin is only {intent_margin:.3f}."
            )

        # -------------------------------------------------
        # Rule 4: Weak historical evidence
        # -------------------------------------------------

        if retrieval_top1_similarity < self.min_retrieval_similarity:
            return EscalationDecision(
                True,
                f"Insufficient historical evidence; "
                f"top retrieval similarity is "
                f"{retrieval_top1_similarity:.3f}."
            )

        # -------------------------------------------------
        # Rule 5: Sensitive/account-specific issue
        # -------------------------------------------------

        if intent in self.SENSITIVE_INTENTS:
            return EscalationDecision(
                True,
                f"{intent} may require account-specific or "
                f"transactional information that should not be "
                f"handled automatically."
            )

        # -------------------------------------------------
        # Rule 6: LLM recommends escalation
        # -------------------------------------------------

        if llm_should_escalate:
            reason = llm_escalation_reason.strip()

            if not reason:
                reason = (
                    "The response generator determined that "
                    "the available evidence is insufficient "
                    "for safe automated handling."
                )

            return EscalationDecision(
                True,
                f"LLM escalation recommendation: {reason}"
            )

        # -------------------------------------------------
        # Otherwise: auto-handle
        # -------------------------------------------------

        return EscalationDecision(
            False,
            "Intent, evidence, and response generation "
            "signals are sufficient for automated handling."
        )


if __name__ == "__main__":

    policy = EscalationPolicy()

    # Example 1: safe battery question
    decision = policy.decide(
        intent="battery_power",
        intent_confidence=0.82,
        intent_margin=0.21,
        retrieval_top1_similarity=0.80,
        llm_should_escalate=False,
        llm_escalation_reason="",
    )

    print("Battery example:")
    print(decision)

    # Example 2: account issue
    decision = policy.decide(
        intent="apple_id_account",
        intent_confidence=0.80,
        intent_margin=0.20,
        retrieval_top1_similarity=0.75,
        llm_should_escalate=False,
        llm_escalation_reason="",
    )

    print("\nAccount example:")
    print(decision)

    # Example 3: low confidence
    decision = policy.decide(
        intent="hardware_accessories",
        intent_confidence=0.32,
        intent_margin=0.02,
        retrieval_top1_similarity=0.48,
        llm_should_escalate=False,
        llm_escalation_reason="",
    )

    print("\nLow-confidence example:")
    print(decision)