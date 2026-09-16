import os
import json
import hashlib
from pathlib import Path

from groq import Groq


# ============================================================
# Configuration
# ============================================================

CACHE_DIR = Path("cache/generation")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "openai/gpt-oss-20b"

# Change this whenever the system prompt/output rules change.
# This prevents old cached responses from being reused.
PROMPT_VERSION = "v3_specific_grounded"


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an AI customer-support assistant for AppleSupport.

Your job is to draft a helpful response to a customer using ONLY the
historical support evidence provided to you.

Rules:

1. Ground the response in the historical evidence.

2. Directly address the customer's actual problem. Do not give a
   generic response when the evidence supports a more specific answer.

3. Use relevant details from the historical evidence, but adapt them
   to the customer's situation. Do not copy historical responses
   verbatim.

4. If the historical responses show that AppleSupport normally asks
   for information such as device model, iOS version, symptoms,
   account details, or other diagnostic information, ask for the
   relevant information when it is needed.

5. Do not invent Apple policies, troubleshooting procedures, URLs,
   product capabilities, refunds, guarantees, or timelines.

6. Never copy URLs from historical responses. If a historical response
   contains a URL, summarize the relevant guidance instead.

7. Do not include ANY URL in the customer-facing reply.

8. If the historical evidence is insufficient to confidently answer
   the customer's problem, recommend escalation rather than guessing.

9. Keep the response concise, professional, empathetic, and suitable
   for a Twitter-style customer-support conversation.

10. Do not mention that you are an AI.

11. Do not mention retrieval, historical evidence, similarity scores,
    or internal system processes.

12. Prefer a useful next step over a vague statement such as
    "We can help." The response should either provide grounded
    guidance or ask a specific useful question.

Return ONLY valid JSON with exactly these fields:

{
  "reply": "customer-facing response",
  "should_escalate": true or false,
  "escalation_reason": "brief reason",
  "evidence_used": [1, 2]
}

The evidence_used field must contain the numbers of the historical
examples that materially influenced the response.

If the evidence does not support a concrete troubleshooting step,
do not invent one. Ask for the information needed or recommend
escalation instead.
"""


# ============================================================
# Response Generator
# ============================================================

class ResponseGenerator:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

    # ========================================================
    # Cache Key
    # ========================================================

    def _cache_key(
        self,
        customer_text,
        intent,
        evidence
    ):

        data = {
            "customer_text": customer_text,
            "intent": intent,
            "evidence": evidence,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
        }

        raw = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # Generate Response
    # ========================================================

    def generate(
        self,
        customer_text,
        intent,
        evidence,
    ):

        # ----------------------------------------------------
        # Create cache key
        # ----------------------------------------------------

        cache_key = self._cache_key(
            customer_text,
            intent,
            evidence
        )

        cache_file = (
            CACHE_DIR /
            f"{cache_key}.json"
        )

        # ----------------------------------------------------
        # Cache lookup
        # ----------------------------------------------------

        if cache_file.exists():

            with open(
                cache_file,
                "r",
                encoding="utf-8"
            ) as f:

                result = json.load(f)

            # Safety check cached responses too.
            cached_reply = str(
                result.get("reply", "")
            )

            cached_reply_lower = cached_reply.lower()

            if (
                "http://" not in cached_reply_lower
                and "https://" not in cached_reply_lower
                and "www." not in cached_reply_lower
            ):
                return result

            # Ignore unsafe cached responses.
            print(
                "Unsafe cached response detected. "
                "Regenerating..."
            )

        # ----------------------------------------------------
        # Format historical evidence
        # ----------------------------------------------------

        evidence_text = []

        for i, item in enumerate(
            evidence,
            start=1
        ):

            evidence_text.append(
                f"""
HISTORICAL CASE {i}

Customer:
{item["customer_text"]}

AppleSupport response:
{item["response_text"]}

Similarity:
{item.get("score", 0):.3f}
"""
            )

        evidence_block = "\n".join(
            evidence_text
        )

        # ----------------------------------------------------
        # User Prompt
        # ----------------------------------------------------

        user_prompt = f"""
Customer message:

{customer_text}

Predicted intent:

{intent}

Historical support evidence:

{evidence_block}

Draft the best grounded response for this customer.

Important:
- Directly address the customer's problem.
- Use the most relevant information from the historical cases.
- If additional information is needed, ask a specific useful question.
- Do not invent troubleshooting steps.
- Do not invent Apple policies or capabilities.
- Do not copy historical responses verbatim.
- Never include a URL.
- Never include http://
- Never include https://
- Never include www.
"""

        # ----------------------------------------------------
        # Groq Request
        # ----------------------------------------------------

        response = (
            self.client.chat.completions.create(
                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.2,

                max_completion_tokens=600,

                response_format={
                    "type": "json_object"
                },

                include_reasoning=False,
            )
        )

        # ----------------------------------------------------
        # Extract model output
        # ----------------------------------------------------

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            result = json.loads(
                content
            )

        except json.JSONDecodeError:

            raise RuntimeError(
                "Groq returned invalid JSON:\n"
                f"{content}"
            )

        # ----------------------------------------------------
        # Validate required fields
        # ----------------------------------------------------

        required_fields = {
            "reply",
            "should_escalate",
            "escalation_reason",
            "evidence_used",
        }

        missing = (
            required_fields -
            set(result.keys())
        )

        if missing:

            raise RuntimeError(
                "Generator response missing "
                f"fields: {missing}"
            )

        # ----------------------------------------------------
        # Validate / normalize fields
        # ----------------------------------------------------

        result["reply"] = str(
            result["reply"]
        )

        result["escalation_reason"] = str(
            result["escalation_reason"]
        )

        if not isinstance(
            result["evidence_used"],
            list
        ):
            raise RuntimeError(
                "evidence_used must be a list."
            )

        result["should_escalate"] = bool(
            result["should_escalate"]
        )

        # ----------------------------------------------------
        # Safety Check: No URLs
        # ----------------------------------------------------

        reply = result["reply"]
        reply_lower = reply.lower()

        if (
            "http://" in reply_lower
            or "https://" in reply_lower
            or "www." in reply_lower
        ):

            raise RuntimeError(
                "Generator returned a URL in "
                "the customer-facing reply."
            )

        # ----------------------------------------------------
        # Save Cache
        # ----------------------------------------------------

        with open(
            cache_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        return result


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    import sys

    # Allow imports from project root.
    sys.path.insert(
        0,
        str(
            Path(__file__)
            .resolve()
            .parents[2]
        )
    )

    from src.retrieval.retriever import Retriever

    # --------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------

    retriever = Retriever()
    generator = ResponseGenerator()

    # --------------------------------------------------------
    # Test customer message
    # --------------------------------------------------------

    customer_message = (
        "My iPhone battery is draining really "
        "quickly after the latest iOS update. "
        "What can I do?"
    )

    # --------------------------------------------------------
    # Retrieve historical evidence
    # --------------------------------------------------------

    evidence = retriever.search(
        customer_message,
        top_k=3
    )

    # --------------------------------------------------------
    # Generate grounded response
    # --------------------------------------------------------

    result = generator.generate(
        customer_text=customer_message,
        intent="battery_power",
        evidence=evidence
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CUSTOMER"
    )

    print(
        "=" * 70
    )

    print(
        customer_message
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATED AGENT OUTPUT"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
