from pathlib import Path
import os
import time
import json
import pandas as pd
from openai import OpenAI


MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

INPUT_FILE = Path("data/golden/reply_judge_set_with_evidence.csv")
OUTPUT_FILE = Path("data/processed/llm_judge_results.csv")

REQUEST_DELAY = 3
MAX_RETRIES = 3
RETRY_DELAYS = [10, 20, 40]


def judge_one(client, row):

    customer_text = str(row["customer_text"])
    reply = str(row["reply"])
    should_escalate = str(row["should_escalate"])
    evidence = str(row["retrieved_evidence"])

    prompt = f"""
You are evaluating an AI customer-support agent for an engineering assignment.

The agent is supposed to answer AppleSupport customer messages using historical
support resolutions as evidence.

Evaluate ONLY the response shown below.

CUSTOMER MESSAGE:
{customer_text}

AGENT REPLY:
{reply}

AGENT ESCALATION DECISION:
{should_escalate}

HISTORICAL RETRIEVED EVIDENCE:
{evidence}

Score these seven dimensions.

correctness:
1-5. Is the response factually and procedurally appropriate?

relevance:
1-5. Does it directly address the customer's message?

groundedness:
1-5. Is the response supported by the historical evidence?
Do not give a high score simply because the response sounds plausible.

completeness:
1-5. Does it give enough useful information to move the customer forward?

tone:
1-5. Is it professional, concise, empathetic, and appropriate for AppleSupport?

hallucination:
0 or 1.
Use 1 if the response invents unsupported policies, procedures, URLs,
guarantees, timelines, refunds, capabilities, or other specific claims.
Use 0 otherwise.

escalation_appropriate:
0 or 1.
Use 1 if the agent's escalation decision is appropriate.
Use 0 if the decision is inappropriate.

Return ONLY one line containing exactly seven comma-separated values in this order:`r`ncorrectness,relevance,groundedness,completeness,tone,hallucination,escalation_appropriate`r`nExample:`r`n5,5,4,4,5,0,1`r`nDo not write an explanation or markdown.
"""

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict customer-support evaluation judge. Return JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "text"}
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError("OpenRouter returned empty content.")

            try:
                import re

                pattern = r"(?<!\d)([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([01])\s*,\s*([01])(?!\d)"
                match = re.search(pattern, content)

                if not match:
                    raise ValueError("Could not find seven valid judge scores.")

                values = [int(x) for x in match.groups()]

                return {
                    "correctness": values[0],
                    "relevance": values[1],
                    "groundedness": values[2],
                    "completeness": values[3],
                    "tone": values[4],
                    "hallucination": values[5],
                    "escalation_appropriate": values[6]
                }

            except Exception as parse_error:
                raise RuntimeError(
                    f"Invalid judge score format: {content[:500]!r}"
                ) from parse_error

        except Exception as e:

            error_text = str(e).lower()

            print(
                f"ERROR (attempt {attempt + 1}/{MAX_RETRIES}): "
                f"{e!r}"
            )

            # Daily/project quota: do not waste more requests.
            if (
                "quota" in error_text
                or "daily" in error_text
                or "limit reached" in error_text
            ):
                print(
                    "OpenRouter quota/rate limit reached. "
                    "Stopping the judge run."
                )
                return None

            retryable = (
                "429" in error_text
                or "503" in error_text
                or "502" in error_text
                or "504" in error_text
                or "temporarily" in error_text
                or "rate-limited" in error_text
                or "invalid judge json" in error_text
                or "empty content" in error_text
            )

            if retryable and attempt < MAX_RETRIES - 1:

                wait = RETRY_DELAYS[attempt]

                print(f"Retrying in {wait}s...")
                time.sleep(wait)

                continue

            print("Skipping this example after retries.")
            return None

    return None


def main():

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    df = pd.read_csv(INPUT_FILE)

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
    else:
        existing = pd.DataFrame()

    successful_ids = set()

    if not existing.empty and "eval_id" in existing.columns:
        successful_ids = set(
            existing["eval_id"]
            .dropna()
            .astype(str)
        )

    results = []

    if not existing.empty:
        results = existing.to_dict("records")

    remaining = len(df) - len(successful_ids)

    print("=" * 70)
    print("OpenRouter LLM Judge")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Input examples: {len(df)}")
    print(f"Already successful: {len(successful_ids)}")
    print(f"Remaining: {remaining}")
    print()

    processed = 0

    for _, row in df.iterrows():

        eval_id = str(row["eval_id"])

        if eval_id in successful_ids:
            continue

        processed += 1


        print("-" * 70)
        print(f"[{processed}/{remaining}] Judging {eval_id}")
        print("-" * 70)

        result = judge_one(client, row)

        if result is None:
            print("No result. Stopping or skipping this case.")
            continue

        output_row = {
            "eval_id": eval_id,
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "reply": row["reply"],
            "should_escalate": row["should_escalate"],
            "judge_model": MODEL,
            "judge_status": "success",
            **result
        }

        results.append(output_row)

        successful_ids.add(eval_id)

        out_df = pd.DataFrame(results)

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        out_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(f"Success: {result}")
        print(f"Saved: {OUTPUT_FILE}")

        time.sleep(REQUEST_DELAY)

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Successful judgments: {len(successful_ids)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()




