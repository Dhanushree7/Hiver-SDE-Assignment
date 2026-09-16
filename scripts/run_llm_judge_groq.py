import csv
import json
import os
import time
from pathlib import Path

from openai import OpenAI

INPUT_FILE = Path("data/golden/reply_judge_set_with_evidence.csv")
OUTPUT_FILE = Path("data/processed/llm_judge_results_groq.csv")

MODEL = "qwen/qwen3.8-27b"
PROMPT_VERSION = "groq_judge_v4_escalation_rubric"

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = """
You are a strict evaluator of an Apple customer-support AI agent.

Evaluate the agent reply using the customer message and retrieved historical evidence.

Return ONLY valid JSON with exactly these integer fields:
correctness, relevance, groundedness, completeness, tone,
hallucination, escalation_appropriate

The first five fields must be integers from 1 to 5.
hallucination must be 0 or 1:
0 = no unsupported claim
1 = unsupported or invented claim

escalation_appropriate must be 0 or 1.

Evaluate whether the agent's escalation decision is appropriate for the customer's issue.

Use 1 only when escalation is justified by factors such as:
- Account access, payment, refund, or security issues requiring authorized support.
- Potential hardware failure, safety concern, or unresolved technical problem.
- Missing information or uncertainty that prevents a reliable self-service response.
- A request that historical evidence cannot safely resolve.

Use 0 when:
- The issue is a routine question that can be answered using the provided evidence.
- The reply escalates unnecessarily without a clear reason.
- The escalation decision does not match the customer's actual problem.
- The reply should provide a safe troubleshooting step or clarification first.

Judge the decision itself, not whether the customer sounds angry.
Do not assume that every complaint or technical issue requires escalation.

Groundedness means the reply is supported by the provided evidence.
Do not assume policies, URLs, procedures, refunds, guarantees, or timelines
that are not supported by the evidence.
"""

def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])

def validate_result(result):
    required = [
        "correctness",
        "relevance",
        "groundedness",
        "completeness",
        "tone",
        "hallucination",
        "escalation_appropriate",
    ]

    for field in required:
        if field not in result:
            raise ValueError(f"Missing field: {field}")

    for field in required[:5]:
        if not isinstance(result[field], int) or not 1 <= result[field] <= 5:
            raise ValueError(f"Invalid score for {field}")

    for field in ["hallucination", "escalation_appropriate"]:
        if result[field] not in [0, 1]:
            raise ValueError(f"Invalid binary value for {field}")

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    completed = {}

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("judge_status") == "success":
                    completed[row.get("example_id")] = row

    output_fields = [
        "example_id",
        "correctness",
        "relevance",
        "groundedness",
        "completeness",
        "tone",
        "hallucination",
        "escalation_appropriate",
        "judge_status",
        "judge_model",
        "prompt_version",
        "error",
    ]

    results = list(completed.values())

    for index, row in enumerate(rows, start=1):
        example_id = row.get("example_id") or row.get("row_id") or str(index)

        if example_id in completed:
            print(f"[{index}/{len(rows)}] Skipping {example_id}")
            continue

        evidence = row.get("retrieved_evidence", row.get("evidence", ""))
        customer_text = row.get("customer_text", "")
        reply = row.get("reply", "")
        predicted_intent = row.get("predicted_intent", "Not provided in this evaluation file")
        should_escalate = row.get("should_escalate", "")
        escalation_reason = row.get("escalation_reason", "")

        user_prompt = f"""
Customer message:
{customer_text}

Predicted intent:
{predicted_intent}

Agent reply:
{reply}

Agent escalation decision:
{should_escalate}

Escalation reason:
{escalation_reason}

Retrieved historical evidence:
{evidence}

Evaluate the reply and return only the required JSON object.
"""

        result_row = {
            "example_id": example_id,
            "judge_model": MODEL,
            "prompt_version": PROMPT_VERSION,
        }

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=200,
                )

                content = response.choices[0].message.content

                if not content:
                    raise ValueError("Empty model response")

                judged = extract_json(content)
                validate_result(judged)

                result_row.update(judged)
                result_row["judge_status"] = "success"
                result_row["error"] = ""
                break

            except Exception as error:
                result_row["judge_status"] = "error"
                result_row["error"] = str(error)[:500]
                print(f"Attempt {attempt + 1} failed for {example_id}: {error}")
                time.sleep(3 * (attempt + 1))

        results = [
            existing for existing in results
            if existing.get("example_id") != example_id
        ]
        results.append(result_row)

        with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(results)

        print(f"[{index}/{len(rows)}] {example_id}: {result_row['judge_status']}")

    successful = sum(
        1 for row in results if row.get("judge_status") == "success"
    )

    print(f"Completed: {successful}/{len(rows)}")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()



