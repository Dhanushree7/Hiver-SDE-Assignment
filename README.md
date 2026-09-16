# Hiver SDE Intern Assignment
## AI Customer Support Agent

An AI-powered customer support agent built using the Customer Support on Twitter dataset.

**Selected Brand:** AppleSupport

## Features

- Customer intent classification
- Historical support response retrieval
- Grounded AI response generation
- Auto-handle or escalation decision
- Human and LLM-based evaluation

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Sentence Transformers
- Groq LLM
- Retrieval-based response generation

## Project Structure

- data/golden: Human-labelled evaluation datasets
- data/processed: Processed datasets and evaluation results
- scripts: Data preparation, training, and evaluation scripts
- src: Agent implementation
- decision_log.md: Project decisions and trade-offs
- failure_analysis.md: Failure analysis with examples

## Intent Categories

The project uses 11 customer-support intents:

1. ios_update_issues
2. apps_services
3. battery_power
4. device_performance
5. hardware_accessories
6. keyboard_input
7. connectivity
8. photos_data
9. apple_id_account
10. purchases_orders_refunds
11. other_unclear

## Evaluation Results

| Method | Accuracy |
|---|---:|
| Trivial baseline | 20.5% |
| Rule-based baseline | 54.0% |
| Zero-shot MiniLM | 57.5% |
| Weakly supervised classifier | 65.5% |

### Retrieval Evaluation

- Mean Top-1 similarity: approximately 0.73
- Similarity >= 0.50: 97.5%
- Similarity >= 0.60: 91.0%
- Similarity >= 0.65: 82.0%

### Escalation Evaluation

The escalation audit included 50 human-labelled examples.

The agent showed a safety-oriented tendency to escalate more cases. This reduced some false auto-handling decisions but increased over-escalation.

### Response Evaluation

Human evaluation was performed on 40 examples using correctness, relevance, groundedness, completeness, tone, hallucination, and escalation appropriateness.

LLM-as-judge results were treated as supplementary evidence rather than human ground truth.

## What Is Misleading About My Headline Number?

The classification accuracy should not be interpreted as production-level performance.

The result depends on the selected AppleSupport subset, intent taxonomy, weakly supervised labels, dataset distribution, and ambiguous customer messages.

Classification accuracy alone does not measure support response quality, retrieval correctness, or escalation safety. These metrics must be evaluated together.

## Failure Analysis

The major observed failure modes include:

1. Hardware issues classified as connectivity problems.
2. Camera issues classified as update issues.
3. Keyboard issues classified as update issues.
4. Purchase or book-download issues classified as update issues.
5. Connectivity issues classified as device-performance problems.

Detailed examples are available in failure_analysis.md.

## Human Evaluation and Agreement

A second annotation pass by the same annotator was used to check consistency.

This is reported as intra-rater agreement, not inter-annotator agreement.

The repeated annotation pass achieved 100% agreement with Cohen's kappa of 1.000. This does not measure agreement between independent annotators.

## Decision Log

Project decisions and trade-offs are documented in decision_log.md.

## Dataset Setup

Download the Customer Support on Twitter dataset from Kaggle.

Place the raw dataset at:

data/raw/twcs.csv

The raw dataset is excluded from Git because of its large size.

## Reproduction

1. Clone the repository.
2. Install the required Python dependencies.
3. Download the dataset.
4. Place twcs.csv inside data/raw.
5. Run the preprocessing scripts.
6. Run the classification and retrieval evaluation scripts.
7. Run the agent evaluation scripts.

The available scripts are located in the scripts directory.

## Limitations

- Some training labels are weakly supervised.
- Customer messages may contain multiple intents.
- Escalation labels can involve subjective interpretation.
- LLM-as-judge results may differ from human ratings.
- Retrieval similarity does not guarantee response correctness.
- Results do not represent production performance.

## Future Improvements

- Expand the independently labelled golden dataset.
- Improve intent definitions and confidence calibration.
- Improve multi-intent handling.
- Improve escalation policies.
- Add stronger evidence attribution.
- Use multiple independent annotators.
- Improve response completeness.

## Attribution

This project uses the Customer Support on Twitter dataset. The original dataset source and borrowed methods should be cited before final submission.
