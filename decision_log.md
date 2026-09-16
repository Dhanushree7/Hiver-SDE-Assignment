# Decision Log

## 1. Brand Selection
- Decision: Selected AppleSupport.
- Rationale: Sufficient conversation volume and diverse support intents.

## 2. Intent Taxonomy
- Decision: Defined 11 support intents.
- Rationale: Covers common customer issues while providing manageable classification categories.

## 3. Weak Supervision
- Decision: Used weakly supervised examples for initial classifier training.
- Rationale: Reduced manual labeling requirements.

## 4. Golden Dataset
- Decision: Created 200 manually labeled intent examples.
- Rationale: Provides an independent evaluation reference.

## 5. Leakage Prevention
- Decision: Excluded golden examples and duplicate customer texts from the retrieval corpus.
- Rationale: Prevents evaluation contamination.

## 6. Baseline Comparison
- Decision: Evaluated trivial and rule-based baselines.
- Rationale: Establishes reference performance before learned approaches.

## 7. Retrieval Model
- Decision: Used semantic retrieval with 384-dimensional embeddings.
- Rationale: Supports matching customer messages with historical resolutions.

## 8. Retrieval Evaluation
- Decision: Evaluated similarity thresholds and top-ranked evidence.
- Rationale: Measures whether retrieved examples are sufficiently relevant.

## 9. Generation Model
- Decision: Used Groq-hosted language models for response generation and judging.
- Rationale: Provides an accessible inference interface for experimentation.

## 10. Grounded Responses
- Decision: Restricted generated responses to information supported by historical evidence.
- Rationale: Reduces unsupported claims and hallucinations.

## 11. Escalation Strategy
- Decision: Escalated low-confidence or uncertain cases.
- Rationale: Reduces the risk of automatically handling cases without sufficient evidence.

## 12. Human and LLM Evaluation
- Decision: Used human ratings and an LLM-as-judge evaluation.
- Rationale: Combines manual quality assessment with scalable supplementary evaluation.

## Limitation
The LLM judge and human evaluator may apply different interpretations of escalation appropriateness. Agreement results are therefore reported as supplementary evidence rather than definitive ground truth.
