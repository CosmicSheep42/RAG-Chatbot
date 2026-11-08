# Datapizza LLM-as-a-Judge evaluation

This is a single-pass blinded evaluation by OpenAI GPT-6 via ChatGPT, not a human review or expert validation.
Evidence support was not evaluated and remains unavailable for all 336 responses.

## Held-out test split

| Configuration | Correctness mean / 2 | Completeness mean / 2 | Correctness 2/2 | Both 2/2 |
|---|---:|---:|---:|---:|
| bm25 | 0.614 | 0.500 | 8/44 | 3/44 |
| dense | 0.795 | 0.636 | 10/44 | 3/44 |
| full | 0.727 | 0.614 | 10/44 | 4/44 |
| hybrid | 0.705 | 0.500 | 10/44 | 1/44 |
| hybrid-rerank | 0.886 | 0.705 | 11/44 | 3/44 |
| standalone | 0.091 | 0.091 | 0/44 | 0/44 |

Scores use an ordinal 0–2 rubric and must not be read as accuracy percentages. The results are limited by one judge, no human adjudication, one D&D 5e domain, 56 questions, and potentially ambiguous reference criteria for medium questions 001, 011, 013, 019, and 028. Medium question 004 has uncertain evidence annotations and lacks gold passages.
On the held-out test split, 8 questions received correctness 0 from all six systems. Evidence support was unavailable for all 336 reviewed responses.

The highest average correctness in this evaluation was observed for Hybrid + Rerank. This supports a finding on this fixed benchmark only; it does not establish universal superiority or isolate reranking as the sole cause. Groundedness and claim-level citation support remain unmeasured.
