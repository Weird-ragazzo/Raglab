# Latest real-PDF evaluation

Run: `20261014T102321015371Z`  
Split: `dev` · Dataset: `1047b85764b840dc`

## Executive summary

| Metric | Value |
| --- | ---: |
| recall_at_k | 0.85 |
| recall_at_5 | 0.85 |
| mrr | 0.7166666666666667 |
| precision_at_k | 0.25 |
| hit_rate | 0.9 |
| citation_validity | not run |
| correct_page_citation_rate | not run |
| abstention_accuracy | not run |
| judge_faithfulness | not run |
| judge_correctness | not run |
| judge_relevance | not run |
| retrieval_latency_ms_mean | 20.624417166800413 |
| retrieval_latency_ms_p95 | 23.619316199983587 |
| end_to_end_latency_ms_p95 | not run |
| failed_cases | 0 |

## Breakdown by question type

| Type | Cases | Recall@K | MRR | Citation validity | Faithfulness |
| --- | ---: | ---: | ---: | ---: | ---: |
| comparison | 4 | 0.5 | 0.4375 | None | None |
| definition | 2 | 1.0 | 1.0 | None | None |
| factual | 12 | 0.875 | 0.8125 | None | None |
| method | 1 | 1.0 | 0.3333333333333333 | None | None |
| numeric | 5 | 1.0 | 0.75 | None | None |

## Breakdown by PDF

| PDF | Cases | Recall@K | MRR | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| Ayush & Aman - BIAS IEEE PAPER.pdf | 6 | 0.75 | 0.5555555555555556 | 0.8333333333333334 |
| esmoopen-2020-000871.pdf | 6 | 0.8333333333333334 | 0.625 | 0.8333333333333334 |
| pone.0312962.pdf | 6 | 0.75 | 0.7083333333333334 | 0.8333333333333334 |
| unanswerable | 4 | None | None | None |
| zjw2214.pdf | 5 | 0.8 | 0.85 | 1.0 |

## Worst five failures

### real-0020 (comparison)

**Question:** How do the findings regarding the efficacy and evaluation of agents without documented single-agent activity in cancer combination trials relate to the obstacles and prevalence of traditional healing practices for treating jaundice?
**Retrieved pages:** pone.0312962.pdf:2, pone.0312962.pdf:8, pone.0312962.pdf:11, pone.0312962.pdf:1, pone.0312962.pdf:18, pone.0312962.pdf:15, pone.0312962.pdf:19, pone.0312962.pdf:14
**Answer:** (not generated)
**Failure:** retrieval miss

### real-0002 (factual)

**Question:** According to the modality-specific analysis by NBRS, which two models tied as the fairest models with an NBRS score of 0.30?
**Retrieved pages:** Ayush & Aman - BIAS IEEE PAPER.pdf:5, Ayush & Aman - BIAS IEEE PAPER.pdf:4, Ayush & Aman - BIAS IEEE PAPER.pdf:4, Ayush & Aman - BIAS IEEE PAPER.pdf:4, Ayush & Aman - BIAS IEEE PAPER.pdf:6, Ayush & Aman - BIAS IEEE PAPER.pdf:5, Ayush & Aman - BIAS IEEE PAPER.pdf:7, Ayush & Aman - BIAS IEEE PAPER.pdf:5
**Answer:** (not generated)
**Failure:** retrieval miss

