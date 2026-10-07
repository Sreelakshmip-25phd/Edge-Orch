# Table 2 - what each mechanism contributes

**Experiment 2 - agentic baselines: 1,000 requests over a compressed 3 h day (compare numbers only within this experiment)** (profile `quick`).

Mean ± 95% CI over seeds (seeds per system: full 3). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

First row: the full system's absolute values. Other rows: quality as the difference from full in percentage points (negative = worse); cost as a ratio to full (above 1.00x = the ablation needs more). no_zone_tier has no escalation success to compare: without zone agents every request counts as escalated, so the rate is over all requests rather than the hard ones.

| system | removes | accepted | escalation success | model calls / req | SLM calls / req | LLM calls / req | calls / req, last quarter of day | setup latency |
|---|---|---|---|---|---|---|---|---|
| **full (proposed)** | - | 91.5 % | 77.0 % | 0.24 | 0.13 | 0.12 | 0.25 | 84 ms |
