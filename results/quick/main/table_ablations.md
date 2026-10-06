# Table 2 - what each mechanism contributes

**Experiment 2 - agentic baselines: 1,000 requests over a compressed 3 h day (compare numbers only within this experiment)** (profile `quick`).

Mean ± 95% CI over seeds (seeds per system: full 3, no_memory 3, no_intent_cache 3, no_cache_sharing 3, no_preempt_degrade 3, no_zone_tier 3, no_cross_zone 3, greedy_oracle 3, rule_based 3, core 3, react 3, agentedge 3, lats 1). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

First row: the full system's absolute values. Other rows: quality as the difference from full in percentage points (negative = worse); cost as a ratio to full (above 1.00x = the ablation needs more). no_zone_tier has no escalation success to compare: without zone agents every request counts as escalated, so the rate is over all requests rather than the hard ones.

| system | removes | accepted | escalation success | model calls / req | SLM calls / req | LLM calls / req | calls / req, last quarter of day | setup latency |
|---|---|---|---|---|---|---|---|---|
| **full (proposed)** | - | 92.0 % | 78.7 % | 0.24 | 0.13 | 0.12 | 0.27 | 126 ms |
| no_memory | similar-case memory, learned rules and the capacity digest | -0.0 pp | -0.1 pp | 2.14x * | 1.00x | 3.37x * | 2.45x * | 2.27x * |
| no_intent_cache | the semantic intent cache | +0.4 pp | +0.1 pp | 4.59x * | 7.92x * | 1.01x | 4.30x * | 3.14x * |
| no_cache_sharing | sharing the intent cache across zones | +0.1 pp | -0.5 pp | 2.18x * | 3.22x * | 1.05x | 2.16x * | 1.73x * |
| no_preempt_degrade | pre-emption and the degradation menu | -10.8 pp | -42.5 pp * | 0.53x * | 1.00x | 0.03x * | 0.45x * | 0.38x * |
| no_zone_tier | Tier-2 zone agents (local translation + local placement) | -0.1 pp | n/a (see note) | 1.19x * | 1.02x | 1.38x | 1.21x | 1.52x * |
| no_cross_zone | cross-zone escalation targets | -3.8 pp | -25.0 pp * | 0.90x * | 1.00x | 0.79x | 0.90x | 0.89x |
