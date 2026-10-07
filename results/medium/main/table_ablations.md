# Table 2 - what each mechanism contributes

**Experiment 1 - main campaign: 6,000 requests over a 24 h day, 5 seeds** (profile `medium`).

Mean ± 95% CI over seeds (seeds per system: full 5, no_memory 5, no_intent_cache 5, no_cache_sharing 5, no_preempt_degrade 5, no_zone_tier 5, no_cross_zone 5, greedy_oracle 5, rule_based 5, core 5). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

First row: the full system's absolute values. Other rows: quality as the difference from full in percentage points (negative = worse); cost as a ratio to full (above 1.00x = the ablation needs more). no_zone_tier has no escalation success to compare: without zone agents every request counts as escalated, so the rate is over all requests rather than the hard ones.

| system | removes | accepted | escalation success | model calls / req | SLM calls / req | LLM calls / req | calls / req, last quarter of day | setup latency |
|---|---|---|---|---|---|---|---|---|
| **full (proposed)** | - | 91.2 % | 78.4 % | 0.14 | 0.04 | 0.10 | 0.15 | 83 ms |
| no_memory | similar-case memory, learned rules and the capacity digest | -0.8 pp * | -2.3 pp * | 3.07x * | 1.00x | 3.88x * | 3.46x * | 2.67x * |
| no_intent_cache | the semantic intent cache | -0.1 pp | -0.4 pp | 7.90x * | 25.64x * | 0.98x | 7.62x * | 4.40x * |
| no_cache_sharing | sharing the intent cache across zones | -0.6 pp | -1.1 pp | 1.87x * | 4.16x * | 0.98x | 1.76x * | 1.39x * |
| no_preempt_degrade | pre-emption and the degradation menu | -10.9 pp * | -42.4 pp * | 0.29x * | 1.00x | 0.01x * | 0.22x * | 0.28x * |
| no_zone_tier | Tier-2 zone agents (local translation + local placement) | -0.4 pp | n/a (see note) | 1.17x * | 1.03x | 1.23x * | 1.23x * | 1.62x * |
| no_cross_zone | cross-zone escalation targets | -4.7 pp * | -26.9 pp * | 0.65x * | 1.00x | 0.51x * | 0.52x * | 0.66x * |
