# Table 1 - full vs baselines

**Experiment 2 - agentic baselines: 1,000 requests over a compressed 3 h day (compare numbers only within this experiment)** (profile `quick`).

Mean ± 95% CI over seeds (seeds per system: full 3, no_memory 3, no_intent_cache 3, no_cache_sharing 3, no_preempt_degrade 3, no_zone_tier 3, no_cross_zone 3, greedy_oracle 3, rule_based 3, core 3, react 3, agentedge 3, lats 1). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

greedy_oracle is handed the true service type: its translation columns are n/a. Setup latency = transport + translation + decision + network hops; service start-up time is excluded (an assumption, not a measurement).

| system | accepted % | new types accepted % | locality violations % | service type correct % | model calls / req | tokens / req | setup latency ms |
|---|---|---|---|---|---|---|---|
| **full (proposed)** | 92.0 ± 1.5 | 96.9 ± 6.4 | 1.5 ± 0.4 | 88.4 ± 1.1 | 0.24 ± 0.03 | 193 ± 28 | 126 ± 14 |
| greedy_oracle | 85.9 ± 5.3 | 84.0 ± 17.7 | 36.5 ± 4.0 * | n/a (given) | 0.00 * | 0 * | n/a (given) |
| rule_based | 73.8 ± 4.6 * | 6.9 ± 9.0 * | 0.0 ± 0.2 * | 82.9 ± 0.7 * | 0.00 * | 0 * | 14 ± 4 * |
| core | 92.6 ± 1.3 | 98.5 ± 3.8 | 1.0 ± 0.5 | 88.9 ± 2.2 | 1.40 ± 0.12 * | 840 ± 118 * | 478 ± 64 * |
| react | 73.2 ± 5.4 * | 64.3 ± 22.8 | 8.7 ± 2.8 * | 61.8 ± 3.6 * | 4.11 ± 0.24 * | 7783 ± 570 * | 2584 ± 49 * |
| agentedge | 75.8 ± 8.0 | 66.4 ± 13.2 * | 0.0 * | 88.5 ± 4.3 | 6.11 ± 0.35 * | 4198 ± 342 * | 2787 ± 53 * |
| lats | 85.0 | 69.5 | 10.4 | 76.2 | 24.27 | 46541 | 11421 |
