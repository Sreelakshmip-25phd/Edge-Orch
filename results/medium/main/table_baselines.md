# Table 1 - full vs baselines

**Experiment 1 - main campaign: 6,000 requests over a 24 h day, 5 seeds** (profile `medium`).

Mean ± 95% CI over seeds (seeds per system: full 5, no_memory 5, no_intent_cache 5, no_cache_sharing 5, no_preempt_degrade 5, no_zone_tier 5, no_cross_zone 5, greedy_oracle 5, rule_based 5, core 5). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

"Completed, correct type" counts a request only if the service that ran to the end is the one asked for (a mistranslated request can still be accepted, as the wrong service). greedy_oracle is handed the true service type: its translation columns are n/a. Setup latency = transport + translation + decision + network hops; service start-up time is excluded (an assumption, not a measurement).

| system | accepted % | completed, correct type % | new types accepted % | locality violations % | service type correct % | model calls / req | tokens / req | setup latency ms |
|---|---|---|---|---|---|---|---|---|
| **full (proposed)** | 91.2 ± 1.4 | 79.0 ± 1.6 | 92.8 ± 4.4 | 0.6 ± 0.1 | 88.9 ± 1.6 | 0.14 ± 0.02 | 136 ± 19 | 83 ± 10 |
| greedy_oracle | 85.4 ± 3.3 * | n/a (given) | 85.7 ± 5.7 * | 37.1 ± 1.3 * | n/a (given) | 0.00 * | 0 * | n/a (given) |
| rule_based | 73.1 ± 2.3 * | 71.6 ± 1.9 | 13.9 ± 2.2 * | 0.0 ± 0.0 * | 84.0 ± 0.4 * | 0.00 * | 0 * | 16 ± 0 * |
| core | 90.9 ± 1.7 | 79.1 ± 1.9 | 96.2 ± 1.9 | 0.5 ± 0.2 | 89.8 ± 0.7 | 1.43 ± 0.04 * | 868 ± 34 * | 457 ± 19 * |
