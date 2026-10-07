# Table 1 - full vs baselines

**Experiment 2 - agentic baselines: 1,000 requests over a compressed 3 h day (compare numbers only within this experiment)** (profile `quick`).

Mean ± 95% CI over seeds (seeds per system: full 3). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

greedy_oracle is handed the true service type: its translation columns are n/a. Setup latency = transport + translation + decision + network hops; service start-up time is excluded (an assumption, not a measurement).

| system | accepted % | new types accepted % | locality violations % | service type correct % | model calls / req | tokens / req | setup latency ms |
|---|---|---|---|---|---|---|---|
| **full (proposed)** | 91.5 ± 1.8 | 96.4 ± 8.6 | 1.3 ± 1.3 | 88.6 ± 1.4 | 0.24 ± 0.05 | 186 ± 52 | 84 ± 12 |
