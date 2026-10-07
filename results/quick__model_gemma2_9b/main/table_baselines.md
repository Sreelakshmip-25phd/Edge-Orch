# Table 1 - full vs baselines

**Experiment 2 - agentic baselines: 1,000 requests over a compressed 3 h day (compare numbers only within this experiment)** (profile `quick`).

Mean ± 95% CI over seeds (seeds per system: full 3). * = differs from full, paired t-test, Holm-corrected within this table, p < 0.05.

greedy_oracle is handed the true service type: its translation columns are n/a. Setup latency = transport + translation + decision + network hops; service start-up time is excluded (an assumption, not a measurement).

| system | accepted % | new types accepted % | locality violations % | service type correct % | model calls / req | tokens / req | setup latency ms |
|---|---|---|---|---|---|---|---|
| **full (proposed)** | 91.5 ± 1.7 | 80.8 ± 17.5 | 0.2 ± 0.8 | 94.8 ± 3.4 | 0.25 ± 0.07 | 201 ± 71 | 187 ± 22 |
