# full vs baselines

Differences vs full: quality = system minus full in percentage points (negative = worse than full, except locality violations where negative = fewer); cost = system / full (above 1.00x = more expensive than full). * = paired t-test vs full significant after Holm correction within this table (p < 0.05).

| system | accepted | completed | escalation success | accepted, last third | locality violations | model calls/req | tokens/req | setup latency |
|---|---|---|---|---|---|---|---|---|
| greedy_oracle | -5.8 pp * | -4.4 pp * | +4.9 pp * | -9.0 pp * | +36.6 pp * | 0.00x | 0.00x | 0.07x |
| rule_based | -18.1 pp * | -16.5 pp * | -55.1 pp * | -29.3 pp * | -0.5 pp * | 0.00x | 0.00x | 0.19x |
| core | -0.3 pp | -0.8 pp * | -0.2 pp | -0.3 pp | -0.0 pp | 10.29x | 6.36x | 5.50x |
