# full vs ablations

Differences vs full: quality = system minus full in percentage points (negative = worse than full, except locality violations where negative = fewer); cost = system / full (above 1.00x = more expensive than full). * = paired t-test vs full significant after Holm correction within this table (p < 0.05).

| system | accepted | completed | escalation success | accepted, last third | locality violations | model calls/req | tokens/req | setup latency |
|---|---|---|---|---|---|---|---|---|
| no_memory | -0.8 pp * | -1.0 pp * | -2.3 pp * | -1.6 pp * | +0.1 pp | 3.07x | 2.36x | 2.67x |
| no_intent_cache | -0.1 pp | -0.4 pp | -0.4 pp | -0.3 pp | -0.1 pp | 7.90x | 4.65x | 4.40x |
| no_cache_sharing | -0.6 pp | -0.7 pp | -1.1 pp | -1.2 pp | -0.0 pp | 1.87x | 1.45x | 1.39x |
| no_preempt_degrade | -10.9 pp * | -9.5 pp * | -42.4 pp * | -15.4 pp * | -0.3 pp * | 0.29x | 0.15x | 0.28x |
| no_zone_tier | -0.4 pp | -0.4 pp | +12.4 pp * | +0.0 pp | +0.9 pp * | 1.17x | 1.16x | 1.62x |
| no_cross_zone | -4.7 pp * | -6.8 pp * | -26.9 pp * | -5.5 pp * | -0.6 pp * | 0.65x | 0.43x | 0.66x |
