# full vs ablations

Differences vs full: quality = system minus full in percentage points (negative = worse than full, except locality violations where negative = fewer); cost = system / full (above 1.00x = more expensive than full). * = paired t-test vs full significant after Holm correction within this table (p < 0.05).

| system | accepted | completed | escalation success | accepted, last third | locality violations | model calls/req | tokens/req | setup latency |
|---|---|---|---|---|---|---|---|---|
| no_memory | -0.0 pp | +0.6 pp | -0.1 pp | +0.0 pp | -0.2 pp | 2.14x | 1.96x | 2.27x |
| no_intent_cache | +0.4 pp | +0.4 pp | +0.1 pp | +0.9 pp | -0.8 pp | 4.59x | 3.36x | 3.14x |
| no_cache_sharing | +0.1 pp | +0.1 pp | -0.5 pp | +0.2 pp | -0.7 pp | 2.18x | 1.79x | 1.73x |
| no_preempt_degrade | -10.8 pp | -7.7 pp | -42.5 pp * | -19.4 pp | -0.6 pp | 0.53x | 0.34x | 0.38x |
| no_zone_tier | -0.1 pp | -0.4 pp | +13.3 pp * | -0.6 pp | +0.3 pp | 1.19x | 1.28x | 1.52x |
| no_cross_zone | -3.8 pp | -6.9 pp | -25.0 pp * | -3.0 pp | -1.5 pp * | 0.90x | 0.72x | 0.89x |
