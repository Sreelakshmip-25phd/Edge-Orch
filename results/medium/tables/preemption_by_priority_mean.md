# Pre-emptions per day, by priority

Mean over seeds. by:<p> = pre-emptions made for a request of priority p; victim:<p> = services of priority p that were evicted, and whether they were moved elsewhere (migrated) or lost.

| system | split | preemptions | victim_migrated | victim_lost |
|---|---|---|---|---|
| full | by:critical | 179.6 | 147.8 | 39.0 |
| full | by:high | 656.4 | 611.8 | 72.4 |
| full | victim:high | 39.6 | 11.4 | 30.6 |
| full | victim:low | 131.2 | 131.0 | 0.2 |
| full | victim:normal | 665.2 | 617.2 | 80.6 |
| no_memory | by:critical | 158.8 | 130.2 | 38.6 |
| no_memory | by:high | 546.4 | 494.8 | 98.2 |
| no_memory | victim:high | 61.2 | 37.0 | 30.4 |
| no_memory | victim:low | 89.6 | 89.4 | 0.2 |
| no_memory | victim:normal | 554.4 | 498.6 | 106.2 |
| no_intent_cache | by:critical | 223.6 | 174.8 | 55.0 |
| no_intent_cache | by:high | 634.4 | 588.4 | 76.4 |
| no_intent_cache | victim:high | 53.4 | 11.0 | 44.0 |
| no_intent_cache | victim:low | 128.4 | 127.8 | 1.0 |
| no_intent_cache | victim:normal | 676.2 | 624.4 | 86.4 |
| no_cache_sharing | by:critical | 183.6 | 146.4 | 41.8 |
| no_cache_sharing | by:high | 662.6 | 612.2 | 78.2 |
| no_cache_sharing | victim:high | 43.2 | 11.0 | 33.4 |
| no_cache_sharing | victim:low | 132.2 | 131.8 | 0.4 |
| no_cache_sharing | victim:normal | 670.8 | 615.8 | 86.2 |
| no_zone_tier | by:critical | 260.2 | 224.8 | 41.4 |
| no_zone_tier | by:high | 893.8 | 846.8 | 82.6 |
| no_zone_tier | victim:high | 43.6 | 14.2 | 31.0 |
| no_zone_tier | victim:low | 231.2 | 231.2 | 0.4 |
| no_zone_tier | victim:normal | 879.2 | 826.2 | 92.6 |
| no_cross_zone | by:critical | 104.0 | 52.0 | 56.8 |
| no_cross_zone | by:high | 332.6 | 173.2 | 175.4 |
| no_cross_zone | victim:high | 39.8 | 11.0 | 30.4 |
| no_cross_zone | victim:low | 71.8 | 66.8 | 5.8 |
| no_cross_zone | victim:normal | 325.0 | 147.4 | 196.0 |
| core | by:critical | 209.0 | 159.8 | 59.2 |
| core | by:high | 592.6 | 539.0 | 92.4 |
| core | victim:high | 84.4 | 42.2 | 49.4 |
| core | victim:low | 96.8 | 96.8 | 0.0 |
| core | victim:normal | 620.4 | 559.8 | 102.2 |
