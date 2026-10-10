# full vs ablations

Each cell: the system's mean over seeds, then in brackets its difference from full in the same unit (percentage points for rates; calls, tokens or ms for costs). Negative = lower than full (worse for quality columns, except locality violations; cheaper for cost columns). * = paired t-test vs full significant after Holm correction within this table (p < 0.05). Setup latency = request arrival to placement; service start-up is not included.

| system | accepted | completed | completed, correct type | escalation success | accepted, last third | locality violations | model calls/req | tokens/req | setup latency |
|---|---|---|---|---|---|---|---|---|---|
| full | 92.0% | 88.7% | 78.9% | 78.7% | 82.0% | 1.5% | 0.244 | 193 | 126 ms |
| no_memory | 92.0% (-0.0 pp) | 89.3% (+0.6 pp) | 79.3% (+0.4 pp) | 78.6% (-0.1 pp) | 82.0% (+0.0 pp) | 1.3% (-0.2 pp) | 0.522 (+0.278) * | 378 (+185) * | 286 ms (+160) * |
| no_intent_cache | 92.4% (+0.4 pp) | 89.1% (+0.4 pp) | 79.4% (+0.5 pp) | 78.8% (+0.1 pp) | 82.9% (+0.9 pp) | 0.7% (-0.8 pp) | 1.119 (+0.875) * | 648 (+455) * | 395 ms (+269) * |
| no_cache_sharing | 92.1% (+0.1 pp) | 88.8% (+0.1 pp) | 79.2% (+0.2 pp) | 78.2% (-0.5 pp) | 82.2% (+0.2 pp) | 0.8% (-0.7 pp) | 0.531 (+0.287) * | 345 (+152) * | 218 ms (+92) * |
| no_preempt_degrade | 81.2% (-10.8 pp) | 81.0% (-7.7 pp) | 71.1% (-7.9 pp) | 36.2% (-42.5 pp) * | 62.6% (-19.4 pp) | 0.9% (-0.6 pp) | 0.130 (-0.114) * | 66 (-127) * | 48 ms (-78) * |
| no_zone_tier | 91.9% (-0.1 pp) | 88.3% (-0.4 pp) | 79.0% (+0.0 pp) | 91.9% (+13.3 pp) * | 81.4% (-0.6 pp) | 1.8% (+0.3 pp) | 0.291 (+0.047) * | 248 (+55) * | 191 ms (+65) * |
| no_cross_zone | 88.2% (-3.8 pp) | 81.8% (-6.9 pp) | 73.0% (-5.9 pp) | 53.7% (-25.0 pp) * | 79.0% (-3.0 pp) | 0.0% (-1.5 pp) * | 0.220 (-0.024) * | 138 (-55) * | 112 ms (-14) |
