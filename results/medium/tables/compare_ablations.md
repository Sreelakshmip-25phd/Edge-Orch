# full vs ablations

Each cell: the system's mean over seeds, then in brackets its difference from full in the same unit (percentage points for rates; calls, tokens or ms for costs). Negative = lower than full (worse for quality columns, except locality violations; cheaper for cost columns). * = paired t-test vs full significant after Holm correction within this table (p < 0.05). Setup latency = request arrival to placement; service start-up is not included.

| system | accepted | completed | completed, correct type | escalation success | accepted, last third | locality violations | model calls/req | tokens/req | setup latency |
|---|---|---|---|---|---|---|---|---|---|
| full | 91.2% | 89.0% | 79.0% | 78.4% | 87.0% | 0.6% | 0.139 | 136 | 83 ms |
| no_memory | 90.4% (-0.8 pp) * | 87.9% (-1.0 pp) * | 78.8% (-0.2 pp) | 76.1% (-2.3 pp) * | 85.4% (-1.6 pp) * | 0.6% (+0.1 pp) | 0.427 (+0.288) * | 322 (+185) * | 222 ms (+139) * |
| no_intent_cache | 91.1% (-0.1 pp) | 88.6% (-0.4 pp) | 79.6% (+0.5 pp) | 78.0% (-0.4 pp) | 86.7% (-0.3 pp) | 0.5% (-0.1 pp) | 1.098 (+0.959) * | 634 (+498) * | 366 ms (+283) * |
| no_cache_sharing | 90.6% (-0.6 pp) | 88.2% (-0.7 pp) | 78.1% (-1.0 pp) | 77.3% (-1.1 pp) | 85.8% (-1.2 pp) | 0.6% (-0.0 pp) | 0.260 (+0.121) * | 197 (+61) * | 115 ms (+32) * |
| no_preempt_degrade | 80.3% (-10.9 pp) * | 79.5% (-9.5 pp) * | 70.7% (-8.4 pp) * | 36.0% (-42.4 pp) * | 71.6% (-15.4 pp) * | 0.3% (-0.3 pp) * | 0.040 (-0.099) * | 20 (-116) * | 24 ms (-60) * |
| no_zone_tier | 90.8% (-0.4 pp) | 88.5% (-0.4 pp) | 77.9% (-1.1 pp) | 90.8% (+12.4 pp) * | 87.0% (+0.0 pp) | 1.5% (+0.9 pp) * | 0.163 (+0.024) * | 159 (+22) * | 135 ms (+52) * |
| no_cross_zone | 86.5% (-4.7 pp) * | 82.2% (-6.8 pp) * | 74.1% (-4.9 pp) * | 51.5% (-26.9 pp) * | 81.4% (-5.5 pp) * | 0.0% (-0.6 pp) * | 0.090 (-0.049) * | 59 (-77) * | 55 ms (-28) * |
