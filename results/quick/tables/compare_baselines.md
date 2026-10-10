# full vs baselines

Each cell: the system's mean over seeds, then in brackets its difference from full in the same unit (percentage points for rates; calls, tokens or ms for costs). Negative = lower than full (worse for quality columns, except locality violations; cheaper for cost columns). * = paired t-test vs full significant after Holm correction within this table (p < 0.05). Setup latency = request arrival to placement; service start-up is not included. n/a (given type): greedy_oracle is handed the true service type, so it has no translation to compare; n/a (not comparable): without zone agents every request counts as escalated.

| system | accepted | completed | completed, correct type | escalation success | accepted, last third | locality violations | model calls/req | tokens/req | setup latency |
|---|---|---|---|---|---|---|---|---|---|
| full | 92.0% | 88.7% | 78.9% | 78.7% | 82.0% | 1.5% | 0.244 | 193 | 126 ms |
| greedy_oracle | 85.9% (-6.1 pp) | 85.8% (-2.9 pp) | n/a (given type) | 83.9% (+5.2 pp) | 67.8% (-14.2 pp) | 36.5% (+35.0 pp) * | 0.000 (-0.244) * | 0 (-193) * | n/a (given type) |
| rule_based | 73.8% (-18.2 pp) * | 73.7% (-15.0 pp) * | 73.2% (-5.7 pp) | 19.8% (-58.9 pp) * | 48.7% (-33.3 pp) * | 0.0% (-1.4 pp) * | 0.000 (-0.244) * | 0 (-193) * | 14 ms (-111) * |
| core | 92.6% (+0.6 pp) | 88.8% (+0.1 pp) | 79.0% (+0.0 pp) | 80.2% (+1.5 pp) | 83.4% (+1.4 pp) | 1.0% (-0.5 pp) | 1.402 (+1.158) * | 840 (+647) * | 478 ms (+353) * |
| react | 73.2% (-18.8 pp) * | 71.7% (-17.0 pp) * | 60.3% (-18.6 pp) * | 36.6% (-42.1 pp) * | 58.2% (-23.8 pp) | 8.7% (+7.2 pp) * | 4.109 (+3.865) * | 7,783 (+7,590) * | 2,584 ms (+2,458) * |
| agentedge | 75.8% (-16.2 pp) | 67.0% (-21.7 pp) * | 63.1% (-15.8 pp) | 69.6% (-9.1 pp) | 61.6% (-20.4 pp) | 0.0% (-1.5 pp) * | 6.108 (+5.864) * | 4,198 (+4,005) * | 2,787 ms (+2,661) * |
| lats | 85.0% (-7.0 pp) | 84.4% (-4.3 pp) | 75.6% (-3.3 pp) | 62.4% (-16.3 pp) | 68.6% (-13.4 pp) | 10.4% (+8.9 pp) | 24.275 (+24.031) | 46,541 (+46,348) | 11,421 ms (+11,295) |
