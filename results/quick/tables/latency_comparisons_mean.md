# Latency with vs without each fast path

Mean over seeds of the per-run means and 95th percentiles; requests = per run. Setup latency = request arrival to placement (service start-up not included).

| system | comparison | A | A mean ms | A p95 ms | A requests | B | B mean ms | B p95 ms | B requests |
|---|---|---|---|---|---|---|---|---|---|
| full | decision | memory | 0.1 | 0.1 | 179 | LLM | 716.8 | 796.4 | 87 |
| full | setup latency | same zone | 113.4 | 783.3 | 785 | cross-zone | 197.1 | 822.9 | 135 |
| full | translation | cache hit | 7.5 | 9.5 | 906 | SLM call | 309.0 | 337.0 | 94 |
| no_memory | decision | memory | n/a | n/a | 0 | LLM | 713.0 | 791.0 | 293 |
| no_memory | setup latency | same zone | 203.8 | 836.7 | 797 | cross-zone | 819.5 | 1083.5 | 122 |
| no_memory | translation | cache hit | 7.5 | 9.5 | 906 | SLM call | 310.1 | 337.0 | 94 |
| no_intent_cache | decision | memory | 0.1 | 0.2 | 165 | LLM | 714.5 | 790.4 | 92 |
| no_intent_cache | setup latency | same zone | 386.1 | 1078.4 | 807 | cross-zone | 453.1 | 1105.4 | 117 |
| no_intent_cache | translation | cache hit | n/a | n/a | 0 | SLM call | 302.2 | 330.7 | 1000 |
| no_cache_sharing | decision | memory | 0.1 | 0.2 | 164 | LLM | 708.8 | 792.1 | 94 |
| no_cache_sharing | setup latency | same zone | 205.0 | 885.6 | 796 | cross-zone | 297.2 | 923.5 | 125 |
| no_cache_sharing | translation | cache hit | 7.5 | 9.5 | 617 | SLM call | 310.6 | 339.1 | 383 |
| no_preempt_degrade | decision | memory | 0.1 | 0.1 | 84 | LLM | n/a | n/a | 0 |
| no_preempt_degrade | setup latency | same zone | 37.5 | 313.5 | 705 | cross-zone | 115.7 | 386.4 | 107 |
| no_preempt_degrade | translation | cache hit | 7.5 | 9.5 | 906 | SLM call | 311.7 | 338.7 | 94 |
| no_zone_tier | decision | memory | 0.1 | 0.2 | 705 | LLM | 712.8 | 796.4 | 120 |
| no_zone_tier | setup latency | same zone | 240.0 | 840.9 | 483 | cross-zone | 137.3 | 609.2 | 437 |
| no_zone_tier | translation | cache hit | 7.4 | 9.5 | 907 | SLM call | 308.3 | 334.1 | 93 |
| no_cross_zone | decision | memory | 0.0 | 0.1 | 56 | LLM | 707.8 | 784.8 | 81 |
| no_cross_zone | setup latency | same zone | 111.7 | 775.4 | 882 | cross-zone | n/a | n/a | 0 |
| no_cross_zone | translation | cache hit | 7.4 | 9.4 | 906 | SLM call | 312.4 | 337.5 | 94 |
| greedy_oracle | decision | memory | n/a | n/a | 0 | LLM | n/a | n/a | 0 |
| greedy_oracle | setup latency | same zone | 1.0 | 1.1 | 123 | cross-zone | 6.6 | 8.5 | 735 |
| greedy_oracle | translation | cache hit | n/a | n/a | 0 | SLM call | n/a | n/a | 0 |
| rule_based | decision | memory | n/a | n/a | 0 | LLM | n/a | n/a | 0 |
| rule_based | setup latency | same zone | 8.4 | 10.5 | 673 | cross-zone | 74.7 | 87.3 | 65 |
| rule_based | translation | cache hit | n/a | n/a | 0 | SLM call | n/a | n/a | 0 |
| core | decision | memory | n/a | n/a | 0 | LLM | 713.1 | 798.0 | 190 |
| core | setup latency | same zone | 467.9 | 1121.5 | 791 | cross-zone | 541.3 | 1121.8 | 135 |
| core | translation | cache hit | n/a | n/a | 0 | SLM call | 302.4 | 331.5 | 1000 |
| react | decision | memory | n/a | n/a | 0 | LLM | 3475.5 | 6678.6 | 1000 |
| react | setup latency | same zone | 2046.3 | 3201.7 | 590 | cross-zone | 4820.9 | 6711.3 | 142 |
| react | translation | cache hit | n/a | n/a | 0 | SLM call | n/a | n/a | 0 |
| agentedge | decision | memory | n/a | n/a | 0 | LLM | 2859.9 | 5087.1 | 1000 |
| agentedge | setup latency | same zone | 2831.0 | 3775.1 | 456 | cross-zone | 2719.4 | 2866.4 | 301 |
| agentedge | translation | cache hit | n/a | n/a | 0 | SLM call | n/a | n/a | 0 |
| lats | decision | memory | n/a | n/a | 0 | LLM | 16368.8 | 47749.6 | 1000 |
| lats | setup latency | same zone | 8050.2 | 17108.9 | 639 | cross-zone | 21629.8 | 43668.6 | 211 |
| lats | translation | cache hit | n/a | n/a | 0 | SLM call | n/a | n/a | 0 |
