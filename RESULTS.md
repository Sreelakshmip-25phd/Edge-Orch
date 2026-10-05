# Results so far

## 1. Quick pilot on the GPU machine (superseded design)

Profile `quick`: 1,000 requests over a compressed 3-hour day, 10 zones,
3 seeds (ReAct, LATS and AgentEdge: seed 0 only). Models: Llama-3.2-3B
(SLM) and Qwen2.5-7B (LLM) on an RTX 4070 SUPER, with calibrated latencies.
The raw outputs are on the `results-quick` branch.

This pilot ran the design *before* the stage-3 changes (ARCHITECTURE.md §5–8):
per-zone caches, the `allowed_actions` decision prompt, ReAct with 6 steps and
the old tool text, and LATS without re-expansion. Its numbers motivated those
changes and are not the paper's results.

| | accepted | model calls / req | setup latency |
|---|---|---|---|
| full | 92.2% | 0.57 | 216 ms |
| core | 92.4% (tie) | 1.44 (2.5×) | 474 ms |
| greedy_oracle | 85.9%, but 36.5% locality violations | 0 | n/a (given type) |
| rule_based | 73.8% | 0 | 14 ms |
| agentedge (1 seed) | 74.4% | 6.05 | 2,755 ms |
| lats (1 seed) | 65.7% | 16.97 | 11,043 ms |
| react (1 seed) | 31.3% | 4.90 | 3,605 ms |

What the pilot showed and what was changed:

- **Cache and memory cut calls with no loss of acceptance.** Without the cache
  the system needed 2.05× the calls; without memory and digest, 1.41×, with
  savings growing from 3% to 34% over the day. Digest-only and memory-only
  ablations had no significant effect alone, so they were merged into one
  `no_memory` ablation.
- **Per-zone caches missed what other zones had learned.** 83–85% of zone
  cache misses were phrases another zone had already translated. The zones
  now share one cache, and `no_cache_sharing` measures the effect.
- **The LLM failed to copy fields back.** 12–19 decisions per seed failed
  verification twice, mostly degrade answers with `zone: null`. The LLM now
  answers with the id of a pre-verified choice.
- **ReAct spent its steps reading.** In 519 of its 687 rejections it never
  tried a placement. **LATS gave up** once the root's first children failed.
  The shared tool text, the running-services list, the step budget (7) and
  LATS re-expansion were fixed (ARCHITECTURE.md §8).

## 2. Still to run (stage 3)

1. Pull, and delete the pilot's quick outputs and caches: the prompts
   changed, so cached LLM answers no longer match.
2. `python scripts/run_parallel.py --workers 1 --profile quick --systems proposed,ablations,simple`,
   then `--systems agentic`.
3. Check `results/quick/main/` (2 tables, 4 figures); then the full campaign
   (`--profile full`, 10 seeds) and `python scripts/paper_results.py --profile full`.
4. `python scripts/run_model_sweep.py` for the model comparison.

The decision prompt changed after the latency calibration (choices by id).
The calibrated `decide` latency was measured with the earlier prompt; it is
similar in length, but re-running `scripts/calibrate_latency.py` before the
full campaign removes the doubt.

## 3. Smoke run (mock LLM; NOT results)

`python main.py --profile smoke` runs every system end to end on synthetic
data with the mock LLM and placeholder latencies, in a few minutes. It shows
that the pipeline, metrics, tests, tables and figures work; its numbers say
nothing about real model behaviour.
