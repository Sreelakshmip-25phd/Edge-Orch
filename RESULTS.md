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

## 2. Stage 3 quick run, Experiment 2's first part (3 seeds)

full, the 6 ablations, greedy_oracle, rule_based and core on quick seeds 0–2
(30 runs, about 2 h on the GPU). Tables: `results/quick/main/` on the GPU
machine. full: 92.0% accepted, 0.24 model calls/request (pilot 0.57),
193 tokens/request (pilot 404), 126 ms setup latency (pilot 216 ms), cache
misses about 10% (pilot about 38%). Removing the cache costs 4.6× calls,
removing memory + digest 2.1×, removing cache sharing 2.2×, removing the
zone tier 1.19× (in the pilot that ablation was cheaper); acceptance stays
within ±0.5 pp in all four. CORE ties on acceptance (92.6%) at 5.8× the calls.

ReAct on quick seed 0 with that code reached only 25.1% acceptance at 6.4
calls/request: a prompt-layout bug of ours made it re-read the same zone
until its steps ran out (ARCHITECTURE.md §8). That run and the LATS run that
shared the layout are discarded; AgentEdge is unaffected.

## 3. Experiment 1: `medium` (6,000 requests over a 24 h day, 5 seeds)

Run on the GPU machine (K = 96, greedy acceptance 0.857), latency
re-measured after the choice-by-id prompt. 50 runs, about 7.5 h.
Mean over 5 seeds; * = paired t-test vs full, Holm-corrected within the table.

full: 91.2% accepted, 78.4% escalation success, 92.8% of new-type requests
accepted, 0.6% locality violations, 88.9% service type correct, 0.14 model
calls/request, 136 tokens/request, 83 ms setup latency; intent-cache misses
2.2% in the first quarter of the day, 1.3% in the last.

| ablation | accepted | escalation success | model calls | setup latency |
|---|---|---|---|---|
| no_intent_cache | −0.1 pp | −0.4 pp | 7.90× * | 4.40× * |
| no_memory (memory, rules, digest) | −0.8 pp * | −2.3 pp * | 3.07× * | 2.67× * |
| no_cache_sharing | −0.6 pp | −1.1 pp | 1.87× * | 1.39× * |
| no_zone_tier | −0.4 pp | n/a (all requests escalate) | 1.17× * | 1.62× * |
| no_cross_zone | −4.7 pp * | −26.9 pp * | 0.65× * | 0.66× * |
| no_preempt_degrade | −10.9 pp * | −42.4 pp * | 0.29× * | 0.28× * |

| baseline | accepted | new types | locality violations | model calls / req | tokens / req | setup latency |
|---|---|---|---|---|---|---|
| **full** | **91.2%** | 92.8% | 0.6% | **0.14** | **136** | **83 ms** |
| core | 90.9% | 96.2% | 0.5% | 1.43 * (10.2×) | 868 * | 457 ms * |
| greedy_oracle | 85.4% * | 85.7% * | 37.1% * | 0 | 0 | n/a (given type) |
| rule_based | 73.1% * | 13.9% * | 0.0% * | 0 | 0 | 16 ms * |

Four mechanisms cut model calls without costing acceptance (cache, memory,
sharing, zone tier); two protect quality (cross-zone placement,
pre-emption/degradation). CORE matches full's quality at 10× the calls. On
the new service types CORE is ahead (96.2% vs 92.8%, not significant; the
same direction in quick): it translates every request afresh, while full's
cache can match a new type to a similar known one.

## 4. Experiment 2: agentic baselines on `quick` (1,000 requests, 3 seeds; LATS 1 seed)

| system | accepted | new types | locality violations | service type correct | model calls / req | tokens / req | setup latency |
|---|---|---|---|---|---|---|---|
| **full** | **92.0%** | **96.9%** | 1.5% | 88.4% | **0.24** | **193** | **126 ms** |
| lats (seed 0) | 85.0% | 69.5% | 10.4% | 76.2% | 24.27 | 46,541 | 11,421 ms |
| agentedge | 75.8% | 66.4% * | 0.0% * | 88.5% | 6.11 * | 4,198 * | 2,787 ms * |
| react | 73.2% * | 64.3% | 8.7% * | 61.8% * | 4.11 * | 7,783 * | 2,584 ms * |

LATS ran on one seed only: one quick seed takes 9.2 h on the RTX 4070 SUPER,
against about 2.5 min for full. Seeds 1–2 can be added later; the LLM answers
of the interrupted seed 1 are cached.

## 5. Model comparison (8 local models; quick, 3 seeds)

Per model: the probe (158 held-out translations, 100 auto-scored decision
cases with the choices shuffled), then the full system on the quick
workloads with that one model in both roles. Every probe call was fresh
and every row reports its own served model file.

| model | size | valid JSON | decisions acceptable / best choice | translation exact | full: accepted | full: completed, correct type | new types accepted | calls / req | setup latency |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | 17% | 8% / 5% | 52% | 86.7% | 60.1% | 67.8% | 0.50 | 211 ms |
| Qwen2.5-1.5B | 1.5B | 100% | 92% / 49% | 71% | 91.9% | 83.0% | 93.3% | 0.25 | 79 ms |
| Llama-3.2-3B | 3B | 100% | 100% / 59% | 71% | 91.5% | 78.6% | 96.4% | 0.24 | 84 ms |
| Phi-3.5-mini | 3.8B | 100% | 100% / 74% | 89% | 91.7% | 83.6% | 85.3% | 0.26 | 138 ms |
| Qwen2.5-7B | 7B | 100% | 100% / 75% | 75% | 89.1% | 76.7% | 76.4% | 0.28 | 152 ms |
| Mistral-7B | 7B | 100% | 98% / 73% | 77% | 91.2% | 83.2% | 87.6% | 0.26 | 175 ms |
| Gemma-2-9B | 9B | 100% | 100% / 80% | 91% | 91.5% | 84.4% | 80.8% | 0.25 | 187 ms |
| Qwen2.5-14B | 14B | 100% | 100% / 81% | 94% | 91.6% | 85.6% | 80.6% | 0.24 | 206 ms |
| main setup: Llama-3.2-3B (SLM) + Qwen2.5-7B (LLM) | 3B + 7B | | | | 92.0% | 78.9% | 96.9% | 0.24 | 126 ms |

"Completed, correct type" counts a request only if the service that ran to
the end is the one asked for: a mistranslated request can still be accepted,
as the wrong service.

- Phi-3.5 and Gemma-2 answer in valid JSON 100% of the time (system prompt
  folded into the user turn where the chat template drops it, plus JSON
  extraction from surrounding text).
- From 1.5B upwards the full system accepts 89–92% at about 0.25 model
  calls/request: the cache and memory keep calls low whichever model is
  used. Llama-3.2-1B is too small (17% valid decision JSON): the system
  falls back more often and needs twice the calls.
- Acceptance alone hides translation quality. Models that translate better
  (Phi-3.5, Mistral, Gemma, Qwen-14B: 89–94% exact) complete 83–86% of
  requests as the right service; the main setup's Llama-3.2-3B translator
  reaches 79%. The main setup is the cheapest and fastest among the
  well-performing ones (0.24 calls, 126 ms), not the most accurate; a
  stronger translator would raise correct-type completion by about 5 points
  at about 1.5–2.5× the setup latency.
- New service types: the larger models recognise them more often (45.8%
  of new-type requests typed correctly vs 26.8% for Llama-3.2-3B; some
  labelled as a new type), and a correctly typed `crowd_safety` request
  carries its real demand (critical, zone-local, 1.5× CPU, accelerator), so
  fewer are placed: 76–81% accepted vs 93–97% when they are mistaken for a
  smaller known type. The same effect explains CORE's and no_intent_cache's
  edge on new types in medium (both type 54% of them correctly, full 35%:
  the cache maps some new phrasings to a similar known type).



## 6. Optional

- LATS seeds 1–2 on quick (about 18 h).
- `medium` seeds 5–9 (about 7.5 h), for 10 seeds in Experiment 1.

## 7. Smoke run (mock LLM; NOT results)

`python main.py --profile smoke` runs every system end to end on synthetic
data with the mock LLM and placeholder latencies, in a few minutes. It shows
that the pipeline, metrics, tests, tables and figures work; its numbers say
nothing about real model behaviour.
