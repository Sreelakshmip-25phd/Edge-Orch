# Architecture

This document explains how the system is built, what each ablation and
baseline isolates, how every reported number is produced, and every place
where a "real" number had to be approximated.

## 1. The three tiers

```
 Tier 1  edge device ──"run video analytics on the stadium cameras, under 100 ms"──►
         (pure NL text only; the receiving zone is decided by where the device is)

 Tier 2  ZoneAgent[z]  (one instance per zone)
           translate:  intent cache (MiniLM cosine ≥ θ) ──miss──► SLM ──► resolve vs live catalog
           place:      deterministic solver on the zone's live node table
           └─ doesn't fit ──► escalate

 Tier 3  GlobalAgent
           1 similar-case memory ─► 2 procedural rules ─► 3 digest scan
           ─► 4 ONE structured LLM decision {place|preempt|degrade|reject}
                verified by the capacity solver, one retry with the error as feedback
           ─► 5 deterministic rule chain (place ─► pre-empt ─► degrade ─► reject)
```

| piece | file | notes |
|---|---|---|
| Tier 1 | `src/edge_device.py` | `EdgeDevice.send()` returns an `IntentRequest(req_id, zone_id, text, t)`. That is the whole payload: no device specs, no IDs used for decisions, no resource hints. |
| Tier 2 | `src/zone_agent.py` | Each `ZoneAgent` owns its own `IntentLibrary` (LRU-capped, starts empty), its own `MultiLLM(role="slm")` handle and counters, and its own placement-policy weights. |
| Tier 3 | `src/global_agent.py` | Keeps only zone *digests* (≤ 5 s old, discarded after 15 s). It reads live capacity only through counted *probes*, which ask a zone's solver and cost a zone↔global round trip. |
| wiring | `src/orchestrator.py` | `HierarchicalOrchestrator`; the ablations are the same class with flags turned off. |
| physics | `src/sim_engine.py` | Discrete-event simulator. It holds the hidden ground truth and scores translation correctness *after* each decision. Agents never see the truth. |

The zone→node resource request comes from a **service manifest**. Once a
request's service type is resolved, the deployment artifact for that type is
fetched and its container resource request is known at placement time
(`sim_engine.Manifests`):

- resolved type == true type: the request's own size (per-request draw from
  the Alibaba distribution);
- wrong registered type: that type's median size (the wrong service is
  deployed, which is a real cost of mistranslation);
- unregistered/novel type: a generic median manifest.

This keeps the user's request pure NL while still giving realistic,
per-request resource demand.

## 2. Dynamic vs. non-stationary

These are different properties, and the repo implements and measures them
separately.

**Dynamic** means the orchestrator reacts to *current state*:

- a full zone escalates;
- digests change which zones are candidates;
- a failed node triggers re-planning of its services;
- memory cases are refused when the live digest says the remembered capacity
  has been consumed;
- a failure switches the zone's solver to a load-balancing policy.

This is a property of the controller and is present in every run.

**Non-stationary** means the *workload's generating distribution* changes
over time (`src/workload.py`). Nothing is a hard cut:

- **Arrival intensity** follows the Milan diurnal curve. In the evening it is
  blended with the real surge day (±2 h raised-cosine ramp around the surge
  zone's peak excess).
- **Spatial mix** follows the per-slot zone distribution from the trace,
  which drifts through the day by itself, plus the surge-day mix in the surge
  window.
- **Service-type mix** follows a logistic ramp for each late type, centred on
  its introduction hour (7 h, 9 h, 11 h). On top of that, the core types have
  a gentle diurnal modulation.
- **Genuinely new types**:
  - `crowd_safety`: first traffic ≈ 13.5 h; registered in the catalog at
    14.5 h, 30 min *after* its traffic starts;
  - `ev_charging`: appears ≈ 17 h and is never registered, so it can only be
    handled through the SLM's free-text label plus the nearest-match/novel
    fallback.
- **Phrasing pool** widens continuously: the available share of held-out
  templates grows linearly from 35% to 100%, place names grow from 10 to 20
  between 8 h and 20 h, and OOD sentences enter after 12 h.
- **Devices move** (§8) and have phrasing habits, so a moving device brings
  phrasings its new zone's cache has never seen.

The phase labels A (00–08 h), B (08–16 h) and C (16–24 h) are *reporting
windows only*. Evidence of drift is written per seed to
`scenario/workload_seed<k>.report.json`: per-phase type mix, zone mix,
distinct phrasings, and first-seen times. For example, the full profile's
seed 0 has 265, 829 and 1212 distinct texts in A, B and C.

## 3. One event log, three outcomes

`src/telemetry.py` is the only place anything is counted. It has four streams:

- **`RequestRecord`** per request: translation source, correctness, path,
  decision source, action, zone/node, cross-zone flag, degradation level,
  victims, latency breakdown, tokens by role × (prompt, completion), calls by
  role × source, every state transition, end cause.
- **`LLMCallRecord`** per SLM/LLM call attempt, at every call site.
- **`EventRecord`** for cluster events: failures, pre-emptions, migrations,
  registrations.
- **`Snapshot`** at digest-tick cadence: cumulative counters, cache and
  memory sizes, per-zone utilisation.

`metrics.py` reads only this log, and refuses incomplete records
(`assert_complete`).

- **Reached the LLM stage** (`reached_llm_stage`), **invoked a model**
  (`calls[*_fresh] + calls[*_cached_disk]`) and **made a fresh network call**
  (`calls[*_fresh]`) are three separate fields, fixed once in
  `MultiLLM.ask()`, which is the only path to a model. The on-disk response
  cache is an *experiment-cost* device (so re-runs don't re-bill every call).
  In the simulated world a `cached_disk` answer is still an invocation and is
  charged the same simulated latency, but it is always counted separately.
- **Accepted** means placed on arrival. **Completed** means it ran to its
  natural end, not lost to eviction or failure. **Escalation success** means:
  of the requests that needed more than a local full-size placement, the
  share that was placed. The three are always reported separately; the test
  `test_metrics.py` checks they differ on a fixture where they should.

## 4. Latency (`src/latency_model.py`)

The total per request is the sum of six components. Raw Python wall-clock of
a whole decision is never reported.

| component | source |
|---|---|
| transport_in | topology `intra_zone_ms` (1.0 ms, RIPE Atlas metro band) |
| translation, cache hit | measured single-sentence MiniLM encode time on the host, plus the measured cosine lookup |
| translation, SLM | the SLM call's latency (see below) |
| escalation | `zone_to_global_ms` of the origin zone (drawn once per topology from RIPE `zone_to_cloud_ms` = 20–40 ms), plus one round trip per probe / running-list query |
| decision | local / memory / digest: measured `perf_counter` of *that deterministic operation only*; LLM: the call's latency, and each retry is another call |
| cross_zone | topology `inter_zone_rtt_ms[origin, target]` whenever the final zone differs. Never silently 0. |
| deployment | per-device-class container start (§8) |

SLM/LLM call latency comes from one of three sources, recorded in each run's
metadata as `latency_source`:

- **`calibrated`**: empirical samples from `src/latency_distributions.json`,
  written by `scripts/calibrate_latency.py` on the GPU machine, per model and
  prompt kind (translate / decide / rule_author / react_step / zone_choice;
  the other baseline kinds map onto these).
- **`live`**: that file is absent, so each call is charged its own measured
  wall time. This is real but host-dependent.
- **`placeholder`**: mock LLM, smoke runs only.

**Status:** measured on the GPU machine with `scripts/calibrate_latency.py`.
It must be committed from there so every run and every parallel shard samples
the same distributions; this repo was written on a machine without a GPU.

Capacity is reserved at decision time. Setup latency is reported, but it does
not delay the service start in the simulation, because setup latency (ms–s)
is small next to service lifetimes (minutes–hours).

Only the **arrival** placement's container start counts towards a request's
latency. When a running service is re-placed (migration after pre-emption,
recovery after a node failure), that deployment draw is recorded on the
`migrate` event as `redeploy_ms`, together with the re-planning latency, and
does not change the original request's breakdown. (Fixed after the first GPU
seeds: earlier code charged every re-placement's deployment to the original
request, slightly inflating total latency for interrupted services.)

## 5. Tier 2 details

- **Cold start.** Every zone cache starts empty (`SEED_CACHE_PER_TYPE = 0`;
  a 1–2 per-type seed from the *training* pool is supported).
- **Training vs held-out phrasings.** The training pool is the 3 hand-written
  seed templates per core type; only these may ever seed a cache. The
  workload draws exclusively from the held-out pool: the 9 LLM paraphrases
  per type, the 84-sentence OOD probe, and the hand-written templates of the
  two new types (these stay held-out even if the paraphrase script later
  expands them). A cache hit is therefore earned either by similarity or by
  an earlier SLM translation of real traffic.
- **Threshold sweep** (`zone_agent.threshold_sweep`). The library is the
  training pool × places; the queries are the held-out pool, **including
  phrasings of the unseen types, where any hit is a false hit**. MiniLM
  results:

  | θ | hit rate (known types) | false-hit rate (unseen types) | precision |
  |---|---|---|---|
  | 0.30 (old repo's choice) | 1.00 | **1.00** | 0.905 |
  | 0.60 | 0.80 | 0.27 | 0.975 |
  | **0.65 (chosen: lowest θ with precision ≥ 0.98)** | 0.65 | 0.09 | 0.990 |
  | 0.75 | 0.42 | 0.00 | 1.000 |

  The old 0.3 threshold looked perfect only because the old evaluation never
  offered an unseen type. It would silently map every new service onto a
  known one.
- **Shadow check (trust-based).** Cache hits are re-translated by the SLM in
  the background at 5%. If it disagrees, the cache entry is replaced. This
  costs tokens (charged to that request) but not request latency, and its
  disagreement rate is a metric. Two refinements stop audits from becoming a
  permanent floor of SLM calls:
  - an **exact repeat** of text the SLM itself translated is never audited:
    at temperature 0 the SLM would give the same answer, so there is nothing
    to catch;
  - an entry that has **passed 2 audits** is trusted and re-audited at 10% of
    the base rate. Any disagreement resets it to full auditing.

  On the real seed-0 workload, this cut audits by 29% (546 → 390) with the
  audit disagreement rate unchanged.
- **Open vocabulary.** The SLM prompt lists the *live* catalog and allows a
  new snake_case label. The label is resolved as exact, then nearest
  registered type by embedding (≥ 0.55), then `novel:<label>` with a generic
  manifest.
- **LRU cap** of 500 entries per zone.

## 6. Tier 3 details

- **Similar-case memory.** The key is (service_type, latency_class, priority,
  log₂ CPU bucket, origin zone). A case stores the chosen zone, node, action,
  degradation level, victim priority, the free CPU seen, and the time.
  Retrieval is exact key first, then similar keys (same type / latency /
  priority, CPU bucket ±1, any origin). Up to 3 remembered cases are tried
  per escalation.
- **Re-check before reuse.** A case is only reused if the live digest still
  shows a node that fits (or, with digests off, a probe succeeds). Otherwise
  it is counted `memory_stale` and skipped.
- **Negative cases.** Zones that failed a probe for a key are avoided for
  60 s and passed to the LLM as `memory_hints.failed_zones`. Memory also
  remembers degrade and pre-empt decisions: a remembered pre-emption
  re-applies its *pattern* (a victim of ≤ that priority, nearest completion)
  in that zone.
- **Procedural rules** are authored every 120 s from episodic majorities
  (≥ 5 episodes, ≥ 60%). The LLM drafts each rule and code validates it
  against the episodic majority.
- **Digest scan.** Fresh digests whose top-3 nodes show a fit, ordered by RTT
  from the origin; at most 3 probes. When the zone agent has just failed a
  live full-size solve in the origin, the origin is not re-offered for
  full-size placement (`tried_local`).
- **One structured LLM call.** The prompt carries the request profile and
  size, per-zone digest summaries and RTTs, and an explicit options block:

  - `place`: zones where the request fits at full size;
  - `preempt`: strictly-lower-priority running services whose eviction alone
    frees enough room; only for high/critical requests; up to 6;
  - `degrade`: (zone, level) with the *highest* level in {0.8, 0.6, 0.4} that
    fits, at or above the service type's floor (lower levels are dominated).
    The floors are drone 0.8; video / AR / twin / crowd 0.6; others 0.4.

  **Options are pre-verified.** One parallel round trip to the nearest
  promising zones (up to 4; one `zone_to_global_ms`, the max over the queried
  zones) asks each zone's solver whether the request fits at full size, else
  at which degradation level. The same round trip collects pre-emption
  candidates. The LLM therefore only chooses among actions that are feasible
  at that instant. (Earlier, options came from digests up to 5 s old: on the
  real seed-0 workload a third of LLM decisions, 548 of 1,693, failed
  verification and cost a second call. With pre-verification, failed
  verifications fell to the injected error rate of the test model and total
  decision calls fell 27%.)

  The chosen action is re-checked at the same instant. On failure there is one
  retry with the verifier's error as feedback, then the rule chain takes
  over. The LLM is not called at all when there are no options.

  **Stated objective.** Every LLM decision prompt carries the same objective
  text (`global_agent.OBJECTIVE`): the proposed system, CORE (which shares
  the prompt), ReAct/LATS (via the tool description) and AgentEdge's planner.
  It says: serve as many requests as possible; a rejection is the worst
  outcome; every offered degradation level is within what the service
  accepts; pre-emption only takes from strictly lower priority. This was
  added after the first GPU pilot. Without it, Qwen2.5-7B answered "reject"
  to 85% of decisions (1,829 of 2,157 on seed 0) even though every offered
  option had been verified to fit, and escalation success stayed around 0.5.
  Each run's metadata keeps up to 25 sample reject reasons and invalid
  answers (`meta.llm_samples`), and counts invalid answers per action type
  (`g_llm_invalid_<action>`). The prompt grew by about 100 tokens after
  latency calibration; the calibrated `decide` latency was measured with the
  shorter prompt.
- **Pre-emption and migration.** The victim is evicted first. The request
  takes the freed capacity. The victim then goes through the same `replan()`
  path as failure recovery (local, then escalate), which may itself pre-empt
  further (depth ≤ 3). If this fails, the victim ends `preempt_unmigrated`.
  Re-planning uses a scratch record, so its cost never pollutes the original
  request's arrival latency, while its LLM calls and tokens still count.

### Measuring "model calls fall over time"

Raw SLM+LLM invocations **per request** follow the daily load curve. At
night the zones are nearly empty and almost nothing escalates; in the day
up to ~60% of requests escalate. On the first real GPU seeds, per-request
usage roughly doubled from the first to the last quarter of the day (about
0.15 → 0.35) while the need for model calls grew even faster. Changing the
workload to hide that would be rigging it, so the claim is measured in
three need-normalised ways instead (all from the same telemetry):

| measure | what falls if the claim holds |
|---|---|
| `cache_miss_first/last_quarter` | share of requests the cache can't translate |
| `llm_per_escalation_*` (incl. busy-hours halves, slope) | LLM calls needed per escalation |
| `tables/savings_vs_ablations*.csv`, `fig_savings_vs_ablations.png` | share of invocations the full system saves vs `no_memory`, `no_intent_cache`, `memory_ablated`, `no_digest` on the *same* workload, hour by hour, paired by seed |

The third is the causal evidence: if memory and the cache matter, the
savings are positive and grow as they fill. The raw per-request curve is
still reported next to them.

## 7. Ablations (`src/ablations.py`)

Each ablation is the same `HierarchicalOrchestrator` with exactly the listed
flags off. `tests/test_ablations.py` checks both the flags and the behaviour.

| ablation | turns off | isolates |
|---|---|---|
| `no_memory` | episodic + procedural memory (digest ON) | memory specifically |
| `no_digest` | digest exchange (memory ON; memory re-checks by probing; LLM sees no capacity; blind probes) | the digest specifically |
| `memory_ablated` | memory **and** digest | the old repo's `MemoryAblated`, now documented as the combination |
| `no_intent_cache` | zone caches | the intent cache (SLM translates every request) |
| `no_preempt_degrade` | pre-emption and degradation, in the LLM's options and the rule chain | the value of those actions |
| `no_zone_tier` | zone agents: a central translator + central placement; every request pays the zone↔global hop | Tier 2 itself |
| `no_cross_zone` | cross-zone targets | zone cooperation |

## 8. Baselines (`src/baselines/`)

All baselines run on the same simulator, workload, seeds and telemetry.
Mechanisms are reimplemented for this environment; no other paper's numbers
are borrowed.

| baseline | adapted from | mechanism here | memory / cache |
|---|---|---|---|
| `greedy_oracle` | old repo | Flat least-loaded placement. **Handed the true service type** (`translation_source=oracle`), so it isolates placement architecture and is the load-calibration reference. Ignores locality; violations are measured. | none |
| `rule_based` | old repo | Static library of training phrasings (never learns, no SLM, unknown intents rejected); local placement, then nearest digest-fit zones; no pre-emption. | static only |
| `react` | ReAct (Yao et al., 2023) | Thought plus one tool call per step over `read_digest(zone)`, `read_running_services(zone)`, `try_place(zone,node,service_type[,evict,degrade_level])`; cap 6 steps (in the 5–7 range where ReAct found more steps stop helping); translation happens inside `try_place`. | none |
| `lats` | LATS (Zhou et al., 2024) | MCTS over the same tools: n = 3 sampled expansions (temperature 0.7); value = 0.8 × LLM self-score + 0.2 × self-consistency; UCT; real execution of reads; `try_place` is terminal; reflection on failed branches; ≤ 20 rollouts, depth 6; stops at first success. The most token-expensive baseline by design. | none |
| `agentedge` | AgentEdge-style + ActSimCrit | Four LLM roles: intent → observability → planning → infra-action. A **simulate-before-execute** validator (the capacity solver and locality check stand in for a digital twin) feeds infeasible plans back to planning (≤ 2 re-plans). ≥ 4 calls per request, not expected to fall over time. | none |
| `core` | CORE-style role affinity | Device role (no model): local solve. Edge role (SLM): translation every request, plus zone choice when a digest-fit zone exists. Cloud role (LLM): the same one-call decision when pre-emption or degradation is needed. | none |
| `optimal_solver` | offline ILP | A **ceiling, not a competing online system.** Pooled-capacity relaxation per 2 h window: y[r, level] binary, capacity at every arrival checkpoint per zone (zone-local services) and globally; failures ignored; empty start. HiGHS MILP, with an LP-relaxation fallback that is labelled as such. A relaxation of every online policy, including migration and degradation, hence a valid upper bound (`test_baselines.py` checks this). | n/a |

## 9. Workload, data and load calibration

| dimension | source |
|---|---|
| intensity, spatial mix, surge | Milan telecom activity, 14 days (Harvard Dataverse doi:10.7910/DVN/EGZHFV); base day 2013-11-07, surge day 2013-11-13 |
| zoning | KMeans over Milan grid cells weighted by mean activity (10 zones) |
| per-request CPU / memory / lifetime | Alibaba cluster-trace-v2018 `batch_task` percentiles × per-type scale |
| failures | Google cluster-trace-v1 machine events. `FailureModel.failure_schedule()` runs a per-node renewal process (log-space inverse-CDF sampling of inter-failure and downtime; length-biased first gap). 3–14 failures per simulated day across the 10 full-profile seeds (mean 8.6) on 50 nodes, the right order for the trace's MTBF of 113 h (≈ 10.6/day expected). `min_failures = 3` forced draws are flagged. |
| RTT | RIPE Atlas metro band (4.3–9.4 ms inter-zone, 1 ms intra, 20–40 ms to the cloud tier), scaled by centroid distance, kept from the old repo |
| nodes | 50 nodes: 4 rack servers (in the 4 highest-activity zones only), 14 Jetson, 20 Pi 4, 12 Coral. Most zones consist only of small devices, so cross-zone escalation matters. |

Load calibration (`src/calibrate_workload.py`) uses a lifetime scale factor K,
bracketed and bisected until `greedy_oracle` acceptance on seed 0 is in
[0.80, 0.90]. It is cheap and model-independent. K is stored in
`scenario/calibration.json` and applied to every seed.

## 10. Where real numbers were approximated, and why

- **Device specs** (`src/device_specs.py`) are transcribed from the four
  official datasheets named in each entry (Raspberry Pi 4 8 GB, Jetson Orin
  Nano 8 GB at 15 W, Coral Dev Board 4 GB, Dell PowerEdge 64-core / 512 GB /
  800 W). Where a datasheet offers a range, the representative choice is
  stated in `config_choice`. The figures were not re-fetched live while
  writing this repo; the transcribed table is what the brief specified.
  MLPerf Tiny (github.com/mlcommons/tiny and tiny_results_v1.0) is cited as
  the standard low-power inference benchmarking methodology only. It is not
  the source of any number.
- **Deployment (container start) time** is an explicit assumption
  (log-normal; medians: Pi 2.5 s, Jetson 1.5 s, Coral 3.0 s, rack 0.6 s),
  flagged `deploy_is_assumption=True`. Run
  `scripts/generate_device_calibration.py --measure-container-start --device-class <class>`
  on real hardware to replace it. It is identical across systems for the same
  placement, so it cannot change a ranking.
- **Mobility** is a coarse Milan-driven approximation, *not* individual
  trajectories. Each device has a home zone ~ activity share. At each 10-min
  slot it moves with probability 0.06: home with probability 0.4, otherwise
  to a gravity-weighted (activity × distance-decay) zone. Public
  taxi-trajectory datasets (e.g. T-Drive Beijing, Porto) exist, but they are
  in different cities from the Milan activity and zoning everything else is
  built on. Mapping them onto the Milan grid would mix two unrelated spatial
  processes, so the Milan-derived approximation was kept deliberately. No web
  search was done while writing this repo.
- **SLM/LLM latency**: see §4. Measured on the GPU machine; the script is
  provided.
- **Resource request known at placement** (manifest assumption, §1), and
  **degraded services keep their lifetime** (running at reduced resources
  lowers quality, not duration).
- **Energy** (`energy_kwh_alloc_share`) is approximate: the allocated CPU
  share × the node's rated power × the time held. It ignores idle power.
- **Synthetic stand-ins** (`SYNTHETIC_FAILURE_MODEL`,
  `SYNTHETIC_RESOURCE_PROFILES`, `synthetic_topology`,
  `synthetic_zone_activity`, `mock_llm.py`) exist only for the smoke profile
  and tests. They are labelled everywhere, and every smoke figure and table
  carries a "not results" banner.

## 11. Model comparison

All 8 local tiers from `local_llm/models.py` are kept (1B–14B; Qwen, Llama,
Phi, Mistral, Gemma families). One hosted reference (Llama-3.3-70B) is added,
labelled "hosted reference, not a deployment candidate".
`scripts/run_model_sweep.py` runs, per tier: the model-comparison probe
(translation on held-out phrasings; 100 auto-scored decision cases), latency
calibration, and one full end-to-end evaluation with that model in both
roles. If GPU memory forces a tier to be dropped, record it here: *(no cuts
yet, as the sweep has not been run)*.

## 12. Old-repo flaws and where they are fixed

| # | flaw | fix |
|---|---|---|
| 1 | one orchestrator object pretending to be every zone | `ZoneAgent` per zone (`zone_agent.py`, `orchestrator.py`) |
| 2 | cache preloaded with every workload phrasing | cold cache, train/held-out split, sweep with unseen-type negatives (§5) |
| 3 | closed 7-type catalog | live `ServiceCatalog`, mid-run registration, never-registered type, open-vocab resolution |
| 4 | "similar case" = same service type | multi-feature key, negatives, live re-check (§6) |
| 5 | two LLM calls (zone, then eviction) | one structured, verified call |
| 6 | fixed 60/80% degradation the LLM never saw | explicit menu with per-type floors inside the same call |
| 7 | Python wall-clock as latency, RTT ignored | component latency model (§4) |
| 8 | "reached LLM" vs "called LLM" blurred | enforced in `MultiLLM.ask()` + telemetry (§3) |
| 9 | acceptance used as completion | three distinct outcomes (§3) |
| 10 | one hard-coded failure | renewal-process failure schedule |
| 11 | three hard-cut phases called "non-stationary" | gradual drift; dynamic vs non-stationary separated (§2) |
| 12 | `MemoryAblated` = memory+digest silently | `no_memory` / `no_digest` / `memory_ablated` |
| 13 | no test of the zone tier's value | `no_zone_tier` |
| 14 | no external baselines | ReAct, LATS, AgentEdge-style, CORE-style (§8) |
| 15 | model comparison never run at scale | 100-case auto-scored probe, served-model check, sweep script. The old table's identical rows for two labels are consistent with both being pointed at one server; the new harness refuses that. |

Also fixed along the way: coarse LLM disk-cache keys (the old
`goa|type|cpu|digest-summary` could replay an answer given for a different
prompt; now the full prompt is hashed), and quadratic cache writes (now
append-only JSONL).

## 13. Running the campaign in parallel

llama-cpp-python's server answers one request at a time, so wall time is
dominated by sequential model calls. `scripts/run_parallel.py` starts N
independent SLM+LLM server pairs and N evaluator shards
(`evaluator.py --shard i/N`, interleaved (system, seed) jobs).

This changes run time only, never results:
- each (system, seed) run is an independent simulation with its own seeded
  RNGs and its own LLM disk cache (`cache/<profile>/<system>/seed<k>/`);
- with calibrated latency, simulated latency is sampled, not measured, so
  contention between servers cannot leak into results. The script refuses to
  run in `live` latency mode unless explicitly allowed.

Checked on the smoke profile: two shards against a sequential run gave
identical values for all 1,358 non-latency metric means. The latency
components that time deterministic compute with `perf_counter` differ in
the last digits, as they do between any two runs.

Baselines run cheapest first (greedy, rule-based, CORE, ReAct, AgentEdge,
LATS), so a partially finished campaign always holds the most important
comparisons. All systems keep all 10 seeds: the paired Wilcoxon test cannot
reach p < 0.05 with fewer than 6 pairs, so cutting seeds for the expensive
baselines would weaken exactly the comparisons that need to be defended.

## 14. Known limitations

- Services do not follow their user when the device moves. Mobility affects
  only where new requests originate.
- The global agent is a single logical instance; its own failure is not
  modelled.
- ReAct, LATS and AgentEdge prompts include the whole catalog and tool
  observations, so their token costs depend on prompt design choices made
  here. The prompts are in the source for inspection.
- With the disk cache on, re-running a seed turns fresh calls into
  `cached_disk` ones. Use invocation counts (fresh + cached_disk) for the
  call-reduction claim, and fresh counts only for actual compute spent.
