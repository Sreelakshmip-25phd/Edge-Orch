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
- **LRU cap** of 500 entries per zone (5,000 for the shared cache of 10 zones).
- **One cache for all zones (stage 3).** All zones belong to one operator, so
  the zone agents share one intent cache. A zone sees its own new entries at
  once and other zones' entries after 5 s, as if each entry travels with the
  next capacity-digest broadcast. The quick pilot gave every zone its own
  cache: a phrase learned in z3 was a miss in z7. Replaying its telemetry,
  83–85% of zone cache misses would have been hits in a shared cache (miss rate
  about 38% → 6%). `no_cache_sharing` keeps the per-zone design, so the value
  of sharing is measured like every other mechanism. Each hit records whether
  another zone wrote the entry (`cache_hit_shared`). Privacy is not an issue
  in this setting: the operator's global agent already sees every profile.

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
  size, per-zone digest summaries and RTTs, and a list of numbered choices
  built from these verified options:

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
  (`g_llm_invalid_<action>`). The quick-profile pilot that followed showed
  rejects down to 7% (8 of 111 decisions), but 68 of 111 first answers
  proposed "place" when no full-size placement existed, each costing a retry.
  The prompt therefore also listed `allowed_actions`. In the quick run that
  followed, 12–19 decisions per seed (of 100–157 that reached the LLM) still
  failed verification twice and went to the rule chain; 46 of 53 failures on seed 0 were degrade answers with the
  right level but `zone: null`, i.e. the model failed to copy a field back.
  **Choices by id (stage 3):** every verified option is now one entry of
  `choices` with an id (`o1`, `o2`, …): place (zone), preempt (zone, victim,
  its priority/type/remaining time), degrade (zone, level), and reject (only
  when no full-size placement exists). The model answers
  `{"option": "o2", "reason": "..."}`, so there is nothing to copy wrongly.
  Place choices come first, nearest first. CORE uses the same call, so it
  gets the same format. The model-comparison probe (`reasoning_cases.py`)
  shuffles the choice order per case, so always answering `o1` is not
  rewarded. The prompt has changed since latency calibration; the calibrated
  `decide` latency was measured with the earlier prompt.
- **Rule chain.** The fixed place → pre-empt → degrade → reject chain runs
  only when the LLM's answer fails verification twice or no model is
  reachable. In the quick pilot it changed the outcome of 4–9 requests per
  1,000 (placed with degradation after two invalid LLM answers). Each run
  reports how often it was used (`fallback_used`).
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
| `tables/savings_vs_ablations*.csv`, `fig_savings_vs_ablations.png` | share of invocations the full system saves vs `no_memory`, `no_intent_cache`, `no_cache_sharing` on the *same* workload, hour by hour, paired by seed |

The third is the causal evidence: if memory and the cache matter, the
savings are positive and grow as they fill. The raw per-request curve is
still reported next to them.

## 7. Ablations (`src/ablations.py`)

Each ablation is the same `HierarchicalOrchestrator` with exactly the listed
flags off. `tests/test_ablations.py` checks both the flags and the behaviour.

The quick pilot also ran memory-only (`no_memory` with the digest kept) and
digest-only (`no_digest`) variants. Digest-only had no measurable effect on
any metric, and memory-only cost 1.17× calls but was not significant (Holm
p = 0.10); memory and digest together cost 1.41× calls (significant). The
two single-mechanism variants were therefore merged into one `no_memory`
ablation (memory, rules and digest off).

| ablation | turns off | isolates |
|---|---|---|
| `no_memory` | episodic + procedural memory **and** the capacity digest | everything the global tier remembers or is told about other zones |
| `no_intent_cache` | the intent cache | the intent cache (SLM translates every request) |
| `no_cache_sharing` | sharing: every zone keeps its own cache | sharing translations across zones |
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
| `react` | ReAct (Yao et al., 2023) | Thought plus one tool call per step over `read_digest(zone)`, `read_running_services(zone, service_type)`, `try_place(zone,node,service_type[,evict,degrade_level])`; cap 7 steps (the top of the 5–7 range where ReAct found more steps stop helping); translation happens inside `try_place`. | none |
| `lats` | LATS (Zhou et al., 2024) | MCTS over the same tools: n = 3 sampled expansions (temperature 0.7); value = 0.8 × LLM self-score + 0.2 × self-consistency; UCT; real execution of reads; `try_place` is terminal; reflection on failed branches; when every child of a node has failed, the node is **re-expanded** with the reflections in the prompt (new samples; actions already tried there are dropped); ≤ 20 rollouts, depth 7; stops at first success or when nothing new can be tried. The most token-expensive baseline by design. | none |
| `agentedge` | AgentEdge-style + ActSimCrit | Four LLM roles: intent → observability → planning → infra-action. A **simulate-before-execute** validator (the capacity solver and locality check stand in for a digital twin) feeds infeasible plans back to planning (≤ 2 re-plans). ≥ 4 calls per request, not expected to fall over time. | none |
| `core` | CORE-style role affinity | Device role (no model): local solve. Edge role (SLM): translation every request, plus zone choice when a digest-fit zone exists. Cloud role (LLM): the same one-call decision when pre-emption or degradation is needed. | none |

**ReAct/LATS fairness fixes (stage 3).** In the quick pilot ReAct rejected
687 of 1,000 requests, and in 519 of those it never called `try_place`: it
spent its 6 steps listing running services (up to 25 of any priority per
call). LATS stopped as soon as all of the root's first children had failed,
whereas the LATS paper keeps searching with reflections. Changes, the same
text for both (`common.TOOLS_DOC`):
- the tool description says that `try_place` is the only way to serve a
  request, that it checks the fit itself and explains a failure, and that
  reading running services is only worth a step when nothing has room and the
  request is high or critical priority;
- `read_running_services` returns only services the request may evict
  (strictly lower priority), lowest priority and soonest finishing first, at
  most 8;
- ReAct's step cap and LATS's depth are both 7;
- LATS re-expands a node whose children have all failed (see the table).

- **Prompt layout (after the stage-3 quick run).** ReAct still reached only
  25% acceptance: in 588 of 981 requests it called `read_digest` on the same
  zone 7 times in a row and never acted (5,653 of its 6,119 calls were
  `read_digest`, 4,806 identical to the previous call), while 251 of its
  301 `try_place` calls succeeded. The cause was ours: each step's message was
  JSON with sorted keys, which put the trajectory (`history`) *before* the
  task (`request`), with no step counter, so the model re-did step 1 every
  turn. ReAct and LATS now send the task first, then the numbered steps,
  then "choose step k of 7" (`common.trajectory_prompt`). The tool text says
  the cluster does not change while a request is decided, and a call
  identical to one already in the trajectory is answered with a note
  pointing to that step instead of being re-run (`common.repeat_note`).

All baselines keep their published mechanism; these changes remove handicaps
that came from our prompts and tool text, not from the methods.

The offline optimal bound of earlier versions was dropped: it saw the whole
day in advance, and its gap to the online systems was not needed for any claim.

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
| 12 | `MemoryAblated` = memory+digest silently | `no_memory` documented as memory + rules + digest (§7) |
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

### The `quick` profile

A first look in hours instead of weeks: the same topology, data and code as
`full`, but 1,000 requests over a compressed 3-hour simulated day (the whole
daily curve, surge and new service types replayed) and 3 seeds. Planned in two
stages: `proposed,ablations,simple` on 3 seeds (~3 h on an RTX 4070 SUPER),
then `agentic` (ReAct, AgentEdge, LATS) on seed 0 overnight. These cost about
52, 77 and 194 GPU-minutes per 1,000-request seed, estimated from their call
counts and the measured per-call latencies. (The pilot measured LATS at about
17 calls per request, roughly 4 hours per quick seed. Re-expansion lets LATS
keep searching where it used to stop, so expect it to take longer.)

With only 1,000 requests on a 50-node cluster, load can only reach the
calibration band if a large share of services run at the same time. So the
load factor is K = 512 (median service lifetime ≈ 85 min of the 3-hour run),
against K = 48 (≈ 8 min of a 24 h day) in `full`. The quick profile is
therefore for comparing systems on the same workload. Its absolute numbers
are not comparable with the full campaign's.

### The `medium` profile and the two experiments

Running every system on `full` (12,000 requests/day × 10 seeds) would take
about a month of GPU time, almost all of it in the agentic baselines: LATS
alone needs about 3.5–4 GPU-hours per 1,000 requests. The evaluation is
therefore split into two experiments, each paired on identical workloads:

| experiment | profile | systems | seeds | estimated GPU time |
|---|---|---|---|---|
| 1, main | `medium`: 6,000 requests over the real 24 h day, 1,200 devices | full, 6 ablations, greedy_oracle, rule_based, core | 5 | about 20 h |
| 2, agentic baselines | `quick`: 1,000 requests, compressed 3 h day | full vs react, agentedge, lats | 3 | about 6–7 h per seed |

`medium` keeps the full day's drift: on seed 0, 1,135 / 2,560 / 2,305
requests in the three thirds of the day, 238 / 697 / 875 distinct phrasings,
and the new types first seen at 6.4 h (federated_ml) to 18.1 h (ev_charging).
Its load factor is calibrated like the others: K = 128 gives greedy
acceptance 0.803 (target 0.80–0.90). Numbers are compared only within an
experiment, because the load factors differ.

With 5 seeds the paired t-test is the significance test: a two-sided
Wilcoxon test on 5 pairs cannot go below p = 0.0625 even when every pair
agrees, so it is reported but cannot decide. Holm correction is applied per
metric within each table (the ablations; the baselines). Seeds 5–9 can be
added to `medium` later without re-running anything.

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
- Model servers are modelled without a queue, and a decision's latency does
  not delay the placement in simulated time: each request is decided at its
  arrival instant, and every model call is charged its own sampled latency.
  A real single GPU answers one call at a time. This favours the systems that
  make many calls per request (LATS about 17, AgentEdge about 6, vs about 0.6
  for full in the quick pilot): with a queue their waits, and so their
  latency, would be larger.
- Service start-up (deployment) time is an assumed constant per device class
  (`device_specs.py`), not a measurement; it is excluded from the setup
  latency used in the main results and reported only in the appendix.
- Degradation floors and service priorities are design assumptions set in
  the service catalog (`scenario.py`), standing in for what a service owner
  would declare.
