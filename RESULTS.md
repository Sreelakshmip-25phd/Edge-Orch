# Results so far

This repo was written on a machine **without a GPU or model server**. Two
kinds of runs have been done there:

1. **Real data, no model.** Scenario, workload and load calibration for the
   `full` profile, plus the two baselines that need no LLM, on all 10 seeds.
   These numbers are real.
2. **Smoke.** All 14 systems end to end on synthetic data with the **mock
   LLM** and placeholder latencies. These only show that the pipeline works;
   they are **not results**.

Everything involving an SLM/LLM (the proposed system, all ablations, ReAct,
LATS, AgentEdge-style, CORE-style, the model comparison, and the latency
calibration) still has to be run on the GPU machine. See "Still to run" below.

## 1. Real-data scenario (`--profile full`)

The topology is 10 KMeans zones over the Milan grid with 50 nodes: 4 rack
servers (only in the 4 busiest zones), 14 Jetson Orin Nano, 20 Raspberry
Pi 4 and 12 Coral boards.

- **Workload:** 12,000 requests per simulated day per seed, from 2,400 mobile
  devices. Base day 2013-11-07, surge day 2013-11-13, surge zone z5.
- **Drift evidence (seed 0):**
  - requests per phase A/B/C: 2358 / 5054 / 4588;
  - distinct phrasings per phase: 265 / 829 / 1212;
  - first appearance: `federated_ml` 6.3 h, `digital_twin` 7.8 h,
    `drone_control` 10.1 h, `crowd_safety` 13.5 h (registered 14.5 h),
    `ev_charging` 17.4 h (never registered);
  - 0 forced device picks (mobility never had to teleport a device).
- **Failures:** 3–14 node failures per seed (mean 8.6), sampled from the
  Google trace renewal model.
- **Intent-cache threshold:** 0.65, the lowest threshold with hit precision
  ≥ 0.98 when unseen-type phrasings count as false hits. The old repo's 0.3
  gives a 100% false-hit rate on unseen types (see ARCHITECTURE.md §5).
- **Load calibration:** K = 48, which gives greedy acceptance 0.847 on seed 0
  (target band 0.80–0.90). This is the same K the old repo found.

## 2. Real-data baselines without an LLM (10 seeds, mean ± 95% CI)

| metric | greedy_oracle | rule_based |
|---|---|---|
| acceptance | 0.834 ± 0.007 | 0.716 ± 0.003 |
| completion | 0.832 ± 0.007 | 0.714 ± 0.003 |
| escalation success | 0.812 ± 0.007 | 0.227 ± 0.006 |
| acceptance A / B / C | 0.999 / 0.805 / 0.781 | 0.967 / 0.724 / 0.578 |
| acceptance of the two new service types | 0.885 ± 0.013 | 0.128 ± 0.004 |
| service-type accuracy | 1.000 (oracle-typed) | 0.829 ± 0.002 |
| locality violations (zone_local placed elsewhere) | **0.377 ± 0.005** | 0.0003 |
| cross-zone share of placements | 0.861 | 0.116 |
| services lost to node failures (per run) | 19 ± 17 | 19.7 ± 12 |
| mean total latency (ms, incl. deployment) | 1318 ± 16 | 1892 ± 24 |

What this already shows:

- **Acceptance and completion differ** once failures are sampled, and escalation success
  is a separate number again.
- **The greedy reference buys its acceptance by ignoring data locality.** It
  places 38% of zone-local services outside their zone. This is now measured
  instead of hidden.
- **A static rule library does not survive the non-stationary workload.**
  Rule-based acceptance falls from 0.97 in phase A to 0.58 in phase C.
- **Rule-based accepted 12.8% of the two new types' requests**, which it has
  no templates for. It only could by mapping them onto a known type: these
  are false cache hits, the risk §5 of ARCHITECTURE.md describes.

Tables are in `results/full/tables/` and figures in `results/full/figures/`
(both gitignored; `python main.py` regenerates them deterministically).

Runtime on this CPU host: `greedy_oracle` takes about 15 s per seed.
`rule_based` took 134–350 s per seed and got slower over the batch; that is
worth profiling before the full campaign.

## 3. Smoke run (mock LLM; NOT results)

This is `python main.py --profile smoke`: 600 requests over a compressed
6-hour day, 4 synthetic zones, 2 seeds, all 14 systems. It runs from scratch
in about 2.5 minutes. It demonstrates that every system, the metrics, paired
tests, tables and figures work end to end. Model invocations per request
(mean of 2 seeds):

| system | invocations/req | system | invocations/req |
|---|---|---|---|
| full | 0.51 | react | 3.0 |
| no_memory | 0.61 | agentedge | 5.6 |
| no_digest | 0.83 | lats | 10.9 |
| memory_ablated | 1.15 | core | 1.4 |
| no_intent_cache | 1.16 | greedy_oracle / rule_based | 0 |

These are mock-LLM numbers. They show the mechanisms are wired correctly
(removing memory, digest or cache raises model usage; LATS is the most
expensive by design), but say nothing about real model behaviour.

## 4. Still to run on the GPU machine

1. `python scripts/calibrate_latency.py --target <slm>@<url> --target <llm>@<url>`,
   then commit `src/latency_distributions.json`. It is currently absent; real
   runs fall back to per-call measured wall time.
2. `python main.py` for the full campaign: 14 systems × 10 seeds. Scenario
   and calibration are deterministic and regenerate identically.
3. `python scripts/run_model_sweep.py`: the probe, latency calibration and
   one end-to-end evaluation for each of the 8 local tiers, plus the hosted
   reference probe.
4. `python scripts/paper_results.py --profile full`, then update this file
   with the real tables.
5. Optional: `scripts/generate_device_calibration.py --measure-container-start --device-class rack_edge_server`
   to replace the assumed container start time for the server class.
