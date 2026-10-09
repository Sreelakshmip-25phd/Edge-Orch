# Edge Agentic Orchestration (v2)

A three-tier multi-agent orchestrator that places services on edge/cloud
infrastructure. SLMs and LLMs do the reasoning, and an intent cache and
similar-case memory make those model calls unnecessary over time. The repo
contains the full evaluation harness: 6 ablations, 6 baselines (incl. ReAct,
LATS, AgentEdge-style and CORE-style reimplementations), a non-stationary
trace-driven workload, and every metric computed from one event log.

**Central claim under test:** SLM/LLM invocations per request fall over time as
the cache and memory fill, without the acceptance rate suffering.

- Design, and every place a real number was approximated: [ARCHITECTURE.md](ARCHITECTURE.md)
- What has been run so far: [RESULTS.md](RESULTS.md)

## Tiers

| tier | code | does |
|---|---|---|
| 1 edge devices | `src/edge_device.py` | send pure natural-language intent; move between zones |
| 2 zone agents (one per zone) | `src/zone_agent.py` | translate intent (one cold intent cache shared by all zones, falling back to the SLM), place locally |
| 3 global agent | `src/global_agent.py` | memory, then rules, then digest scan, then one LLM call that picks one of a list of pre-verified choices (place / pre-empt / degrade / reject) by id, then a fixed rule chain as a safety net ,in case LLM is unavailable or makes invalid decisions repeatedly|

## Quick start

```bash
pip install -r requirements.txt
python main.py --profile smoke      # ~3 min, no GPU, no model server: synthetic data + MOCK LLM
pytest -q                           # unit tests
```

The smoke profile proves the pipeline end to end. Its numbers are **not results**:
it uses a mock LLM and placeholder latencies, and its outputs are banner-tagged
as such under `results/smoke/`.

## Real runs (the GPU machine)

```bash
# 1. data: download (see src/data_foundation.py for the one-time Harvard
#    Dataverse guestbook step).
python src/data_foundation.py --import-from ../agentic_edge_orchestration

# 2. models
cd local_llm && pip install -r requirements.txt   # CUDA build: see local_llm/README.md
python start_server.py llama32_3b --port 8080 --also medium --also-port 8081 &
cd ..
export LOCAL_LLM_URL_SLM=http://127.0.0.1:8080 LOCAL_LLM_URL_LLM=http://127.0.0.1:8081
export SLM_MODEL_LABEL=llama32_3b LLM_MODEL_LABEL=medium

# 3. measure real model latency once, commit the file
python scripts/calibrate_latency.py --target llama32_3b@$LOCAL_LLM_URL_SLM --target medium@$LOCAL_LLM_URL_LLM
git add src/latency_distributions.json && git commit -m "Measured LLM/SLM latency on <GPU>"

# 4. run
python main.py --profile small      # sanity: 2k requests x 3 seeds
python main.py --profile medium --systems proposed,ablations,simple   # Experiment 1: 6k requests/day x 5 seeds
python main.py --profile quick --systems agentic                     # Experiment 2: 1k requests x 3 seeds
python main.py                      # full: 12k requests/day x 10 seeds x 13 systems (about a month of GPU time)
python scripts/paper_results.py --profile full

# 5. multi-model comparison (probe + calibration + one full evaluation per tier)
python scripts/run_model_sweep.py --tiers small,medium,large,llama32_1b,llama32_3b,phi35_mini,mistral7b,gemma2_9b
```

### Latency calibration status

`src/latency_distributions.json` is measured on the GPU machine by
`scripts/calibrate_latency.py`; commit it from there. Without it, real runs
charge each SLM/LLM call its own measured wall time (`latency_source: live`
in every run's metadata), and `scripts/run_parallel.py` refuses to start.
Nothing in this repo ships an invented latency distribution.

### Running the campaign faster (same results)

```bash
python scripts/run_parallel.py --workers 3 --slm llama32_3b --llm medium --dry-run   # plan
python scripts/run_parallel.py --workers 3 --slm llama32_3b --llm medium
```

This starts N independent SLM+LLM server pairs and N evaluator shards; pick N
so N copies of both models fit in GPU memory. Results are identical to a
sequential run (ARCHITECTURE.md §13).
Interrupting is safe: finished (system, seed) runs are kept, unfinished ones
restart from scratch next time.

## Useful commands

```bash
python main.py --profile full --from 5 --force          # re-run evaluation + report
python main.py --systems full,no_memory,react --seeds 0,1,2
python src/evaluator.py --profile full --report-only    # rebuild tables/figures
python src/calibrate_workload.py --profile full --force # re-derive the load factor K
```

## Outputs (`results/<profile>/`, gitignored)

| path | content |
|---|---|
| `scenario/` | topology, activity matrices, threshold sweep, workload per seed (+ drift report), calibration K |
| `runs/<system>/seed<k>.telemetry.jsonl.gz` | the complete event log of one run |
| `runs/<system>/seed<k>.metrics.json` | every metric, computed only from that log |
| `main/` | **the main results**: `table_baselines.md`, `table_ablations.md`, `fig1_calls_over_time.png`, `fig2_quality_vs_cost.png`, `fig3_acceptance_by_phase.png`, `fig4_setup_latency.png` (see `src/report.py`) |
| `tables/` | appendix: `summary.md/csv` (mean ± 95% CI), `paired_tests_vs_full.csv`, latency comparisons, calls over time, pre-emption by priority, placements per zone |
| `figures/` | appendix: outcomes, calls over time, latency, Pareto (success × latency × tokens), failures/pre-emption, tokens, load balance |
| `main/table_models.md` | model comparison, written by `scripts/paper_results.py` when it exists |

## Systems compared

- **proposed:** `full`
- **ablations:** `no_memory` (memory, rules and digest), `no_intent_cache`,
  `no_cache_sharing`, `no_preempt_degrade`, `no_zone_tier`, `no_cross_zone`
- **baselines:** `greedy_oracle`, `rule_based`, `react`, `lats`, `agentedge`, `core`

See ARCHITECTURE.md for what each one isolates or reimplements.

## Repository layout

```
main.py  config.py  requirements.txt
src/        telemetry, data_foundation, device_specs, scenario, workload,
            calibrate_workload, latency_model, sim_engine, edge_device,
            zone_agent, global_agent, orchestrator, llm_client, mock_llm,
            ablations, metrics, evaluator, scenario_build, baselines/
local_llm/  model servers, model_compare.py, reasoning_cases.py
scripts/    calibrate_latency, generate_device_calibration,
            generate_intent_paraphrases, run_model_sweep, paper_results
intent_data/  phrasing pools (seed templates, paraphrases, OOD probe)
tests/      unit tests
```
