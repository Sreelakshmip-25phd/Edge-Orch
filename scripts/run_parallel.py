#!/usr/bin/env python3
"""Run the evaluation campaign in N parallel shards - RUN ON THE GPU MACHINE.

Why this is safe (changes run time, never results):
  * llama-cpp-python's server answers one request at a time, so one server
    pair is the bottleneck. This starts N independent SLM+LLM server pairs
    (N copies of each model in GPU memory) and N evaluator processes, each
    pointed at its own pair and given every N-th (system, seed) job.
  * Every (system, seed) run is an independent simulation with its own
    seeded RNGs and its own LLM disk cache, so which process runs it - and in
    what order - does not change any decision.
  * With src/latency_distributions.json present (latency_source="calibrated")
    simulated latency is sampled from the measured distribution, not taken
    from wall time, so contention between servers can't leak into results.
    This script refuses to start without that file unless you pass
    --allow-live-latency.

    python scripts/run_parallel.py --workers 3 --slm llama32_3b --llm medium
    python scripts/run_parallel.py --workers 2 --slm llama32_3b --llm medium \\
        --systems react,agentedge,lats            # just the expensive baselines
    python scripts/run_parallel.py --workers 3 --dry-run   # show the plan only
    python scripts/run_parallel.py --workers 1 --profile quick --systems proposed,ablations,simple
--systems takes system names and/or groups: proposed, ablations, simple
(greedy_oracle, rule_based, core), agentic (react, agentedge, lats), all.

Memory: each worker holds one copy of both models plus their KV caches
(llama32_3b + medium at Q4 with 4k context is roughly 8-9 GB). Pick
--workers so N x that fits in GPU memory with ~10% headroom; check with
nvidia-smi after the servers are up. Logs: results/<profile>/logs/.
"""
import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import config  # noqa: E402
from config import LATENCY_CALIBRATION, PROFILES, results_dir  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "local_llm"))
sys.path.insert(0, os.path.join(BASE, "src"))
from models import MODELS  # noqa: E402


def wait_ready(url, timeout=1200):
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url + "/v1/models", timeout=5).ok:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--slm", default=config.SLM_MODEL_LABEL, choices=list(MODELS))
    ap.add_argument("--llm", default=config.LLM_MODEL_LABEL, choices=list(MODELS))
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--systems", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-live-latency", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import evaluator as E
    systems = E.expand_systems(a.systems) if a.systems else E.ALL_SYSTEMS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else PROFILES[a.profile]["seeds"]
    n = a.workers
    ports = [(a.base_port + 2 * i, a.base_port + 2 * i + 1) for i in range(n)]
    jobs = [(s, k) for s in systems for k in seeds]
    print(f"{len(jobs)} (system, seed) jobs over {n} workers; models slm={a.slm} llm={a.llm}")
    for i, (ps, pl) in enumerate(ports):
        mine = [j for idx, j in enumerate(jobs) if idx % n == i]
        print(f"  worker {i}: servers :{ps} (slm) :{pl} (llm), {len(mine)} jobs")
    if a.dry_run:
        return
    if not os.path.exists(LATENCY_CALIBRATION) and not a.allow_live_latency:
        raise SystemExit(f"{LATENCY_CALIBRATION} is missing: with several servers sharing the GPU, "
                         "'live' latency would absorb contention. Run scripts/calibrate_latency.py "
                         "first (or pass --allow-live-latency knowingly).")

    # 1. scenario, workloads and calibration once, in this process (no races)
    import calibrate_workload
    import scenario_build as SB
    E.Context(a.profile)                       # topology + threshold sweep
    SB.build_workloads(a.profile, seeds)
    calibrate_workload.calibrate(a.profile)

    logs = os.path.join(results_dir(a.profile), "logs")
    os.makedirs(logs, exist_ok=True)
    py = sys.executable
    from start_server import build_cmd
    servers, shards = [], []
    try:
        # 2. N server pairs, owned directly by this process
        for i, (ps, pl) in enumerate(ports):
            for tier, port in ((a.slm, ps), (a.llm, pl)):
                cmd = build_cmd(tier, "127.0.0.1", port, a.ctx)
                servers.append(subprocess.Popen(cmd, stdout=open(os.path.join(
                    logs, f"server_w{i}_{tier}.log"), "w"), stderr=subprocess.STDOUT))
        for ps, pl in ports:
            for port in (ps, pl):
                if not wait_ready(f"http://127.0.0.1:{port}"):
                    raise SystemExit(f"server on :{port} never became ready - see {logs}")
        print("all servers up")

        # 3. N shards
        for i, (ps, pl) in enumerate(ports):
            env = dict(os.environ,
                       LOCAL_LLM_URL_SLM=f"http://127.0.0.1:{ps}",
                       LOCAL_LLM_URL_LLM=f"http://127.0.0.1:{pl}",
                       SLM_MODEL_LABEL=a.slm, LLM_MODEL_LABEL=a.llm)
            for k in ("LOCAL_LLM_URL", "LOCAL_LLM_URL_GOA"):
                env.pop(k, None)
            if MODELS[a.slm].get("no_system_role"):
                env["LOCAL_LLM_URL_SLM_NO_SYSTEM_ROLE"] = "1"
            if MODELS[a.llm].get("no_system_role"):
                env["LOCAL_LLM_URL_LLM_NO_SYSTEM_ROLE"] = "1"
            cmd = [py, os.path.join(BASE, "src", "evaluator.py"), "--profile", a.profile,
                   "--systems", ",".join(systems), "--seeds", ",".join(map(str, seeds)),
                   "--shard", f"{i}/{n}"] + (["--force"] if a.force else [])
            shards.append(subprocess.Popen(cmd, env=env, stdout=open(os.path.join(
                logs, f"shard{i}.log"), "w"), stderr=subprocess.STDOUT))
        print(f"{n} shards running - follow progress in {logs}/shard*.log")
        codes = [p.wait() for p in shards]
        print("shard exit codes:", codes)
    finally:
        for p in shards:
            if p.poll() is None:
                p.terminate()
        for p in servers:
            p.terminate()
        for p in servers:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()

    # 4. one report over everything that finished
    subprocess.run([py, os.path.join(BASE, "src", "evaluator.py"), "--profile", a.profile,
                    "--systems", ",".join(systems), "--seeds", ",".join(map(str, seeds)),
                    "--report-only"], check=False)


if __name__ == "__main__":
    main()
