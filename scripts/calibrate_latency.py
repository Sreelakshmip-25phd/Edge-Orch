#!/usr/bin/env python3
"""Phase 5 measurement - RUN THIS ON THE GPU MACHINE.

Measures the real latency distribution of each local model on a fixed,
representative prompt set, per prompt kind, and writes the empirical
samples to src/latency_distributions.json, which latency_model.py then
samples from ("calibrated" mode). Nothing is estimated or made up here:
every sample is the wall time of one real, uncached call.

    python local_llm/start_server.py llama32_3b --port 8080 --also medium --also-port 8081
    python scripts/calibrate_latency.py \\
        --target llama32_3b@http://127.0.0.1:8080 \\
        --target medium@http://127.0.0.1:8081 --repeats 3
    git add src/latency_distributions.json && git commit -m "Measured LLM/SLM latency on <GPU>"

Prompt kinds measured (the ones whose sizes differ materially):
  translate    SLM intent translation (zone agent), held-out phrasings
  zone_choice  CORE baseline's SLM zone pick
  decide       global agent's one-call structured decision (reasoning cases)
  rule_author  procedural-rule authoring
  react_step   a ReAct step with a two-step history (also stands in for the
               LATS / AgentEdge prompts, see latency_model._KIND_FALLBACK)
Measure every model you will run in either role; the file is merged, so
models can be measured in batches. The evaluation picks the entries named
by SLM_MODEL_LABEL / LLM_MODEL_LABEL (config.py).
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
import config  # noqa: E402
from config import LATENCY_CALIBRATION  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "local_llm"))
sys.path.insert(0, os.path.join(BASE, "src"))
from global_agent import DECIDE_SYS, RULE_SYS  # noqa: E402
from baselines.react_baseline import REACT_SYS  # noqa: E402
from baselines.core_baseline import ZONE_SYS  # noqa: E402
from llm_client import MultiLLM  # noqa: E402
from model_compare import provider_for, served_model, translation_set  # noqa: E402
from reasoning_cases import generate as gen_cases  # noqa: E402
from scenario import ServiceCatalog  # noqa: E402
from zone_agent import slm_system_prompt  # noqa: E402

WARMUP = 3


def prompt_set(n):
    cat = ServiceCatalog()
    slm_sys = slm_system_prompt(cat)
    items = translation_set()
    step = max(len(items) // n, 1)
    out = {"translate": [(slm_sys, "INPUT: " + t) for _, t in items[::step][:n]]}
    cases = gen_cases()
    out["decide"] = [(DECIDE_SYS, c["user"]) for c in cases[::max(len(cases) // n, 1)][:n]]
    out["rule_author"] = [(RULE_SYS, json.dumps({"service_type": st, "successful_zone_counts":
                                                 {"z1": 5 + i, "z4": 2, "z7": i % 3}}))
                          for i, st in enumerate(list(cat.names()) * 3)][:max(n // 3, 5)]
    zones = [{"zone": f"z{i}", "rtt_ms": 4.0 + i * 0.5} for i in range(10)]
    hist = [{"thought": "check the origin zone", "action": {"tool": "read_digest", "args": {"zone": "z0"}},
             "observation": {"zone": "z0", "nodes": [{"node": "n0", "class": "raspberry_pi_4",
                                                      "cpu_free": 0.4, "mem_free": 1.0, "accel": None}]}},
            {"thought": "try the nearest neighbour", "action": {"tool": "read_digest", "args": {"zone": "z3"}},
             "observation": {"zone": "z3", "nodes": [{"node": "n9", "class": "jetson_orin_nano",
                                                      "cpu_free": 4.2, "mem_free": 5.0, "accel": "gpu"}]}}]
    out["react_step"] = [(REACT_SYS, json.dumps({"request": {"intent": t, "origin_zone": "z0",
                                                             "zones": zones, "catalog": cat.prompt_block()},
                                                 "history": hist})) for _, t in items[:max(n // 2, 5)]]
    out["zone_choice"] = [(ZONE_SYS, json.dumps({"request": {"profile": {"service_type": "traffic_monitor"},
                                                             "cpu": 1.0 + i / 10, "mem": 0.5},
                                                 "zones": {z["zone"]: {"rtt_ms": z["rtt_ms"], "fits": i % 3 == 0,
                                                                       "top_cpu_free": 2.0}
                                                           for z in zones}}))
                          for i in range(max(n // 3, 5))]
    return out


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       timeout=10).decode().strip()
    except Exception:
        return None


def measure(label, url, n, repeats):
    prov = provider_for(label, url)
    kinds = {}
    for kind, prompts in prompt_set(n).items():
        role = "slm" if kind in ("translate", "zone_choice") else "llm"
        m = MultiLLM(None, role, model_label=label, providers=[prov])   # no cache: every call is real
        samples, tin, tout = [], [], []
        for rep in range(repeats):
            for i, (s, u) in enumerate(prompts):
                # vary the prompt trivially per repeat so server-side prompt caching
                # can't turn repeats into near-zero measurements
                r = m.ask(s, u + ("" if rep == 0 else f"\n(request {rep}-{i})"), kind=kind)
                if rep == 0 and i < WARMUP:
                    continue
                samples.append(round(r.wall_ms, 2))
                tin.append(r.tokens_in)
                tout.append(r.tokens_out)
        import numpy as np
        kinds[kind] = {"samples_ms": samples, "n": len(samples),
                       "mean_ms": round(float(np.mean(samples)), 2),
                       "p50_ms": round(float(np.percentile(samples, 50)), 2),
                       "p95_ms": round(float(np.percentile(samples, 95)), 2),
                       "tokens_in_mean": round(float(np.mean(tin)), 1),
                       "tokens_out_mean": round(float(np.mean(tout)), 1)}
        print(f"  {label} {kind:12s} n={len(samples)} mean={kinds[kind]['mean_ms']}ms "
              f"p95={kinds[kind]['p95_ms']}ms", flush=True)
    return kinds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", required=True, help="label@url")
    ap.add_argument("--n", type=int, default=40, help="prompts per kind")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=LATENCY_CALIBRATION)
    a = ap.parse_args(argv)
    cal = json.load(open(a.out)) if os.path.exists(a.out) else {"models": {}}
    cal.setdefault("models", {})
    for spec in a.target:
        label, url = spec.split("@", 1)
        print(f"=== {label} @ {url} ===")
        cal["models"][label] = {
            "kinds": measure(label, url, a.n, a.repeats),
            "meta": {"served_model": served_model(url) if url != "hosted" else "hosted",
                     "gpu": gpu_name(), "host": platform.node(), "python": platform.python_version(),
                     "measured_utc": datetime.datetime.utcnow().isoformat() + "Z",
                     "n_per_kind": a.n, "repeats": a.repeats, "warmup_dropped": WARMUP,
                     "concurrency": 1}}
    cal["meta"] = {"description": "Empirical per-call wall-time samples (ms) of real, uncached "
                                  "SLM/LLM calls, measured by scripts/calibrate_latency.py",
                   "models": sorted(cal["models"])}
    json.dump(cal, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} - commit it so every evaluation run samples the same distributions")


if __name__ == "__main__":
    main()
