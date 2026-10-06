#!/usr/bin/env python3
"""Phase 9 driver - RUN ON THE GPU MACHINE. For each model tier:
  1. start its llama.cpp server (local_llm/start_server.py), wait until ready
  2. model_compare.py on it (translation + 100 reasoning cases + latency/tokens)
  3. scripts/calibrate_latency.py on it (merged into src/latency_distributions.json)
  4. one full end-to-end evaluation (all systems x all seeds of --profile) with
     that model in BOTH roles, into results/<profile>__model_<tier>/
  5. stop the server
then (optionally) the hosted reference goes through step 2 only - it is an
upper-bound quality reference, not a deployment candidate.

    python scripts/run_model_sweep.py --tiers small,medium,large,llama32_1b,llama32_3b,phi35_mini,mistral7b,gemma2_9b
    python scripts/run_model_sweep.py --tiers medium --profile small --systems full,react
    python scripts/run_model_sweep.py --tiers medium --skip-eval        # probes only

If GPU memory forces dropping a tier, drop it here and record the cut in
ARCHITECTURE.md ("Model comparison").

The server is started directly (start_server.build_cmd), not through the
start_server.py launcher: on Windows, terminating the launcher left the real
server running, so every later tier was silently measured on the first tier's
model. The sweep now refuses a port that is already serving, checks that the
served model file is the tier's own before measuring anything, and waits for
the port to close after each tier.
"""
import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import config  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "local_llm"))
from models import MODELS  # noqa: E402
from start_server import build_cmd  # noqa: E402


def wait_ready(url, timeout=900):
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


def answering(url):
    import requests
    try:
        return requests.get(url + "/v1/models", timeout=3).ok
    except Exception:
        return False


def served_ids(url):
    import requests
    try:
        return [str(m.get("id")) for m in requests.get(url + "/v1/models", timeout=10)
                .json().get("data", [])]
    except Exception:
        return []


def stop(srv, url):
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        srv.kill()
        srv.wait(timeout=30)
    for _ in range(60):
        if not answering(url):
            return
        time.sleep(1)
    raise SystemExit(f"{url} still answers after stopping the server - stop the stray "
                     "llama_cpp.server process before continuing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default=",".join(MODELS))
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--hosted-ref", default="hosted_ref_llama70b")
    a = ap.parse_args()
    py = sys.executable
    url = f"http://127.0.0.1:{a.port}"
    for tier in [t for t in a.tiers.split(",") if t]:
        print(f"\n########## {tier} ##########", flush=True)
        if answering(url):
            raise SystemExit(f"{url} is already serving {served_ids(url)} - stop that server "
                             "first (a stray server would be measured in place of this tier)")
        srv = subprocess.Popen(build_cmd(tier, "127.0.0.1", a.port, 4096),
                               stdout=open(f"server_{tier}.log", "w"), stderr=subprocess.STDOUT)
        try:
            if not wait_ready(url):
                print(f"{tier}: server never became ready - see server_{tier}.log; skipping")
                continue
            want = os.path.basename(MODELS[tier]["filename"])
            ids = served_ids(url)
            if not any(os.path.basename(i.replace("\\", "/")) == want for i in ids):
                print(f"{tier}: the server reports {ids}, not {want} - skipping this tier")
                continue
            print(f"{tier}: serving {want}", flush=True)
            subprocess.run([py, os.path.join(BASE, "local_llm", "model_compare.py"),
                            "--target", f"{tier}@{url}"], check=False)
            if not a.skip_latency:
                subprocess.run([py, os.path.join(BASE, "scripts", "calibrate_latency.py"),
                                "--target", f"{tier}@{url}"], check=False)
            if not a.skip_eval:
                env = dict(os.environ, LOCAL_LLM_URL=url, SLM_MODEL_LABEL=tier, LLM_MODEL_LABEL=tier,
                           RESULTS_TAG=f"model_{tier}")
                if MODELS[tier].get("no_system_role"):
                    env["LOCAL_LLM_URL_NO_SYSTEM_ROLE"] = "1"
                for k in ("LOCAL_LLM_URL_SLM", "LOCAL_LLM_URL_LLM", "LOCAL_LLM_URL_GOA"):
                    env.pop(k, None)
                cmd = [py, os.path.join(BASE, "main.py"), "--profile", a.profile]
                if a.systems:
                    cmd += ["--systems", a.systems]
                if a.seeds:
                    cmd += ["--seeds", a.seeds]
                subprocess.run(cmd, env=env, check=False)
        finally:
            stop(srv, url)
    ref = config.HOSTED_REFERENCE.get(a.hosted_ref)
    if ref and os.environ.get(ref["provider_env"]):
        subprocess.run([py, os.path.join(BASE, "local_llm", "model_compare.py"),
                        "--target", f"{a.hosted_ref}@hosted"], check=False)
    else:
        print(f"hosted reference skipped ({ref['provider_env'] if ref else a.hosted_ref} not set)")


if __name__ == "__main__":
    main()
