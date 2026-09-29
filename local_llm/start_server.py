#!/usr/bin/env python3
"""Start an OpenAI-compatible local LLM server (llama_cpp.server).

Auto-detects the environment and picks sane defaults:
  - GPU present (nvidia-smi works)  -> offload all layers to GPU
  - CPU only                        -> thread count from the cgroup CPU
                                        quota (not os.cpu_count(), which
                                        can lie in containers), minus a
                                        couple cores of headroom.

Usage:
    python3 start_server.py [small|medium|large] [--port 8080]

    # two concurrent servers - a small model for SLM (ZOA intent
    # compilation) and a larger one for LLM role (global agent decisions
    # reasoning) - so MultiLLM(role="slm")/MultiLLM(role="llm") can each
    # be pointed at a differently-sized model (see ../src/llm_client.py
    # discover_providers(), LOCAL_LLM_URL_SLM / LOCAL_LLM_URL_LLM):
    python3 start_server.py llama32_3b --port 8080 \\
        --also mistral7b --also-port 8081

Downloads the model(s) first if not already present (delegates to
download_model.py).
"""
import argparse
import os
import shutil
import subprocess
import sys

from download_model import main as download_model
from models import DEFAULT_TIER, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))


def has_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], capture_output=True).returncode == 0


def cgroup_cpu_quota():
    """Return the effective core count from the cgroup CPU quota, if any.
    Falls back to os.cpu_count() when unconstrained/unavailable — this
    matters because os.cpu_count()/nproc can report the host's full core
    count even inside a container throttled to a fraction of it."""
    for path in ("/sys/fs/cgroup/cpu.max",):
        if os.path.exists(path):
            try:
                quota, period = open(path).read().split()
                if quota != "max":
                    return max(1, int(quota) // int(period))
            except Exception:
                pass
    for quota_path, period_path in [
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
         "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ]:
        if os.path.exists(quota_path):
            try:
                quota = int(open(quota_path).read())
                period = int(open(period_path).read())
                if quota > 0:
                    return max(1, quota // period)
            except Exception:
                pass
    return os.cpu_count() or 4


def build_cmd(tier, host, port, ctx):
    model_path = download_model_for(tier)
    chat_format = MODELS[tier].get("chat_format")

    gpu = has_gpu()
    cmd = [sys.executable, "-m", "llama_cpp.server",
           "--model", model_path,
           "--n_ctx", str(ctx),
           "--n_batch", "512",
           "--host", host,
           "--port", str(port)]
    if chat_format:
        cmd += ["--chat_format", chat_format]
    else:
        print(f"  {tier}: no chat_format override — using the model's own "
              f"embedded GGUF chat template")

    if gpu:
        print(f"  {tier}: GPU detected — offloading all layers to GPU")
        cmd += ["--n_gpu_layers", "-1"]
    else:
        quota = cgroup_cpu_quota()
        n_threads = max(1, quota - 2)  # leave headroom for the caller
        print(f"  {tier}: no GPU — CPU mode, cgroup quota ~{quota} cores, "
              f"using n_threads={n_threads}")
        cmd += ["--n_gpu_layers", "0", "--n_threads", str(n_threads)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier", nargs="?", default=DEFAULT_TIER,
                     choices=list(MODELS))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--also", default=None, choices=list(MODELS),
                    help="Launch a second server concurrently (e.g. a "
                         "small SLM-role model alongside a larger "
                         "GOA-role one) instead of exec-replacing this "
                         "process with a single server.")
    ap.add_argument("--also-port", type=int, default=None,
                    help="Port for --also's server (default: --port + 1)")
    args = ap.parse_args()

    if not args.also:
        # Single-server case (the original, default behavior): exec-
        # replace this process so signals/exit codes pass through
        # untouched, same as before this feature existed.
        cmd = build_cmd(args.tier, args.host, args.port, args.ctx)
        print("NOTE: on a shared/contended host, CPU inference of even "
              "the 'small' tier can be impractically slow — see "
              "README.md before relying on this for a real run.")
        print("launching:", " ".join(cmd))
        os.execvp(cmd[0], cmd)
        return

    also_port = args.also_port or (args.port + 1)
    cmd1 = build_cmd(args.tier, args.host, args.port, args.ctx)
    cmd2 = build_cmd(args.also, args.host, also_port, args.ctx)
    print("launching two servers:")
    print(f"  [{args.tier}] :{args.port} ->", " ".join(cmd1))
    print(f"  [{args.also}] :{also_port} ->", " ".join(cmd2))
    print(f"point the pipeline at them with:\n"
          f"  export LOCAL_LLM_URL_SLM=http://{args.host}:{args.port}\n"
          f"  export LOCAL_LLM_URL_LLM=http://{args.host}:{also_port}\n"
          f"(role assignment is just a suggestion here — pick whichever "
          f"port matches the smaller/larger of the two tiers you chose)")
    procs = [subprocess.Popen(cmd1), subprocess.Popen(cmd2)]
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


def download_model_for(tier):
    spec = MODELS[tier]
    target = os.path.join(HERE, spec["filename"])
    if not os.path.exists(target):
        sys.argv = ["download_model.py", tier]
        download_model()
    return target


if __name__ == "__main__":
    main()
