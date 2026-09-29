#!/usr/bin/env python3
"""Write data_foundation/validated/device_calibration.json: the node
capacity table (from src/device_specs.py, vendor datasheets) plus,
optionally, a MEASURED container start-up time for the deployment latency
component.

    python scripts/generate_device_calibration.py
    python scripts/generate_device_calibration.py --measure-container-start \\
        --device-class rack_edge_server --runs 20

--measure-container-start times `docker run --rm <image> true` on THIS
host and records it for the device class you name - only name the class
this host actually is (a server-class GPU box is `rack_edge_server`; run
it on a real Pi / Jetson / Coral to calibrate those). Classes you don't
measure keep the assumption stated in device_specs.py, flagged as such.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import config  # noqa: E402
from config import DEVICE_CALIBRATION  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "src"))
from device_specs import BENCHMARK_METHODOLOGY_REFS, DEVICE_CLASSES, as_table  # noqa: E402


def measure(image, runs):
    if not shutil.which("docker"):
        raise SystemExit("docker not found - cannot measure container start-up here")
    subprocess.run(["docker", "pull", image], check=True, capture_output=True)
    subprocess.run(["docker", "run", "--rm", image, "true"], check=True, capture_output=True)  # warm
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(["docker", "run", "--rm", image, "true"], check=True, capture_output=True)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-container-start", action="store_true")
    ap.add_argument("--device-class", choices=list(DEVICE_CLASSES), default=None)
    ap.add_argument("--image", default="alpine:3.19")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args(argv)
    cal = json.load(open(DEVICE_CALIBRATION)) if os.path.exists(DEVICE_CALIBRATION) else {}
    cal["device_table"] = as_table()
    cal["benchmark_methodology_refs"] = BENCHMARK_METHODOLOGY_REFS
    cal.setdefault("devices", {})
    if a.measure_container_start:
        if not a.device_class:
            raise SystemExit("--device-class is required with --measure-container-start")
        import numpy as np
        xs = measure(a.image, a.runs)
        med = float(np.median(xs))
        sigma = float(np.std(np.log(xs)))
        cal["devices"][a.device_class] = {
            "deploy_ms_median": round(med, 1), "deploy_ms_sigma": round(max(sigma, 0.05), 3),
            "samples_ms": [round(x, 1) for x in xs], "image": a.image,
            "measured_on": f"{platform.node()} ({platform.machine()})"}
        print(f"{a.device_class}: container start median {med:.0f} ms over {a.runs} runs")
    os.makedirs(os.path.dirname(DEVICE_CALIBRATION), exist_ok=True)
    json.dump(cal, open(DEVICE_CALIBRATION, "w"), indent=1)
    print(f"wrote {DEVICE_CALIBRATION}")


if __name__ == "__main__":
    main()
