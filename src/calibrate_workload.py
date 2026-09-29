#!/usr/bin/env python3
"""Load calibration (kept from the old repo, updated for the new workload).

On the raw trace-driven workload every system accepts ~everything: the
scenario is under-loaded and has no discriminative power. This searches a
lifetime scale factor K (bracket + bisection) so that the flat greedy
least-loaded baseline (greedy_oracle - no LLM, no translation cost, so the
search is cheap and model-independent) lands in the CALIBRATION_TARGET
acceptance band on seed 0. K is then applied to every seed's workload at
load time (results/<profile>/scenario/calibration.json).

Run: python src/calibrate_workload.py --profile full
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CALIBRATION_TARGET, PROFILES  # noqa: E402
import metrics as M  # noqa: E402
import scenario_build as SB  # noqa: E402
import workload as W  # noqa: E402


def greedy_acceptance(ctx, wl, k):
    from evaluator import simulate
    wl = dict(wl, meta=dict(wl["meta"], lifetime_scale=k))
    tel = simulate(ctx, "greedy_oracle", wl, seed=wl["meta"]["seed"])
    return M.compute(tel)["scalars"]["acceptance_rate"]


def calibrate(profile, seed=0, force=False, verbose=True):
    p = os.path.join(SB.scen_dir(profile), "calibration.json")
    if os.path.exists(p) and not force:
        cal = json.load(open(p))
        if verbose:
            print(f"calibration cached: K={cal['K']} (greedy acceptance {cal['greedy_acceptance_at_K']})")
        return cal
    from evaluator import Context
    SB.build_workloads(profile, [seed])
    ctx = Context(profile, need_embedder=False)
    wl = W.load(SB.workload_path(profile, seed))
    lo_t, hi_t = CALIBRATION_TARGET
    trace = []

    def f(k):
        a = greedy_acceptance(ctx, wl, k)
        trace.append({"K": round(k, 4), "greedy_acceptance": a})
        if verbose:
            print(f"  K={k:8.3f} -> greedy acceptance {a:.3f}", flush=True)
        return a

    # acceptance falls monotonically as K (load) grows
    def in_band(a):
        return lo_t <= a <= hi_t

    k, acc = 1.0, f(1.0)
    if not in_band(acc):
        if acc > hi_t:                       # under-loaded: grow K until too loaded
            lo, hi = k, k * 2
            while True:
                acc = f(hi)
                if in_band(acc) or acc < lo_t or hi >= 4096:
                    break
                lo, hi = hi, hi * 2
            k = hi
        else:                                # over-loaded: shrink K
            lo, hi = k / 2, k
            while True:
                acc = f(lo)
                if in_band(acc) or acc > hi_t or lo <= 1 / 1024:
                    break
                lo, hi = lo / 2, lo
            k = lo
        if not in_band(acc):                 # bisect between lo (too light) and hi (too heavy)
            for _ in range(16):
                k = (lo + hi) / 2
                acc = f(k)
                if in_band(acc):
                    break
                lo, hi = (k, hi) if acc > hi_t else (lo, k)
    cal = {"K": round(k, 4), "greedy_acceptance_at_K": round(acc, 4),
           "target_band": list(CALIBRATION_TARGET), "seed": seed, "trace": trace,
           "method": "lifetime scaling, bracket + bisection on greedy_oracle acceptance"}
    json.dump(cal, open(p, "w"), indent=1)
    if verbose:
        print(f"calibrated: K={cal['K']} (greedy acceptance {cal['greedy_acceptance_at_K']})")
    return cal


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    calibrate(a.profile, force=a.force)


if __name__ == "__main__":
    main()
