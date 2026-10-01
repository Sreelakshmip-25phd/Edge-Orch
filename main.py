#!/usr/bin/env python3
"""Full pipeline, phase by phase, with checkpoint auto-detection (an
interrupted run resumes; finished phases are skipped).

    python main.py --profile smoke     # synthetic data + MOCK LLM, minutes, no GPU:
                                       # proves the pipeline end to end (NOT results)
    python main.py --profile small     # real data + real LLM, reduced size (GPU machine)
    python main.py                     # full campaign: 12k req/day x 10 seeds x all systems
    python main.py --from 5 --force    # re-run evaluation onwards
    python main.py --systems full,react --seeds 0,1

Phases:
  1  data foundation      real datasets (skipped for smoke)
  2  scenario             topology, catalog, threshold sweep
  3  workload             non-stationary workload per seed + load calibration (K)
  4  latency calibration  status check only - the measurement itself is
                          scripts/calibrate_latency.py on the GPU machine
  5  evaluation           all systems x all seeds
  6  report               tables, paired tests, figures
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402
from config import LATENCY_CALIBRATION, PROFILES, VALIDATED, results_dir  # noqa: E402

sys.path.insert(0, config.SRC)


def done_data(profile):
    if PROFILES[profile]["data"] == "synthetic":
        return True
    need = ["milan_grid_centroids.csv", "milan_activity.parquet", "milan_surge_candidates.csv",
            "service_resource_profiles.json", "failure_model.json", "rtt_model.json"]
    return all(os.path.exists(os.path.join(VALIDATED, f)) for f in need)


def done_scenario(profile):
    d = os.path.join(results_dir(profile), "scenario")
    return all(os.path.exists(os.path.join(d, f)) for f in
               ("topology.json", "activity.npz", "threshold_sweep.json"))


def done_workload(profile, seeds):
    import scenario_build as SB
    return SB.calibration(profile) is not None and all(
        os.path.exists(SB.workload_path(profile, s)) for s in seeds)


def done_eval(profile, systems, seeds):
    return all(os.path.exists(os.path.join(results_dir(profile), "runs", s, f"seed{k}.metrics.json"))
               for s in systems for k in seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=config.DEFAULT_PROFILE, choices=list(PROFILES))
    ap.add_argument("--from", dest="from_phase", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-run phases >= --from even if done")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--import-data-from", default=None,
                    help="existing checkout whose data_foundation/validated/ to copy")
    a = ap.parse_args()
    prof = a.profile
    cfg = PROFILES[prof]
    config.ensure_dirs(prof)
    import evaluator as E
    systems = E.expand_systems(a.systems) if a.systems else E.ALL_SYSTEMS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else cfg["seeds"]

    print(f"profile={prof} data={cfg['data']} llm={cfg['llm']} "
          f"requests={cfg['n_requests']} seeds={seeds} systems={len(systems)}")
    if cfg["llm"] == "mock":
        print("NOTE: smoke profile uses the MOCK LLM and placeholder latencies - "
              "its outputs demonstrate the pipeline, they are NOT results.")
    else:
        from llm_client import discover_providers
        if not discover_providers():
            sys.exit("No LLM provider configured. Start one (local_llm/README.md) and set "
                     "LOCAL_LLM_URL (or LOCAL_LLM_URL_SLM + LOCAL_LLM_URL_LLM), or run "
                     "`python main.py --profile smoke` to exercise the pipeline without a model.")

    def phase(n, label, done, fn):
        print("\n" + "=" * 70 + f"\nPhase {n}: {label}\n" + "=" * 70, flush=True)
        if n < a.from_phase:
            print("skip (--from)")
            return
        if done() and not (a.force and n >= a.from_phase):
            print("skip (checkpoint reached)")
            return
        fn()

    def p1():
        import data_foundation
        ok = data_foundation.main(["--import-from", a.import_data_from] if a.import_data_from else [])
        if not ok:
            sys.exit("data foundation incomplete - see messages above")

    def p2():
        import scenario_build as SB
        ctx = E.Context(prof)
        sw = SB.threshold(prof, ctx.embedder)
        print(f"topology: {ctx.topo['n_zones']} zones / {ctx.topo['n_nodes']} nodes; "
              f"cache threshold {sw['chosen']} (embedder {sw['backend']})")

    def p3():
        import calibrate_workload
        import scenario_build as SB
        SB.build_workloads(prof, seeds, force=a.force)
        calibrate_workload.calibrate(prof, force=a.force)

    def p4():
        import scenario_build as SB
        st = SB.latency_status()
        if cfg["llm"] == "mock":
            print("smoke: placeholder latencies (mock LLM)")
        elif st["present"]:
            print(f"calibrated LLM/SLM latency distributions: {LATENCY_CALIBRATION}")
        else:
            print("latency_distributions.json NOT present - LLM/SLM latency will be charged from "
                  "each call's own measured wall time ('live'). For host-independent numbers run "
                  "scripts/calibrate_latency.py on the GPU machine first.")

    phase(1, "data foundation", lambda: done_data(prof), p1)
    phase(2, "scenario (topology, catalog, threshold sweep)", lambda: done_scenario(prof), p2)
    phase(3, "workload generation + load calibration", lambda: done_workload(prof, seeds), p3)
    phase(4, "latency calibration status", lambda: False, p4)
    phase(5, "evaluation (systems x seeds)", lambda: done_eval(prof, systems, seeds),
          lambda: E.run_matrix(prof, systems, seeds, force=a.force))
    phase(6, "report (tables, tests, figures)", lambda: False,
          lambda: E.report(prof, systems, seeds))
    print(f"\nDone. Results in {results_dir(prof)}/")


if __name__ == "__main__":
    main()
