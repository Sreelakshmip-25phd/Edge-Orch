"""Build (or load) everything a profile's runs share: topology, activity
matrices, threshold sweep, and one workload per seed. Idempotent: every
artifact under results/<profile>/scenario/ is reused if present."""
import json
import os

import numpy as np

from config import (LATENCY_CALIBRATION, PROFILES, TOPOLOGY_SEED, VALIDATED,
                    results_dir)
import workload as W
from scenario import (PLACES_INITIAL, SYNTHETIC_RESOURCE_PROFILES, ResourceModel,
                      build_topology_from_cells, load_intent_pools, milan_zone_activity,
                      synthetic_topology, synthetic_zone_activity)


def scen_dir(profile):
    d = os.path.join(results_dir(profile), "scenario")
    os.makedirs(d, exist_ok=True)
    return d


def _real_cells():
    import pandas as pd
    grid = pd.read_csv(os.path.join(VALIDATED, "milan_grid_centroids.csv"))
    act = pd.read_parquet(os.path.join(VALIDATED, "milan_activity.parquet"),
                          columns=["cell_id", "activity"])
    w = act.groupby("cell_id")["activity"].mean().rename("w").reset_index()
    return grid.merge(w, on="cell_id", how="left").fillna({"w": 0.0})


def resources_for(profile):
    if PROFILES[profile]["data"] == "synthetic":
        return ResourceModel(SYNTHETIC_RESOURCE_PROFILES)
    return ResourceModel.load()


def failure_model_for(profile):
    from data_foundation import SYNTHETIC_FAILURE_MODEL, FailureModel
    if PROFILES[profile]["data"] == "synthetic":
        return FailureModel(SYNTHETIC_FAILURE_MODEL)
    return FailureModel.load()


def topology_and_activity(profile):
    d = scen_dir(profile)
    tp, ap = os.path.join(d, "topology.json"), os.path.join(d, "activity.npz")
    if os.path.exists(tp) and os.path.exists(ap):
        topo = json.load(open(tp))
        z = np.load(ap, allow_pickle=True)
        return topo, {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    cfg = PROFILES[profile]
    if cfg["data"] == "synthetic":
        topo = synthetic_topology(cfg["n_zones"], TOPOLOGY_SEED)
        act = synthetic_zone_activity(cfg["n_zones"], seed=TOPOLOGY_SEED)
    else:
        rttm = json.load(open(os.path.join(VALIDATED, "rtt_model.json")))
        cells = _real_cells()
        topo, lab = build_topology_from_cells(cells, rttm, cfg["n_zones"], "eval")
        act = milan_zone_activity(lab, cells)
    json.dump(topo, open(tp, "w"), indent=1)
    np.savez(ap, **{k: np.asarray(v) for k, v in act.items()})
    return topo, act


def threshold(profile, embedder):
    from zone_agent import threshold_sweep
    p = os.path.join(scen_dir(profile), "threshold_sweep.json")
    if os.path.exists(p):
        sw = json.load(open(p))
        if sw.get("backend") == embedder.backend:
            return sw
    tr, ht, hf = load_intent_pools()
    sw = threshold_sweep(embedder, tr, ht, hf, PLACES_INITIAL)
    json.dump(sw, open(p, "w"), indent=1)
    return sw


def workload_path(profile, seed):
    return os.path.join(scen_dir(profile), f"workload_seed{seed}.json.gz")


def build_workloads(profile, seeds=None, force=False):
    cfg = PROFILES[profile]
    topo, act = topology_and_activity(profile)
    pools, res, fm = load_intent_pools(), resources_for(profile), failure_model_for(profile)
    out = {}
    for s in seeds if seeds is not None else cfg["seeds"]:
        p = workload_path(profile, s)
        if force or not os.path.exists(p):
            wl = W.generate(topo, act, res, fm, pools, seed=s, n_requests=cfg["n_requests"],
                            n_devices=cfg["n_devices"], horizon_s=cfg["horizon_s"],
                            min_failures=2 if cfg["data"] == "synthetic" else 3)
            W.save(wl, p)
            rep = W.drift_report(wl)
            json.dump({"meta": wl["meta"], "drift": rep},
                      open(p.replace(".json.gz", ".report.json"), "w"), indent=1, default=str)
        out[s] = p
    return out


def calibration(profile):
    p = os.path.join(scen_dir(profile), "calibration.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_workload(profile, seed):
    cal = calibration(profile)
    return W.load(workload_path(profile, seed), cal["K"] if cal else None)


def latency_status():
    return {"calibration_file": LATENCY_CALIBRATION,
            "present": os.path.exists(LATENCY_CALIBRATION)}
