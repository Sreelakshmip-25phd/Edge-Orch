"""Phase 2 - topology (heterogeneous real-spec nodes), the *open* service
catalog, intent phrasing pools, and the Tier-1 mobility model.

Old-repo flaws fixed here:
  #3 closed 7-type catalog known to everyone up front -> ServiceCatalog is
     a live registry: agents only see what has been registered so far, a
     genuinely unseen type (`crowd_safety`) is registered mid-run, and a
     second one (`ev_charging`) is never registered at all, so it can only
     be handled by the SLM's free-text output + nearest-match fallback.
  #2 intent cache preloaded with every phrasing the workload uses ->
     phrasings are split into a training pool (the 3 hand-written seed
     templates per type; only these may seed a cache) and a held-out pool
     (LLM paraphrases + the OOD probe + hand-written templates of the new
     types) that the workload draws from. A cache hit is only ever earned
     by similarity or by an earlier translation of real traffic.
  Two hard-coded HW tiers -> four real device classes (device_specs.py).
"""
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INTENT_DATA, TOPOLOGY_SEED, VALIDATED  # noqa: E402
from device_specs import DEVICE_CLASSES  # noqa: E402

LATENCY_CLASSES = ("realtime", "interactive", "batch")
LOCALITIES = ("zone_local", "any")
PRIORITIES = ("low", "normal", "high", "critical")
PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}
PROFILE_FIELDS = ("service_type", "latency_class", "data_locality", "priority")


def latency_class_of(ms):
    return "realtime" if ms < 100 else ("interactive" if ms <= 500 else "batch")


# --- service types -------------------------------------------------------------
# intro_h / ramp_h drive the gradual non-stationary type-mix drift in
# workload.py (logistic ramp centred on intro_h). register_h: when the
# operator registers the type in the catalog (None = present from t=0;
# "never" = never registered). cpu_scale multiplies the Alibaba per-task
# core distribution; mem_gb_base scales the Alibaba memory distribution.
# degrade_floor: lowest fraction of requested CPU/mem the service can run
# at (the degradation menu never goes below it).
SERVICE_TYPES = {
    "traffic_monitor": dict(
        description="monitor road traffic flow and congestion at intersections",
        cpu_scale=0.5, mem_gb_base=0.5, latency_ms=200, data_locality="any",
        priority="normal", degrade_floor=0.4, accel_pref=False,
        intro_h=None, ramp_h=0, base_weight=1.0, register_h=None),
    "video_analytics": dict(
        description="real-time analytics on camera video streams, footage stays local",
        cpu_scale=2.0, mem_gb_base=2.0, latency_ms=100, data_locality="zone_local",
        priority="high", degrade_floor=0.6, accel_pref=True,
        intro_h=None, ramp_h=0, base_weight=1.3, register_h=None),
    "iot_aggregator": dict(
        description="aggregate and summarise IoT sensor telemetry in batches",
        cpu_scale=0.25, mem_gb_base=0.25, latency_ms=800, data_locality="any",
        priority="low", degrade_floor=0.4, accel_pref=False,
        intro_h=None, ramp_h=0, base_weight=1.0, register_h=None),
    "ar_session": dict(
        description="augmented-reality overlay session with ultra low latency",
        cpu_scale=1.5, mem_gb_base=1.5, latency_ms=50, data_locality="zone_local",
        priority="high", degrade_floor=0.6, accel_pref=True,
        intro_h=None, ramp_h=0, base_weight=0.7, register_h=None),
    "federated_ml": dict(
        description="federated learning training round on local data partitions",
        cpu_scale=3.0, mem_gb_base=4.0, latency_ms=900, data_locality="any",
        priority="normal", degrade_floor=0.4, accel_pref=False,
        intro_h=7.0, ramp_h=1.5, base_weight=0.8, register_h=None),
    "digital_twin": dict(
        description="keep a digital twin of infrastructure in sync and simulate load",
        cpu_scale=2.0, mem_gb_base=3.0, latency_ms=250, data_locality="any",
        priority="normal", degrade_floor=0.6, accel_pref=False,
        intro_h=9.0, ramp_h=1.5, base_weight=0.8, register_h=None),
    "drone_control": dict(
        description="real-time drone flight control loop, control stays local",
        cpu_scale=1.0, mem_gb_base=1.0, latency_ms=30, data_locality="zone_local",
        priority="critical", degrade_floor=0.8, accel_pref=False,
        intro_h=11.0, ramp_h=1.5, base_weight=0.6, register_h=None),
    # --- genuinely unseen at t=0 -------------------------------------------
    "crowd_safety": dict(
        description="crowd density monitoring and crush-risk alerting from camera feeds",
        cpu_scale=1.5, mem_gb_base=1.5, latency_ms=90, data_locality="zone_local",
        priority="critical", degrade_floor=0.6, accel_pref=True,
        intro_h=14.0, ramp_h=1.0, base_weight=0.7, register_h=14.5),
    "ev_charging": dict(
        description="coordinate electric-vehicle charging schedules across charging points",
        cpu_scale=0.5, mem_gb_base=0.5, latency_ms=400, data_locality="any",
        priority="normal", degrade_floor=0.4, accel_pref=False,
        intro_h=18.0, ramp_h=1.0, base_weight=0.4, register_h="never"),
}
CORE_TYPES = [k for k, v in SERVICE_TYPES.items() if v["register_h"] is None]

# Hand-written templates for the two new types (no LLM available when this
# repo was written; scripts/generate_intent_paraphrases.py can expand them
# on the GPU machine). All held-out: nothing may seed a cache with them.
NEW_TYPE_TEMPLATES = {
    "crowd_safety": [
        "Watch crowd density at the {place} gates and alert if it gets dangerous",
        "Detect crush risk in the {place} crowd from the camera feeds within 90 ms",
        "Keep an eye on how packed the {place} concourse is and warn stewards early",
        "Run crowd-safety monitoring over {place}, video must not leave the area",
        "Alert me when people density near {place} passes the safe limit",
        "Count people flowing through {place} exits and flag overcrowding in real time",
    ],
    "ev_charging": [
        "Schedule the electric car chargers at {place} to avoid peak load",
        "Balance charging sessions across the {place} EV charging points",
        "Coordinate which vehicles charge first at the {place} charging hub",
        "Plan tonight's EV charging queue for the {place} car park",
        "Spread the electric vehicle charging demand around {place} over the afternoon",
    ],
}

PLACES_INITIAL = ["stadium", "central station", "old town", "market square",
                  "university campus", "harbour district", "tech park",
                  "exhibition centre", "north ring", "riverside"]
PLACES_LATER = ["airport link", "cathedral square", "science museum",
                "east docks", "hospital quarter", "business district",
                "canal side", "fairground", "south terminal", "lakefront"]


def ground_truth_profile(stype, spec=None):
    s = spec or SERVICE_TYPES[stype]
    return {"service_type": stype,
            "latency_class": latency_class_of(s["latency_ms"]),
            "data_locality": s["data_locality"],
            "priority": s["priority"]}


class ServiceCatalog:
    """Live, open service registry. Agents hold a reference to one
    instance; `register()` (fired by a workload event) makes a new type
    visible to every agent from that moment on. Nothing else is known in
    advance."""

    def __init__(self, initial=None):
        self.types = {}
        self.registered_at = {}
        for k in (initial if initial is not None else CORE_TYPES):
            self.register(k, SERVICE_TYPES[k], t=0.0)
        self._label_vecs = None
        self._listeners = []

    def register(self, name, spec, t):
        self.types[name] = dict(spec)
        self.registered_at[name] = float(t)
        self._label_vecs = None
        for cb in getattr(self, "_listeners", []):
            cb(name)

    def on_register(self, cb):
        self._listeners.append(cb)

    def known(self, name):
        return name in self.types

    def names(self):
        return sorted(self.types)

    def profile(self, name):
        return ground_truth_profile(name, self.types[name])

    def degrade_floor(self, name):
        return self.types.get(name, {}).get("degrade_floor", 0.6)

    def prompt_block(self):
        """Compact catalog description used in SLM/LLM prompts."""
        return {n: {"desc": s["description"],
                    "latency_class": latency_class_of(s["latency_ms"]),
                    "data_locality": s["data_locality"],
                    "priority": s["priority"]}
                for n, s in sorted(self.types.items())}

    # --- open-vocabulary fallback ------------------------------------------
    def resolve(self, label, embedder, threshold=0.55):
        """Map a free-text service_type from the SLM to a registered type:
        exact label -> 'exact'; else nearest registered type by embedding
        similarity (label vs. 'name: description') if >= threshold ->
        'nearest'; else 'novel' (caller treats it as an unregistered type
        with a generic manifest). Returns (type_or_None, sim, kind)."""
        if not label:
            return None, 0.0, "novel"
        norm = str(label).strip().lower().replace(" ", "_").replace("-", "_")
        if norm in self.types:
            return norm, 1.0, "exact"
        names = self.names()
        if self._label_vecs is None or self._label_vecs[0] != names:
            texts = [f"{n.replace('_', ' ')}: {self.types[n]['description']}"
                     for n in names]
            self._label_vecs = (names, embedder.encode(texts))
        q = embedder.encode([str(label).replace("_", " ")])[0]
        sims = self._label_vecs[1] @ q
        i = int(np.argmax(sims))
        if sims[i] >= threshold:
            return names[i], float(sims[i]), "nearest"
        return None, float(sims[i]), "novel"


# --- resource sizing (Alibaba cluster-trace-v2018 distributions) ----------
PCT_X = [1, 5, 25, 50, 75, 95, 99]


def inv_cdf(pct_dict):
    ys = [float(pct_dict[str(p)] if str(p) in pct_dict else pct_dict[p])
          for p in PCT_X]
    return lambda u: float(np.interp(u, PCT_X, ys))


# Shape-compatible stand-in used ONLY by smoke runs/tests when the real
# Alibaba-derived profile file isn't present. Illustrative, not data.
SYNTHETIC_RESOURCE_PROFILES = {
    "source": "SYNTHETIC (smoke/tests only)",
    "cpu_cores_pct": {"1": 0.1, "5": 0.2, "25": 0.5, "50": 1.0, "75": 1.0, "95": 2.0, "99": 4.0},
    "mem_norm_pct": {"1": 0.03, "5": 0.05, "25": 0.2, "50": 0.3, "75": 0.4, "95": 0.6, "99": 1.5},
    "lifetime_s_pct": {"1": 1.0, "5": 2.0, "25": 5.0, "50": 10.0, "75": 45.0, "95": 300.0, "99": 900.0}}


class ResourceModel:
    def __init__(self, profiles):
        self.cpu = inv_cdf(profiles["cpu_cores_pct"])
        self.mem = inv_cdf(profiles["mem_norm_pct"])
        self.life = inv_cdf(profiles["lifetime_s_pct"])
        self.source = profiles.get("source", "")

    @classmethod
    def load(cls, path=None):
        return cls(json.load(open(path or os.path.join(
            VALIDATED, "service_resource_profiles.json"))))

    def size(self, spec, rng):
        cpu = float(np.clip(self.cpu(rng.uniform(1, 99)) * spec["cpu_scale"], 0.1, 16.0))
        mem = float(np.clip(spec["mem_gb_base"] * (0.6 + self.mem(rng.uniform(1, 99))),
                            0.1, 32.0))
        return round(cpu, 2), round(mem, 2)

    def median_size(self, spec):
        return (round(float(np.clip(self.cpu(50) * spec["cpu_scale"], 0.1, 16.0)), 2),
                round(float(np.clip(spec["mem_gb_base"] * (0.6 + self.mem(50)), 0.1, 32.0)), 2))

    def lifetime(self, rng, k=1.0):
        return float(self.life(rng.uniform(1, 99))) * k


# --- intent phrasing pools -----------------------------------------------------
def load_intent_pools():
    """Returns (train_pool, heldout_templates, heldout_fixed):
    train_pool[type]        - seed templates (only these may seed a cache)
    heldout_templates[type] - {place} templates the workload draws from
    heldout_fixed[type]     - complete OOD sentences (no {place}), used
                              in the later part of the day
    For the 7 core types the first 3 templates of
    intent_templates_expanded.json are the hand-written seeds (see the old
    repo's SEED_INTENT_TEMPLATES) and the other 9 are LLM paraphrases."""
    exp = json.load(open(os.path.join(INTENT_DATA, "intent_templates_expanded.json")))
    ood = json.load(open(os.path.join(INTENT_DATA, "intent_ood_probe.json")))
    train, held, fixed = {}, {}, {}
    for st in SERVICE_TYPES:
        if st in CORE_TYPES:
            train[st] = list(exp[st][:3])
            held[st] = list(exp[st][3:])
        else:
            # types unseen at t=0: every phrasing is held-out, even if the
            # paraphrase script has since expanded them
            train[st] = []
            held[st] = list(dict.fromkeys(NEW_TYPE_TEMPLATES[st] + exp.get(st, [])))
        fixed[st] = [o["text"] for o in ood if o["service_type"] == st]
    return train, held, fixed


def seed_cache_entries(train_pool, per_type, rng, types=None):
    """Tiny cache seed: `per_type` sentences per *registered* type, drawn
    only from the training pool (seed templates x initial places)."""
    out = []
    for st in (types or CORE_TYPES):
        tmpls = train_pool.get(st, [])
        for i in range(min(per_type, len(tmpls))):
            place = PLACES_INITIAL[int(rng.integers(len(PLACES_INITIAL)))]
            out.append((tmpls[i].format(place=place), ground_truth_profile(st)))
    return out


# --- topology ------------------------------------------------------------------
# Node mix for the evaluation topology (50 nodes, as in the old repo's
# topology_eval). Rack servers go to the highest-activity zones only, so
# most zones are made of small devices and cross-zone escalation matters.
NODE_MIX = {"eval": {"rack_edge_server": 4, "jetson_orin_nano": 14,
                     "raspberry_pi_4": 20, "coral_dev_board": 12},
            "dev": {"rack_edge_server": 2, "jetson_orin_nano": 7,
                    "raspberry_pi_4": 10, "coral_dev_board": 6}}


def _node(nid, cls_key):
    d = DEVICE_CLASSES[cls_key]
    return {"node_id": f"n{nid}", "device_class": cls_key, "cpu": float(d.cores),
            "mem_gb": float(d.ram_gb), "power_w": float(d.power_w),
            "accelerator": d.accelerator}


def allocate_nodes(shares, mix, rng):
    """Deterministic, activity-proportional allocation: rack servers to the
    top-activity zones, every zone gets >= 1 Pi and >= 1 accelerator node,
    the rest proportional to activity share (largest remainder)."""
    n = len(shares)
    order = list(np.argsort(-np.asarray(shares)))
    alloc = [dict() for _ in range(n)]
    for i in range(mix.get("rack_edge_server", 0)):
        z = order[i % n]
        alloc[z]["rack_edge_server"] = alloc[z].get("rack_edge_server", 0) + 1
    for cls in ("raspberry_pi_4", "jetson_orin_nano", "coral_dev_board"):
        total = mix.get(cls, 0)
        base = [0] * n
        if cls == "raspberry_pi_4":
            base = [1] * n
        elif cls == "jetson_orin_nano":
            base = [1 if z % 2 == 0 else 0 for z in range(n)]
        elif cls == "coral_dev_board":
            base = [1 if z % 2 == 1 else 0 for z in range(n)]
        rest = max(total - sum(base), 0)
        raw = np.asarray(shares) * rest
        fl = np.floor(raw).astype(int)
        rem = rest - fl.sum()
        for z in np.argsort(-(raw - fl))[:rem]:
            fl[z] += 1
        for z in range(n):
            c = base[z] + int(fl[z])
            if c:
                alloc[z][cls] = alloc[z].get(cls, 0) + c
    return alloc


def _rtt_table(centers, rttm):
    lo, hi = rttm["min_ms"], rttm["max_ms"]
    n = len(centers)
    d = {(a, b): math.dist(centers[a], centers[b])
         for a, b in itertools.combinations(range(n), 2)}
    if not d:
        return {}
    dmin, dmax = min(d.values()), max(d.values())
    return {f"z{a}-z{b}": round(lo + (hi - lo) * ((v - dmin) / max(dmax - dmin, 1e-12)), 2)
            for (a, b), v in d.items()}


def build_topology_from_cells(cells, rttm, n_zones, name, seed=TOPOLOGY_SEED):
    """KMeans over Milan grid cells weighted by mean activity (as in the
    old repo), heterogeneous real-spec nodes per zone. Returns (topo,
    cell->zone labels)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_zones, n_init=10, random_state=seed)
    lab = km.fit_predict(cells[["lon", "lat"]], sample_weight=cells.w + 1e-9)
    w = np.bincount(lab, weights=cells.w.values, minlength=n_zones)
    shares = w / max(w.sum(), 1e-9)
    topo = assemble_topology(name, km.cluster_centers_, shares, rttm,
                             NODE_MIX.get(name, NODE_MIX["eval"]), seed)
    topo["provenance"].update({"cells": "OpenData Milano grid (Telecom Italia)",
                               "weights": "Milan activity trace (Harvard Dataverse)"})
    return topo, lab


def assemble_topology(name, centers, shares, rttm, mix, seed=TOPOLOGY_SEED):
    rng = np.random.default_rng(seed)
    alloc = allocate_nodes(shares, mix, rng)
    zones, nid = [], 0
    lo_c, hi_c = rttm.get("zone_to_cloud_ms", [20, 40])
    for z in range(len(shares)):
        nodes = []
        for cls in ("rack_edge_server", "jetson_orin_nano", "coral_dev_board",
                    "raspberry_pi_4"):
            for _ in range(alloc[z].get(cls, 0)):
                nodes.append(_node(nid, cls))
                nid += 1
        zones.append({"zone_id": f"z{z}",
                      "centroid": [float(centers[z][0]), float(centers[z][1])],
                      "activity_share": round(float(shares[z]), 4),
                      # zone agent <-> global agent RTT (the global agent is
                      # hosted in the regional cloud tier)
                      "zone_to_global_ms": round(float(rng.uniform(lo_c, hi_c)), 2),
                      "nodes": nodes})
    return {"name": name, "n_zones": len(zones),
            "n_nodes": sum(len(z["nodes"]) for z in zones),
            "zones": zones,
            "inter_zone_rtt_ms": _rtt_table([z["centroid"] for z in zones], rttm),
            "intra_zone_ms": rttm["intra_zone_ms"],
            "zone_to_cloud_ms": [lo_c, hi_c],
            "device_mix": mix,
            "provenance": {"rtt": rttm.get("source", ""),
                           "nodes": "device_specs.py (vendor datasheets)"}}


def synthetic_topology(n_zones=4, seed=TOPOLOGY_SEED):
    """Small synthetic topology for smoke runs / tests (no datasets)."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1, size=(n_zones, 2))
    shares = rng.dirichlet(np.ones(n_zones) * 3)
    mix = {"rack_edge_server": 1, "jetson_orin_nano": n_zones,
           "raspberry_pi_4": n_zones + 1, "coral_dev_board": n_zones}
    rttm = {"min_ms": 4.3, "max_ms": 9.4, "intra_zone_ms": 1.0,
            "zone_to_cloud_ms": [20, 40], "source": "synthetic (RIPE band)"}
    topo = assemble_topology("synthetic", centers, shares, rttm, mix, seed)
    topo["provenance"]["cells"] = "synthetic"
    return topo


def rtt(topo, a, b):
    if a == b:
        return 0.0
    k = f"{a}-{b}" if f"{a}-{b}" in topo["inter_zone_rtt_ms"] else f"{b}-{a}"
    return float(topo["inter_zone_rtt_ms"][k])


def zone_distance_matrix(topo):
    c = np.array([z["centroid"] for z in topo["zones"]])
    return np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))


# --- Milan activity per (slot, zone) --------------------------------------
def milan_zone_activity(topo_cells_labels, cells):
    """Returns dict with base/surge day (144 x n_zones) matrices from the
    real trace, the surge zone, and the day labels."""
    import pandas as pd
    act = pd.read_parquet(os.path.join(VALIDATED, "milan_activity.parquet"))
    surge = pd.read_csv(os.path.join(VALIDATED, "milan_surge_candidates.csv"))
    cz = pd.DataFrame({"cell_id": cells.cell_id.values, "zone": topo_cells_labels})
    act = act.merge(cz, on="cell_id")
    surge_day = surge.day.mode().iat[0]
    normal_days = sorted(d for d in act.day.unique() if d != surge_day)
    base_day = normal_days[len(normal_days) // 2]

    def mat(day):
        d = act[act.day == day]
        m = d.pivot_table(index="ts", columns="zone", values="activity",
                          aggfunc="sum").fillna(0.0).sort_index()
        return m.values[:144]

    mb, ms = mat(base_day), mat(surge_day)
    surge_zone = int(np.argmax(ms.sum(axis=0) - mb.sum(axis=0)))
    return {"base": mb, "surge": ms, "surge_zone": surge_zone,
            "base_day": base_day, "surge_day": surge_day,
            "source": "Milan telecom activity (Harvard Dataverse doi:10.7910/DVN/EGZHFV)"}


def synthetic_zone_activity(n_zones, seed=0, n_slots=144):
    """Diurnal synthetic activity (smoke/tests only)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_slots) / n_slots
    diurnal = 0.35 + 0.65 * np.clip(np.sin(np.pi * (t - 0.25) / 0.75), 0, None)
    w = rng.dirichlet(np.ones(n_zones) * 2)
    base = diurnal[:, None] * w[None, :] * 1000
    surge = base.copy()
    surge[:, 1 % n_zones] *= 3.0
    return {"base": base, "surge": surge, "surge_zone": 1 % n_zones,
            "base_day": "synthetic", "surge_day": "synthetic",
            "source": "synthetic diurnal (smoke/tests only)"}


# --- mobility --------------------------------------------------------------------
class MobilityModel:
    """Coarse Tier-1 mobility driven by the Milan trace's spatio-temporal
    structure - an approximation, NOT individual device trajectories.

    Devices get a home zone ~ mean zone activity share. At each 10-minute
    slot boundary a device moves with probability `move_p`; a move goes
    home with probability `home_p` (commuting), otherwise to zone j with
    probability proportional to activity(slot, j) * exp(-d(cur, j)/scale)
    (gravity model: attracted to where activity currently is, damped by
    distance). The result is a per-device piecewise-constant zone
    trajectory; a request's origin zone is a function of its send time."""

    def __init__(self, activity, dist, move_p=0.06, home_p=0.4, slot_s=600.0):
        self.act = np.asarray(activity, float)          # (slots, zones)
        self.dist = np.asarray(dist, float)
        scale = np.median(self.dist[self.dist > 0]) if (self.dist > 0).any() else 1.0
        self.gravity = np.exp(-self.dist / max(scale, 1e-9))
        self.move_p, self.home_p, self.slot_s = move_p, home_p, slot_s

    def trajectories(self, n_devices, rng):
        n_slots, n_z = self.act.shape
        share = self.act.sum(axis=0)
        share = share / share.sum()
        home = rng.choice(n_z, size=n_devices, p=share)
        traj = np.empty((n_devices, n_slots), dtype=np.int16)
        traj[:, 0] = home
        for s in range(1, n_slots):
            cur = traj[:, s - 1].copy()
            moving = rng.uniform(size=n_devices) < self.move_p
            go_home = rng.uniform(size=n_devices) < self.home_p
            a = self.act[s] / max(self.act[s].sum(), 1e-12)
            for i in np.where(moving)[0]:
                if go_home[i]:
                    cur[i] = home[i]
                else:
                    p = a * self.gravity[cur[i]]
                    cur[i] = rng.choice(n_z, p=p / p.sum())
            traj[:, s] = cur
        return home, traj


def compress_trajectory(row):
    """[slot, zone] change points."""
    out = [[0, int(row[0])]]
    for s in range(1, len(row)):
        if row[s] != row[s - 1]:
            out.append([s, int(row[s])])
    return out
