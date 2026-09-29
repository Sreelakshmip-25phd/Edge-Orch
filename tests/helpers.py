"""Small hand-built fixtures shared by the tests (no datasets, no model
server, no MiniLM download needed)."""
import json

import numpy as np

import config  # noqa: F401  (thread caps, paths)
from llm_client import LLMResult, LLMUnavailable
from scenario import SERVICE_TYPES, ground_truth_profile, ServiceCatalog


def node(nid, cls, cpu, mem):
    return {"node_id": nid, "device_class": cls, "cpu": float(cpu), "mem_gb": float(mem),
            "power_w": 10.0, "accelerator": None}


def tiny_topology():
    """z0: one Pi (4 cores / 8 GB); z1: one Jetson (6/8) + one Pi."""
    return {"name": "tiny", "n_zones": 2, "n_nodes": 3,
            "zones": [{"zone_id": "z0", "centroid": [0, 0], "activity_share": 0.5,
                       "zone_to_global_ms": 25.0,
                       "nodes": [node("n0", "raspberry_pi_4", 4, 8)]},
                      {"zone_id": "z1", "centroid": [1, 0], "activity_share": 0.5,
                       "zone_to_global_ms": 30.0,
                       "nodes": [node("n1", "jetson_orin_nano", 6, 8),
                                 node("n2", "raspberry_pi_4", 4, 8)]}],
            "inter_zone_rtt_ms": {"z0-z1": 6.0}, "intra_zone_ms": 1.0,
            "zone_to_cloud_ms": [20, 40]}


def req(i, t, zone, stype, cpu, mem, life, text=None, device=None):
    s = SERVICE_TYPES[stype]
    return {"req_id": f"r{i}", "t_s": float(t), "device_id": device or f"d{zone}",
            "origin_zone": zone, "phase": "A",
            "intent_text": text or f"{stype} request number {i}",
            "truth": {**ground_truth_profile(stype), "latency_ms": s["latency_ms"],
                      "cpu": float(cpu), "mem": float(mem), "lifetime_base_s": float(life),
                      "degrade_floor": s["degrade_floor"], "accel_pref": False}}


def workload(reqs, events=(), horizon=1000.0):
    zones = sorted({r["origin_zone"] for r in reqs} | {"z0", "z1"})
    devices = [{"device_id": f"d{z}", "home_zone": z, "trajectory": [[0, int(z[1:])]]}
               for z in zones]
    return {"meta": {"seed": 0, "horizon_s": horizon, "slot_s": horizon, "lifetime_scale": 1.0,
                     "n_requests": len(reqs)},
            "devices": devices, "requests": sorted(reqs, key=lambda r: r["t_s"]),
            "events": list(events)}


class HashEmbedder:
    """Deterministic bag-of-words embedder: identical texts -> sim 1,
    disjoint vocab -> sim ~0. Lets tests control cache hits exactly."""
    backend = "hash_bow"

    def __init__(self, dim=256):
        self.dim = dim

    def encode_uncached(self, texts):
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in str(t).lower().replace(",", " ").split():
                out[i, hash(w) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)

    encode = encode_uncached

    def warm(self, texts):
        return 0


class ScriptedLLM:
    """Duck-typed MultiLLM: returns queued dicts per kind, records calls."""

    def __init__(self, role="llm", script=None, default=None):
        self.role = role
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default
        self.calls = []

    def ask(self, system, user, *, kind, req_id=None, temperature=0.0, sample_idx=0, **_):
        self.calls.append({"kind": kind, "user": user, "req_id": req_id})
        q = self.script.get(kind)
        if q:
            out = q.pop(0)
        elif self.default is not None:
            out = self.default(kind, system, user) if callable(self.default) else self.default
        else:
            raise LLMUnavailable(f"no scripted answer for {kind}")
        return LLMResult(data=out, source="fresh", tokens_in=10, tokens_out=5,
                         wall_ms=100.0, sim_ms=100.0)


def user_json(call):
    return json.loads(call["user"])


def catalog():
    return ServiceCatalog()


def mock_env(n_requests=150, seed=0, n_zones=3, horizon=3 * 3600.0, lifetime_scale=40.0):
    """Synthetic scenario + a factory that runs any system on it with the
    mock LLM and the hash embedder. Returns run(name) -> (telemetry, orch)."""
    import workload as W
    from data_foundation import SYNTHETIC_FAILURE_MODEL, FailureModel
    from edge_device import DeviceFleet
    from latency_model import LatencyModel
    from llm_client import MultiLLM
    from scenario import (SYNTHETIC_RESOURCE_PROFILES, ResourceModel, load_intent_pools,
                          synthetic_topology, synthetic_zone_activity)
    from sim_engine import EdgeSimulation, Manifests
    from telemetry import Telemetry
    topo = synthetic_topology(n_zones)
    pools = load_intent_pools()
    res = ResourceModel(SYNTHETIC_RESOURCE_PROFILES)
    wl = W.generate(topo, synthetic_zone_activity(n_zones), res,
                    FailureModel(SYNTHETIC_FAILURE_MODEL), pools, seed=seed,
                    n_requests=n_requests, n_devices=40, horizon_s=horizon, min_failures=1)
    wl["meta"]["lifetime_scale"] = lifetime_scale
    emb = HashEmbedder()
    mock = [{"name": "MOCK", "mock": True, "roles": None, "models": ["mock"]}]

    def run(name, build=None):
        import evaluator as E
        tel = Telemetry({"horizon_s": horizon})
        lat = LatencyModel(topo, "placeholder")
        cat = ServiceCatalog()

        def make_llm(role, agent):
            return MultiLLM(None, role, telemetry=tel, agent=agent, model_label="mock",
                            latency=lat, providers=mock)

        class Ctx:
            pass
        ctx = Ctx()
        ctx.embedder, ctx.topo, ctx.resources, ctx.pools, ctx.threshold = emb, topo, res, pools, 0.5
        orch = (build or E.build_system)(name, ctx, cat, make_llm, seed)
        sim = EdgeSimulation(topo, wl, orch, tel, lat, cat, DeviceFleet.from_workload(wl, topo),
                             Manifests(cat, res))
        sim.run()
        sim.check_invariants()
        tel.assert_complete()
        return tel, orch
    return run, wl, topo
