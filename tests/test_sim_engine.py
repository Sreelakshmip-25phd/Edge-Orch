import numpy as np

from helpers import req, tiny_topology, workload
from orchestrator import ScratchRecord
from scenario import SYNTHETIC_RESOURCE_PROFILES, ResourceModel, ServiceCatalog
from sim_engine import CapacityViolation, Decision, EdgeSimulation, Manifests, Node
from telemetry import Telemetry


class _Lat:
    def transport_in(self, ms): return ms
    def deployment(self, c): return 5.0
    def cross_zone(self, a, b): return 6.0


class Adversary:
    """Always tries to squeeze the newest request onto n0 by pre-empting
    everything else there, degrading when that isn't enough; victims are
    migrated onto n0 again when possible (forcing nested pre-emption)."""
    name = "adversary"

    def attach(self, view, lat):
        self.v = view

    def _on_n0(self, rid, demand, prof, allow_evict=True):
        v = self.v
        running = [s for s in v.running("z0") if s["node"] == "n0" and s["req_id"] != rid]
        for level in (1.0, 0.8, 0.6, 0.4):
            d = {**demand, "cpu": demand["cpu"] * level, "mem": demand["mem"] * level}
            for k in range(len(running) + 1):
                victims = [s["req_id"] for s in running[:k]] if allow_evict else []
                if v.node_fits_after_evict("n0", victims, d):
                    return Decision(node="n0", path="preempt_local" if victims else "local",
                                    decision_source="rule",
                                    action="preempt" if victims else ("degrade" if level < 1 else "place"),
                                    demand=d, degrade_level=level, victims=victims, profile=prof)
        return Decision(node=None, path="unresolved", decision_source="rule", action="reject",
                        demand=demand, profile=prof)

    def handle(self, t, r, rec):
        rec.translation_source = "oracle"
        prof = self.v.oracle_profile(r.req_id)
        # every third request may not evict -> must degrade instead
        return self._on_n0(r.req_id, self.v.manifest(r.req_id, prof["service_type"]), prof,
                           allow_evict=int(r.req_id[1:]) % 3 != 0)

    def replan(self, t, svc, scratch, reason=""):
        nid = self.v.solve("z1", svc.demand, svc.profile)
        if nid:
            return Decision(node=nid, path="digest", decision_source="rule",
                            demand=dict(svc.demand), profile=svc.profile)
        return self._on_n0(svc.req_id, dict(svc.demand), svc.profile)


def test_capacity_invariants_hold_under_preemption_migration_degradation():
    rng = np.random.default_rng(0)
    types = ["iot_aggregator", "traffic_monitor", "video_analytics", "drone_control"]
    reqs = [req(i, float(i), "z0", types[i % 4], float(rng.uniform(0.5, 3.5)),
                float(rng.uniform(0.2, 3.0)), float(rng.uniform(5, 60))) for i in range(120)]
    events = [{"type": "node_failure", "t_s": 40.0, "node_id": "n1", "recover_t_s": 70.0},
              {"type": "node_failure", "t_s": 55.0, "node_id": "n0", "recover_t_s": 60.0}]
    topo, wl = tiny_topology(), workload(reqs, events)
    cat = ServiceCatalog()
    tel = Telemetry()
    from edge_device import DeviceFleet
    sim = EdgeSimulation(topo, wl, Adversary(), tel, _Lat(), cat, DeviceFleet.from_workload(wl, topo),
                         Manifests(cat, ResourceModel(SYNTHETIC_RESOURCE_PROFILES)))
    checked = []
    orig_commit = sim._commit

    def commit_and_check(*a, **kw):
        orig_commit(*a, **kw)
        sim.check_invariants()
        for n in sim.nodes.values():
            assert n.cpu_used >= -1e-9 and n.mem_used >= -1e-9
            assert n.cpu_used <= n.cpu_cap + 1e-9 and n.mem_used <= n.mem_cap + 1e-9
        checked.append(1)
    sim._commit = commit_and_check
    sim.run()
    sim.check_invariants()
    tel.assert_complete()
    kinds = [e.kind for e in tel.events]
    assert kinds.count("preempt") > 5 and "migrate" in kinds and "node_failure" in kinds
    assert any(r.degraded for r in tel.requests.values())
    # nested: a migrated victim itself pre-empted someone
    assert any(e.kind == "preempt" and e.data["by"] in
               {m.data["req"] for m in tel.events if m.kind == "migrate"} for e in tel.events)
    assert len(checked) > 50
    for n in sim.nodes.values():            # everything released at the end
        assert abs(n.cpu_used) < 1e-6 and not n.services


def test_node_rejects_overallocation():
    n = Node({"node_id": "n", "device_class": "raspberry_pi_4", "cpu": 4, "mem_gb": 8}, "z0")
    n.alloc("a", 3.0, 1.0)
    try:
        n.alloc("b", 1.5, 1.0)
        assert False, "should have raised"
    except CapacityViolation:
        pass
    n.release("a")
    assert n.cpu_used == 0.0


def test_migration_does_not_inflate_arrival_latency():
    """Deployment is charged once, at arrival; re-placements record their
    container start on the migrate event instead."""
    import numpy as np
    rng = np.random.default_rng(1)
    types = ["iot_aggregator", "traffic_monitor", "video_analytics", "drone_control"]
    reqs = [req(i, float(i), "z0", types[i % 4], float(rng.uniform(0.5, 3.5)),
                float(rng.uniform(0.2, 3.0)), float(rng.uniform(5, 60))) for i in range(80)]
    topo, wl = tiny_topology(), workload(reqs)
    cat = ServiceCatalog()
    tel = Telemetry()
    from edge_device import DeviceFleet
    sim = EdgeSimulation(topo, wl, Adversary(), tel, _Lat(), cat, DeviceFleet.from_workload(wl, topo),
                         Manifests(cat, ResourceModel(SYNTHETIC_RESOURCE_PROFILES)))
    sim.run()
    moved = {e.data["req"] for e in tel.events if e.kind == "migrate"}
    assert moved
    for rid in moved:
        assert tel.requests[rid].latency_breakdown["deployment"] == 5.0   # exactly one draw
    assert all("redeploy_ms" in e.data for e in tel.events if e.kind == "migrate")
