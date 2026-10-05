import json

from global_agent import CaseMemory, GlobalAgent
from helpers import ScriptedLLM, catalog, req, tiny_topology, workload
from orchestrator import ScratchRecord
from scenario import ground_truth_profile
from sim_engine import EdgeSimulation, Manifests, ServiceInfo
from telemetry import Telemetry


class _Lat:
    def transport_in(self, ms): return ms
    def deployment(self, c): return 0.0
    def cross_zone(self, a, b): return 6.0
    def embed_ms(self): return 0.0


def _sim(reqs, orch=None):
    from edge_device import DeviceFleet
    from scenario import SYNTHETIC_RESOURCE_PROFILES, ResourceModel
    topo, wl = tiny_topology(), workload(reqs)
    cat = catalog()

    class _Null:
        def handle(self, t, r, rec): raise AssertionError
    sim = EdgeSimulation(topo, wl, orch or _Null(), Telemetry(), _Lat(), cat,
                         DeviceFleet.from_workload(wl, topo),
                         Manifests(cat, ResourceModel(SYNTHETIC_RESOURCE_PROFILES)))
    return sim, cat


def _occupy(sim, rid, node, cpu, mem, prio="low", st="iot_aggregator", life=500.0):
    sim.nodes[node].alloc(rid, cpu, mem)
    sim.services[rid] = ServiceInfo(req_id=rid, origin_zone=sim.node_zone(node),
                                    profile={**ground_truth_profile(st), "priority": prio},
                                    demand={"cpu": cpu, "mem": mem}, node=node,
                                    t_end=life)


def _digests(sim, t=0.0):
    return {z: zr.digest(t, 0.0) for z, zr in sim.zones.items()}


PROF = ground_truth_profile("traffic_monitor")          # locality "any"
DEM = {"cpu": 3.0, "mem": 1.0, "accel_pref": False}


def test_stale_memory_case_rejected_by_live_digest_check():
    sim, cat = _sim([req(1, 0, "z0", "traffic_monitor", 3, 1, 10)])
    ga = GlobalAgent(ScriptedLLM(default={"action": "reject"}), cat, sim.zone_ids,
                     use_llm=False, use_procedural=False)
    key = CaseMemory.key(PROF, DEM, "z0")
    ga.mem.record(key, "z1", "n1", "place", 1.0, None, 10.0, t=0.0)   # z1 had room back then
    _occupy(sim, "x0", "n0", 3.5, 1.0)                   # origin full (local just failed)
    _occupy(sim, "x1", "n1", 5.0, 1.0)                   # z1's room since consumed
    _occupy(sim, "x2", "n2", 3.5, 1.0)
    ga.on_digest_tick(1.0, _digests(sim, 1.0))
    rec = ScratchRecord("r1")
    dec = ga.escalate(1.0, sim, "r1", "z0", PROF, DEM, ["z0", "z1"], rec, tried_local=True)
    assert ga.mem.n_stale == 1                           # the case was checked and refused
    assert dec.path != "episodic" and dec.node is None   # nothing fits anywhere
    assert sim.probe_count == 0                          # refused on the digest, no wasted probe
    # capacity comes back -> the same case is reused
    sim.nodes["n1"].release("x1")
    ga.on_digest_tick(2.0, _digests(sim, 2.0))
    dec2 = ga.escalate(2.0, sim, "r2", "z0", PROF, DEM, ["z0", "z1"], ScratchRecord("r2"),
                       tried_local=True)
    assert dec2.path == "episodic" and dec2.node == "n1" and dec2.decision_source == "memory"


def test_llm_proposal_that_does_not_fit_is_rejected_and_retried_once():
    sim, cat = _sim([req(1, 0, "z0", "video_analytics", 3, 1, 10)])
    _occupy(sim, "x0", "n0", 3.5, 1.0, prio="low")      # z0 full, low-priority victim present
    _occupy(sim, "x1", "n1", 5.5, 1.0, prio="high")
    _occupy(sim, "x2", "n2", 3.5, 1.0, prio="high")
    llm = ScriptedLLM(script={"decide": [
        {"option": "o9", "reason": "looks roomy"},                        # not a choice -> invalid
        {"option": "o1", "reason": "low priority"}]})                     # o1 = evict x0
    ga = GlobalAgent(llm, cat, sim.zone_ids, use_memory=False)
    ga.on_digest_tick(0.0, _digests(sim))
    prof = ground_truth_profile("video_analytics")       # high priority, zone_local
    rec = ScratchRecord("r1")
    dec = ga.escalate(0.0, sim, "r1", "z0", prof, DEM, ["z0"], rec)
    assert [c["kind"] for c in llm.calls] == ["decide", "decide"]
    choices = json.loads(llm.calls[0]["user"])["choices"]
    assert choices[0] == {"id": "o1", "action": "preempt", "zone": "z0", "victim": "x0",
                          "victim_priority": "low", "victim_service_type": choices[0]["victim_service_type"],
                          "victim_remaining_s": choices[0]["victim_remaining_s"],
                          "rtt_ms": choices[0]["rtt_ms"]}
    assert "place" not in {c["action"] for c in choices} and choices[-1]["action"] == "reject"
    assert "feedback" in json.loads(llm.calls[1]["user"])   # retry carries the verifier's error
    assert rec.llm_verified and rec.llm_retries == 1
    assert dec.action == "preempt" and dec.victims == ["x0"] and dec.path == "preempt_local"


def test_llm_invalid_twice_falls_back_to_rule_chain():
    sim, cat = _sim([req(1, 0, "z0", "video_analytics", 3, 1, 10)])
    _occupy(sim, "x0", "n0", 3.5, 1.0, prio="low")
    llm = ScriptedLLM(default={"option": "o9"})
    ga = GlobalAgent(llm, cat, sim.zone_ids, use_memory=False)
    ga.on_digest_tick(0.0, _digests(sim))
    rec = ScratchRecord("r1")
    dec = ga.escalate(0.0, sim, "r1", "z0", ground_truth_profile("video_analytics"), DEM,
                      ["z0"], rec)
    assert len(llm.calls) == 2 and rec.fallback_used and rec.llm_verified is False
    assert dec.decision_source == "rule" and dec.victims == ["x0"]


def test_degradation_menu_respects_floor():
    sim, cat = _sim([req(1, 0, "z0", "drone_control", 3, 1, 10)])
    _occupy(sim, "x0", "n0", 1.5, 1.0, prio="critical")     # 2.5 free: 0.8*3=2.4 fits
    ga = GlobalAgent(ScriptedLLM(), cat, sim.zone_ids, use_memory=False)
    ga.on_digest_tick(0.0, _digests(sim))
    opts = ga._options(0.0, sim, ScratchRecord("r"), "z0", ground_truth_profile("drone_control"),
                       DEM, ["z0"], None)
    assert opts["degrade"] and all(o["level"] >= 0.8 for o in opts["degrade"])   # drone floor 0.8
