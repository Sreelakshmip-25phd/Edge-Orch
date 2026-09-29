"""Greedy least-loaded placement (kept from the old repo), zone-blind and
flat: the healthy node with the most free CPU anywhere that fits.

It is ORACLE-TYPED: it is handed the true service type (translation assumed
perfect and free - translation_source="oracle"), so it isolates the
*placement policy/architecture* and is also the reference the workload
load calibration targets (calibrate_workload.py). It ignores data locality
(violations are measured, not prevented) and never pre-empts or degrades.
"""
from latency_model import measure
from sim_engine import Decision


class GreedyOracle:
    name = "greedy_oracle"
    uses_llm = False
    oracle = True

    def __init__(self, **_):
        self.view = None

    def attach(self, view, latency):
        self.view = view

    def _best(self, demand):
        best, bf = None, -1.0
        for z in self.view.zone_ids:
            for nid, r in self.view.zone_table(z).items():
                if r["healthy"] and r["cpu_free"] >= demand["cpu"] - 1e-6 and \
                        r["mem_free"] >= demand["mem"] - 1e-6 and r["cpu_free"] > bf:
                    best, bf = nid, r["cpu_free"]
        return best

    def _decide(self, origin, prof, demand, rec):
        nid, ms = measure(self._best, demand)
        rec.add_latency("decision", ms)
        if nid is None:
            rec.escalated = True
            return Decision(node=None, path="unresolved", decision_source="rule",
                            action="reject", demand=demand, profile=prof)
        z = self.view.node_zone(nid)
        rec.escalated = z != origin
        return Decision(node=nid, path="local" if z == origin else "digest",
                        decision_source="local" if z == origin else "rule",
                        demand=demand, profile=prof, meta={"zone": z})

    def handle(self, t, req, rec):
        prof = self.view.oracle_profile(req.req_id)
        rec.translation_source = "oracle"
        rec.add_latency("translation", 0.0)
        demand = self.view.manifest(req.req_id, prof["service_type"])
        return self._decide(req.zone_id, prof, demand, rec)

    def replan(self, t, svc, scratch, reason=""):
        return self._decide(svc.origin_zone, svc.profile, dict(svc.demand), scratch)

    def stats(self):
        return {}
