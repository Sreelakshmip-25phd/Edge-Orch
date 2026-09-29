"""CORE-style role-affinity baseline (device / edge / cloud stand-in).

A lightweight scheduler routes each decision to the cheapest capable role
by task complexity, mirroring CORE's device/edge/cloud split:

  device role (no model)  deterministic local placement in the origin zone
  edge role   (SLM)       intent translation - every request, no cache;
                          "moderate" decisions: a cross-zone target exists
                          according to the digests -> the SLM picks the zone
                          (kind="zone_choice")
  cloud role  (LLM)       "complex" decisions: nothing fits as requested ->
                          the LLM picks pre-emption / degradation / reject
                          with the same structured one-call schema the
                          proposed system uses (kind="decide")

What it deliberately lacks versus the proposed system: the intent cache and
the similar-case memory / procedural rules. Its LLM/SLM calls therefore
track the complexity mix of the workload, not its history.
"""
import json

from global_agent import GlobalAgent
from latency_model import measure
from llm_client import LLMUnavailable
from sim_engine import Decision, digest_fits
from zone_agent import resolve_profile, slm_system_prompt

from .common import finish_decision

ZONE_SYS = """You choose the target zone for a service that does not fit in its origin zone.
Respond with ONLY a JSON object: {"zone": "<zone id>"}. Pick a zone whose "fits" is true;
prefer low rtt_ms for latency-sensitive services."""


class COREBaseline:
    name = "core"
    uses_llm = True

    def __init__(self, *, catalog, embedder, make_llm, topo, **_):
        self.catalog, self.emb = catalog, embedder
        self.slm = make_llm("slm", "core_edge")
        self.cloud = GlobalAgent(make_llm("llm", "core_cloud"), catalog,
                                 [z["zone_id"] for z in topo["zones"]],
                                 use_memory=False, use_procedural=False, use_digest=True)
        self.view, self.digests = None, {}
        self.counters = {"device": 0, "edge": 0, "cloud": 0}

    def attach(self, view, latency):
        self.view = view

    def on_digest_tick(self, t, digests):
        self.digests = digests
        self.cloud.on_digest_tick(t, digests)

    def _route(self, t, rid, origin, prof, demand, rec):
        nid, ms = measure(self.view.solve, origin, demand, prof)
        rec.add_latency("decision", ms)
        if nid is not None:
            self.counters["device"] += 1
            return Decision(node=nid, path="local", decision_source="local",
                            demand=demand, profile=prof, meta={"zone": origin})
        allowed = [origin] if prof.get("data_locality") == "zone_local" else list(self.view.zone_ids)
        fits = {z: d for z, d in self.digests.items()
                if z in allowed and z != origin and t - d["t"] <= 15.0 and digest_fits(d, demand)}
        if fits:                                             # moderate -> edge SLM
            self.counters["edge"] += 1
            zones = {z: {"rtt_ms": round(self.view.rtt(origin, z), 2), "fits": z in fits,
                         "top_cpu_free": self.digests[z]["top_cpu_free"]}
                     for z in allowed if z in self.digests}
            order = sorted(fits, key=lambda z: self.view.rtt(origin, z))
            try:
                res = self.slm.ask(ZONE_SYS, json.dumps({"request": {"profile": prof,
                                                                      "cpu": demand["cpu"],
                                                                      "mem": demand["mem"]},
                                                          "zones": zones}, sort_keys=True),
                                   kind="zone_choice", req_id=rid)
                rec.add_latency("decision", res.sim_ms)
                rec.reached_llm_stage = True
                z = res.data.get("zone")
                src = "slm_fresh" if res.source == "fresh" else "slm_cached_disk"
                if z in fits:
                    order = [z] + [o for o in order if o != z]
            except LLMUnavailable:
                src = "rule"
            for z in order:
                rec.add_latency("escalation", self.view.rtt(origin, z))
                nid = self.view.solve(z, demand, prof)
                if nid is not None:
                    return Decision(node=nid, path="llm" if src != "rule" else "digest",
                                    decision_source=src, demand=demand, profile=prof,
                                    meta={"zone": z})
        self.counters["cloud"] += 1                            # complex -> cloud LLM
        rec.add_latency("escalation", self.view.zone_to_global_ms(origin))
        key = (prof.get("service_type"), None, None, 0, origin)
        dec = self.cloud._from_llm(t, self.view, rid, origin, prof, demand, allowed, rec, key,
                                   {}, allowed == [origin])
        if dec is None:
            dec = self.cloud._rule_chain(t, self.view, rid, origin, prof, demand, allowed, rec,
                                         key, {}, allowed == [origin])
        return dec

    def handle(self, t, req, rec):
        rid = req.req_id
        try:
            res = self.slm.ask(slm_system_prompt(self.catalog), "INPUT: " + req.text,
                               kind="translate", req_id=rid)
        except LLMUnavailable:
            rec.translation_source = "none"
            return finish_decision(rec, Decision(node=None, path="unresolved",
                                                 decision_source="none", action="reject",
                                                 demand={"cpu": 0.0, "mem": 0.0}), req.zone_id)
        rec.add_latency("translation", res.sim_ms)
        rec.translation_source = "slm_fresh" if res.source == "fresh" else "slm_cached_disk"
        prof, rkind = resolve_profile(res.data, self.catalog, self.emb)
        rec.type_resolution = rkind
        demand = self.view.manifest(rid, prof["service_type"])
        dec = self._route(t, rid, req.zone_id, prof, demand, rec)
        dec.meta.setdefault("zone", self.view.node_zone(dec.node) if dec.node else None)
        return finish_decision(rec, dec, req.zone_id)

    def replan(self, t, svc, scratch, reason=""):
        return self._route(t, svc.req_id, svc.origin_zone, svc.profile, dict(svc.demand), scratch)

    def stats(self):
        return {**self.counters, **{f"cloud_{k}": v for k, v in self.cloud.counters.items()}}
