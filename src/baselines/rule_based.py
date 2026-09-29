"""Static rule-based hierarchical orchestrator (kept from the old repo,
adapted to the cold-start / open-vocabulary setting).

  * translation: a *static* intent library built once from the training
    phrasing pool (hand-written seed templates x places) for the types
    registered at t=0 - never learns, no SLM. Below-threshold intents are
    rejected as unknown, so novel phrasings and new service types fail.
  * placement: local solver in the origin zone; if it doesn't fit and the
    service may leave the zone, the nearest zones whose digest fits are
    probed. No LLM, no memory, no pre-emption, no degradation.
"""
from latency_model import measure
from scenario import PLACES_INITIAL, PLACES_LATER, ground_truth_profile
from sim_engine import Decision, digest_fits
from zone_agent import IntentLibrary


class RuleBasedHierarchical:
    name = "rule_based"
    uses_llm = False

    def __init__(self, *, catalog, embedder, topo, threshold, train_pool, **_):
        self.catalog = catalog
        seed = [(tpl.format(place=p), ground_truth_profile(st))
                for st in catalog.names() for tpl in train_pool.get(st, [])
                for p in PLACES_INITIAL + PLACES_LATER]
        embedder.warm([t for t, _ in seed])
        self.lib = IntentLibrary(embedder, threshold, cap=10 ** 6, seed_entries=seed)
        self.digests, self.view, self.latency = {}, None, None
        self.counters = {"static_hits": 0, "novel_rejected": 0}

    def attach(self, view, latency):
        self.view, self.latency = view, latency

    def on_digest_tick(self, t, digests):
        self.digests = digests

    def _place(self, t, origin, prof, demand, rec):
        nid, ms = measure(self.view.solve, origin, demand, prof)
        rec.add_latency("decision", ms)
        if nid is not None:
            return Decision(node=nid, path="local", decision_source="local",
                            demand=demand, profile=prof)
        rec.escalated = True
        if prof["data_locality"] == "zone_local":
            return Decision(node=None, path="unresolved", decision_source="rule",
                            action="reject", demand=demand, profile=prof)
        rec.add_latency("escalation", self.view.zone_to_global_ms(origin))
        cands = sorted((self.view.rtt(origin, z), z) for z, d in self.digests.items()
                       if z != origin and t - d["t"] <= 15.0 and digest_fits(d, demand))
        for _, z in cands[:3]:
            rec.add_latency("escalation", self.view.zone_to_global_ms(z))
            nid, ms = measure(self.view.solve, z, demand, prof)
            rec.add_latency("decision", ms)
            if nid is not None:
                return Decision(node=nid, path="digest", decision_source="digest",
                                demand=demand, profile=prof)
        return Decision(node=None, path="unresolved", decision_source="rule",
                        action="reject", demand=demand, profile=prof)

    def handle(self, t, req, rec):
        (prof, sim, _), ms = measure(self.lib.lookup, req.text)
        rec.add_latency("translation", ms + (self.latency.embed_ms() if self.latency else 0.0))
        rec.translation_source = "static_rule"
        rec.cache_similarity = round(sim, 4)
        if prof is None:
            self.counters["novel_rejected"] += 1
            rec.escalated = True
            return Decision(node=None, path="unresolved", decision_source="rule",
                            action="reject", demand={"cpu": 0.0, "mem": 0.0})
        self.counters["static_hits"] += 1
        rec.type_resolution = "static"
        demand = self.view.manifest(req.req_id, prof["service_type"])
        return self._place(t, req.zone_id, prof, demand, rec)

    def replan(self, t, svc, scratch, reason=""):
        return self._place(t, svc.origin_zone, svc.profile, dict(svc.demand), scratch)

    def stats(self):
        return dict(self.counters)
