"""The proposed system: Tier-2 zone agents (one per zone) + Tier-3 global
agent, wired together. ablations.py switches individual mechanisms off
through the constructor flags; nothing else differs between them.

    handle(request):  zone agent translates (shared intent cache -> SLM)
                      zone agent tries local placement
                      else escalate to the global agent
                           (memory -> rules -> digest -> LLM -> rule chain)
    replan(service):  migration of a pre-emption victim / failure recovery:
                      local try in the service's origin zone, then escalate.
                      Uses a scratch record so re-planning cost never pollutes
                      the original request's arrival-time latency breakdown
                      (its LLM calls/tokens are still attributed via req_id).
"""
import numpy as np

from global_agent import GlobalAgent
from sim_engine import Decision
from zone_agent import IntentLibrary, ZoneAgent

SURGE_POLICY = {"w_fit": 0.2, "w_balance": 0.7, "w_accel": 0.05, "w_cap": 0.05}


class ScratchRecord:
    """Duck-typed stand-in for RequestRecord during re-planning."""

    def __init__(self, req_id):
        self.req_id = req_id
        self.latency_breakdown = {}
        self.reached_llm_stage = False
        self.llm_verified = None
        self.llm_retries = 0
        self.fallback_used = False
        self.escalated = False
        self.cache_similarity = None

    def add_latency(self, component, ms):
        self.latency_breakdown[component] = self.latency_breakdown.get(component, 0.0) + ms

    @property
    def total_latency_ms(self):
        return sum(self.latency_breakdown.values())


class HierarchicalOrchestrator:
    name = "full"
    uses_llm = True
    disables = ()

    def __init__(self, *, catalog, embedder, topo, make_llm, threshold, rng,
                 seed_entries=(), cache_cap=500, shadow_rate=0.05,
                 use_cache=True, use_memory=True, use_digest=True, use_llm=True,
                 use_preempt_degrade=True, use_zone_tier=True, use_cross_zone=True,
                 use_procedural=True, use_cache_sharing=True, cache_sync_s=5.0, name=None):
        if name:
            self.name = name
        self.catalog, self.topo = catalog, topo
        self.flags = dict(use_cache=use_cache, use_memory=use_memory, use_digest=use_digest,
                          use_llm=use_llm, use_preempt_degrade=use_preempt_degrade,
                          use_zone_tier=use_zone_tier, use_cross_zone=use_cross_zone,
                          use_procedural=use_procedural, use_cache_sharing=use_cache_sharing)
        self.zone_ids = [z["zone_id"] for z in topo["zones"]]
        self.use_zone_tier, self.use_cross_zone = use_zone_tier, use_cross_zone
        mk = dict(threshold=threshold, cache_cap=cache_cap, shadow_rate=shadow_rate,
                  seed_entries=seed_entries, use_cache=use_cache)
        self.shared_lib = None
        if use_zone_tier:
            if use_cache and use_cache_sharing:
                # one operator, one cache: same total capacity as the
                # per-zone caches it replaces
                self.shared_lib = IntentLibrary(embedder, threshold, cache_cap * len(self.zone_ids),
                                                seed_entries, sync_s=cache_sync_s)
            self.zones = {z: ZoneAgent(z, catalog, embedder, make_llm("slm", f"zone:{z}"),
                                       rng=np.random.default_rng(rng.integers(1 << 31)),
                                       library=self.shared_lib, **mk)
                          for z in self.zone_ids}
            self.central = None
        else:
            # no zone tier: one central translator (with its own cache) at
            # the global agent; every request travels there first
            self.zones = {}
            self.central = ZoneAgent("global", catalog, embedder, make_llm("slm", "global_translator"),
                                     rng=np.random.default_rng(rng.integers(1 << 31)), **mk)
        self.global_ = GlobalAgent(make_llm("llm", "global"), catalog, self.zone_ids,
                                   use_memory=use_memory, use_digest=use_digest,
                                   use_llm=use_llm, use_preempt_degrade=use_preempt_degrade,
                                   use_procedural=use_procedural)
        self.view = None
        self.surge_policies = 0

    def attach(self, view, latency):
        self.view = view
        for za in self._agents():
            za.latency = latency

    def _agents(self):
        return list(self.zones.values()) + ([self.central] if self.central else [])

    def _policies(self):
        return {z: za.policy for z, za in self.zones.items()}

    def _allowed(self, origin, profile):
        if profile.get("data_locality") == "zone_local" or not self.use_cross_zone:
            return [origin]
        return list(self.zone_ids)

    # ------------------------------------------------------------------
    def handle(self, t, req, rec):
        view, origin = self.view, req.zone_id
        if self.use_zone_tier:
            za = self.zones[origin]
        else:
            rec.add_latency("escalation", view.zone_to_global_ms(origin))
            za = self.central
        prof = za.translate(t, req.req_id, req.text, rec)
        if prof is None:
            return Decision(node=None, path="unresolved", decision_source="none",
                            action="reject", demand={"cpu": 0.0, "mem": 0.0})
        demand = view.manifest(req.req_id, prof["service_type"])
        if self.use_zone_tier:
            nid = za.try_local(view, demand, prof, rec)
            if nid is not None:
                return Decision(node=nid, path="local", decision_source="local",
                                demand=demand, profile=prof)
        rec.escalated = True
        return self.global_.escalate(t, view, req.req_id, origin, prof, demand,
                                     self._allowed(origin, prof), rec,
                                     policies=self._policies(),
                                     charge_hop=self.use_zone_tier,
                                     tried_local=self.use_zone_tier)

    def replan(self, t, svc, scratch, reason=""):
        view, origin = self.view, svc.origin_zone
        prof, demand = svc.profile, dict(svc.demand)
        if self.use_zone_tier:
            nid = self.zones[origin].try_local(view, demand, prof, scratch)
            if nid is not None:
                return Decision(node=nid, path="local", decision_source="local",
                                demand=demand, profile=prof)
        return self.global_.escalate(t, view, svc.req_id, origin, prof, demand,
                                     self._allowed(origin, prof), scratch,
                                     policies=self._policies(), charge_hop=True,
                                     tried_local=self.use_zone_tier)

    # --- ticks / events --------------------------------------------------------
    def on_digest_tick(self, t, digests):
        self.global_.on_digest_tick(t, digests)

    def on_policy_tick(self, t):
        self.global_.on_policy_tick(t)

    def on_failure(self, t, zone):
        if zone in self.zones:
            self.zones[zone].policy = dict(SURGE_POLICY)
            self.surge_policies += 1

    def samples(self):
        return self.global_.samples

    def stats(self):
        agg = {}
        for za in self._agents():
            for k, v in za.stats().items():
                agg[k] = agg.get(k, 0) + v
        if self.shared_lib is not None:
            agg["cache_entries"] = len(self.shared_lib)
            agg["cache_evictions"] = self.shared_lib.evictions
        agg.update(self.global_.stats())
        agg["surge_policies"] = self.surge_policies
        return agg
