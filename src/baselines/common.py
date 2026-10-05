"""Shared pieces for the baselines: the tool environment ReAct/LATS act
through, path labelling, and a catalog view with typical manifest sizes.

The baselines are centralised agents (hosted with the global tier), so
every tool call is a round trip to the queried zone (zone_to_global_ms),
charged to the request's `escalation` latency component, exactly like the
proposed system's probes.
"""
import json

from global_agent import OBJECTIVE
from scenario import PRIORITY_RANK, SERVICE_TYPES
from sim_engine import EPS, Decision

RUNNING_LIST_MAX = 8
STEP_BUDGET = 7          # ReAct's step cap and LATS's tree depth
GENERIC = {"latency_class": "interactive", "data_locality": "any", "priority": "normal"}


def profile_for(catalog, st):
    if st and catalog.known(st):
        return catalog.profile(st)
    return {"service_type": f"novel:{st}" if st else None, **GENERIC}


def catalog_sizes(catalog, resources):
    out = {}
    for n, spec in catalog.types.items():
        c, m = resources.median_size(spec)
        out[n] = {"cpu": c, "mem": m}
    return out


def label_path(origin, zone, action):
    if action == "preempt":
        return "preempt_local" if zone == origin else "preempt"
    if action == "degrade":
        return "llm_degraded"
    return "local" if zone == origin else "llm"


def finish_decision(rec, dec, origin):
    """Uniform 'escalated' semantics for flat baselines: anything other
    than a plain full-size placement in the origin zone needed more than
    a local step."""
    rec.escalated = not (dec.node is not None and dec.action == "place"
                         and dec.meta.get("zone") == origin)
    return dec


class ToolEnv:
    """Per-request tool space: read_digest(zone), read_running_services(zone),
    try_place(zone, node, service_type, evict=None, degrade_level=1.0)."""

    def __init__(self, view, catalog, req_id, origin, rec, allow_preempt=True,
                 allow_degrade=True, demand_override=None, priority_override=None):
        self.view, self.catalog, self.req_id, self.origin, self.rec = \
            view, catalog, req_id, origin, rec
        self.allow_preempt, self.allow_degrade = allow_preempt, allow_degrade
        self.demand_override, self.priority_override = demand_override, priority_override
        self.decision = None
        self.calls = 0

    def _rtt(self, zone):
        self.calls += 1
        self.rec.add_latency("escalation", self.view.zone_to_global_ms(zone))

    def read_digest(self, zone):
        if zone not in self.view.zone_ids:
            return {"error": f"unknown zone {zone}"}
        self._rtt(zone)
        tab = self.view.zone_table(zone)
        return {"zone": zone, "nodes": [
            {"node": n, "class": r["device_class"], "cpu_free": round(r["cpu_free"], 2),
             "mem_free": round(r["mem_free"], 2), "accel": r["accelerator"]}
            for n, r in sorted(tab.items()) if r["healthy"]]}

    def read_running_services(self, zone, service_type=None):
        """Only the services that could be evicted for this request: strictly
        lower priority than the request's (its service_type's catalog
        priority, or the re-plan's own priority; below "critical" when
        unknown), lowest priority and soonest-finishing first, at most
        RUNNING_LIST_MAX. The quick pilot listed up to 25 services of any
        priority, and ReAct spent its step budget reading them."""
        if zone not in self.view.zone_ids:
            return {"error": f"unknown zone {zone}"}
        self._rtt(zone)
        pr = self.priority_override or (profile_for(self.catalog, service_type)["priority"]
                                        if service_type and self.catalog.known(service_type)
                                        else "critical")
        rank = PRIORITY_RANK.get(pr, 3)
        run = [s for s in self.view.running(zone) if PRIORITY_RANK.get(s["priority"], 1) < rank]
        run.sort(key=lambda s: (PRIORITY_RANK.get(s["priority"], 1), s["remaining_s"]))
        return {"zone": zone, "evictable_for_priority": pr, "n_evictable": len(run),
                "services": run[:RUNNING_LIST_MAX]}

    def try_place(self, zone, node, service_type, evict=None, degrade_level=1.0):
        if zone not in self.view.zone_ids:
            return {"ok": False, "reason": f"unknown zone {zone}"}
        self._rtt(zone)
        tab = self.view.zone_table(zone)
        if node not in tab:
            return {"ok": False, "reason": f"node {node} is not in zone {zone}"}
        prof = profile_for(self.catalog, service_type)
        if self.priority_override:
            prof = {**prof, "priority": self.priority_override}
        demand = dict(self.demand_override or self.view.manifest(self.req_id, prof["service_type"]))
        try:
            lvl = float(degrade_level or 1.0)
        except (TypeError, ValueError):
            lvl = 1.0
        if lvl < 1.0 - 1e-9:
            floor = self.catalog.degrade_floor(prof["service_type"]) \
                if self.catalog.known(prof["service_type"]) else 0.6
            if not self.allow_degrade or lvl + 1e-9 < floor:
                return {"ok": False, "reason": f"degrade_level {lvl} not allowed (floor {floor})"}
        want = {**demand, "cpu": demand["cpu"] * lvl, "mem": demand["mem"] * lvl}
        victims = []
        if evict:
            if not self.allow_preempt:
                return {"ok": False, "reason": "pre-emption not allowed"}
            v = next((s for s in self.view.running(zone) if s["req_id"] == evict), None)
            if v is None or v["node"] != node:
                return {"ok": False, "reason": f"{evict} is not running on {node}"}
            if PRIORITY_RANK.get(v["priority"], 1) >= PRIORITY_RANK.get(prof["priority"], 1):
                return {"ok": False, "reason": "victim priority is not lower than the request's"}
            victims = [evict]
        if not self.view.node_fits_after_evict(node, victims, want):
            r = tab[node]
            return {"ok": False, "reason": f"{node} has cpu_free={r['cpu_free']:.2f} "
                                           f"mem_free={r['mem_free']:.2f}, needs "
                                           f"cpu={want['cpu']:.2f} mem={want['mem']:.2f}"}
        action = "preempt" if victims else ("degrade" if lvl < 1.0 - 1e-9 else "place")
        self.decision = Decision(node=node, path=label_path(self.origin, zone, action),
                                 decision_source="llm_fresh", action=action, demand=want,
                                 degrade_level=lvl, victims=victims, profile=prof,
                                 meta={"zone": zone})
        return {"ok": True}

    def run_tool(self, tool, args):
        args = args if isinstance(args, dict) else {}
        try:
            if tool == "read_digest":
                return self.read_digest(args.get("zone"))
            if tool == "read_running_services":
                return self.read_running_services(args.get("zone"), args.get("service_type"))
            if tool == "try_place":
                return self.try_place(args.get("zone"), args.get("node"),
                                      args.get("service_type"), args.get("evict"),
                                      args.get("degrade_level", 1.0))
        except Exception as e:                      # malformed args from the model
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown tool {tool!r}"}


# Shared, word for word, by ReAct and LATS. Rewritten after the quick pilot,
# where ReAct never called try_place in 519 of its 687 rejected requests:
# the old text did not say that try_place checks the fit itself, nor when
# reading running services is worth a step.
TOOLS_DOC = """Tools (call exactly one per step; at most """ + str(STEP_BUDGET) + """ steps per request, and a
request that is not placed by then is rejected):
- try_place {"zone": z, "node": n, "service_type": t, "evict": req_id|null, "degrade_level": 1.0|0.8|0.6|0.4}
  The ONLY way to serve the request. Deploys it as service type t on node n and commits on success.
  It checks the fit itself and, on failure, says why (free CPU/memory vs. need), so trying a
  promising node directly is cheap. Optionally evict ONE strictly lower-priority service running on
  that node, or run at a reduced resource level (not below the type's floor).
- read_digest {"zone": z}: the healthy nodes of zone z with their free CPU cores / memory GB.
  Use it to find a node with room before calling try_place.
- read_running_services {"zone": z, "service_type": t}: the services in z that a request of type t
  may evict (strictly lower priority), lowest priority and soonest finishing first.
  Only worth a step when no node has room and the request is high or critical priority.
- finish {}: give up (the request is rejected).
service_type must be a catalog name (or a short new snake_case name if nothing fits).
data_locality "zone_local" services must stay in the origin zone.
""" + OBJECTIVE


def request_block(catalog, sizes, text, origin, view):
    return {"intent": text, "origin_zone": origin,
            "zones": [{"zone": z, "rtt_ms": round(view.rtt(origin, z), 2)}
                      for z in view.zone_ids],
            "catalog": catalog.prompt_block()}


def dumps(o):
    return json.dumps(o, sort_keys=True, default=str)


__all__ = ["ToolEnv", "TOOLS_DOC", "profile_for", "catalog_sizes", "label_path",
           "finish_decision", "request_block", "dumps", "EPS", "SERVICE_TYPES"]
