"""Discrete-event core: heterogeneous node capacities, zone capacity
tables + digests, the deterministic placement solver, and the simulation
driver that executes orchestrator decisions (placement, pre-emption with
migration, degradation, failure re-planning) and writes every state
change into telemetry.

No agent logic lives here. Orchestrators receive only an IntentRequest
(pure NL text + receiving zone) and interact with the cluster through
the ClusterView methods on EdgeSimulation:
  zone_table(z)   live node table of one zone (what that zone's agent sees)
  digests         last broadcast digests (what the global tier sees; may be stale)
  solve(z, ...)   ask zone z's deterministic solver for a node (a "probe")
  running(z)      running services in z (pre-emption candidates)
  manifest(...)   service registry lookup: resource request for a resolved type
The sim never lets an orchestrator read hidden ground truth.
"""
import heapq
from dataclasses import dataclass, field
from typing import List, Optional

from scenario import PRIORITY_RANK, latency_class_of

EPS = 1e-6
DEFAULT_POLICY = {"w_fit": 0.40, "w_balance": 0.30, "w_accel": 0.15,
                  "w_cap": 0.15, "forbid_classes": []}
MAX_MIGRATION_DEPTH = 3


def solve_placement(demand, profile, table, policy=None):
    """Hard-constraint filter -> weighted score -> best node id or None.
    Kept from the old solver (fit/balance weights, policy-modulated) and
    extended for heterogeneity: accelerator-preferring services prefer
    GPU/TPU nodes, realtime services prefer high-capacity nodes."""
    p = {**DEFAULT_POLICY, **(policy or {})}
    cpu, mem = demand["cpu"], demand["mem"]
    forbid = p.get("forbid_classes", [])
    realtime = (profile or {}).get("latency_class") == "realtime"
    accel = bool(demand.get("accel_pref"))
    best, best_s = None, -1e18
    for nid in sorted(table):
        s = table[nid]
        if not s["healthy"] or s["cpu_free"] < cpu - EPS or s["mem_free"] < mem - EPS:
            continue
        if s["device_class"] in forbid:
            continue
        cap = s["cpu_cap"]
        sc = (p["w_fit"] * (1 - (s["cpu_free"] - cpu) / cap)
              + p["w_balance"] * (s["cpu_free"] / cap)
              + p["w_accel"] * (1.0 if accel and s.get("accelerator") else 0.0)
              + p["w_cap"] * (min(cap / 64.0, 1.0) if realtime else 0.0))
        if sc > best_s:
            best, best_s = nid, sc
    return best


class Sim:
    """Min-heap discrete-event scheduler."""

    def __init__(self):
        self.t, self._q, self._seq = 0.0, [], 0

    def at(self, t, fn, *a):
        heapq.heappush(self._q, (float(t), self._seq, fn, a))
        self._seq += 1

    def run(self, on_step=None):
        while self._q:
            t, _, fn, a = heapq.heappop(self._q)
            self.t = t
            if on_step:
                on_step(t)
            fn(*a)


class CapacityViolation(AssertionError):
    pass


class Node:
    def __init__(self, spec, zid):
        self.node_id, self.zone_id = spec["node_id"], zid
        self.device_class = spec["device_class"]
        self.cpu_cap, self.mem_cap = float(spec["cpu"]), float(spec["mem_gb"])
        self.power_w = float(spec.get("power_w", 0.0))
        self.accelerator = spec.get("accelerator")
        self.cpu_used = self.mem_used = 0.0
        self.healthy = True
        self.services = {}               # req_id -> (cpu, mem)

    cpu_free = property(lambda s: s.cpu_cap - s.cpu_used)
    mem_free = property(lambda s: s.mem_cap - s.mem_used)

    def can_fit(self, c, m):
        return self.healthy and self.cpu_free >= c - EPS and self.mem_free >= m - EPS

    def alloc(self, rid, c, m):
        if not self.can_fit(c, m) or rid in self.services:
            raise CapacityViolation(
                f"over-alloc {self.node_id}: {rid} needs cpu={c} mem={m}, free "
                f"cpu={self.cpu_free:.3f} mem={self.mem_free:.3f} healthy={self.healthy}")
        self.cpu_used += c
        self.mem_used += m
        self.services[rid] = (c, m)
        self._check()

    def release(self, rid):
        c, m = self.services.pop(rid)
        self.cpu_used -= c
        self.mem_used -= m
        if abs(self.cpu_used) < EPS:
            self.cpu_used = 0.0
        if abs(self.mem_used) < EPS:
            self.mem_used = 0.0
        self._check()
        return c, m

    def _check(self):
        if self.cpu_used < -EPS or self.mem_used < -EPS or \
                self.cpu_used > self.cpu_cap + EPS or self.mem_used > self.mem_cap + EPS:
            raise CapacityViolation(
                f"{self.node_id}: cpu_used={self.cpu_used} mem_used={self.mem_used} "
                f"caps=({self.cpu_cap},{self.mem_cap})")

    def fail(self):
        self.healthy = False
        lost = list(self.services)
        self.services.clear()
        self.cpu_used = self.mem_used = 0.0
        return lost

    def recover(self):
        self.healthy = True


class ZoneRuntime:
    """Per-zone capacity table + digest (< 500 bytes) for cross-zone
    exchange every DIGEST_S seconds."""

    def __init__(self, zid, nodes):
        self.zone_id = zid
        self.nodes = nodes           # node_id -> Node (shared with the sim)

    def table(self):
        return {nid: {"cpu_free": n.cpu_free if n.healthy else 0.0,
                      "mem_free": n.mem_free if n.healthy else 0.0,
                      "cpu_cap": n.cpu_cap, "mem_cap": n.mem_cap,
                      "healthy": n.healthy, "device_class": n.device_class,
                      "accelerator": n.accelerator}
                for nid, n in self.nodes.items()}

    def digest(self, t, prev_cpu_free):
        h = [n for n in self.nodes.values() if n.healthy]
        cur = sum(n.cpu_free for n in h)
        cap = sum(n.cpu_cap for n in self.nodes.values())
        top = sorted(((round(n.cpu_free, 2), round(n.mem_free, 2)) for n in h),
                     reverse=True)[:3]
        return {"zone": self.zone_id, "t": round(t, 2),
                "cpu_free": round(cur, 2),
                "mem_free": round(sum(n.mem_free for n in h), 2),
                "top_cpu_free": top[0][0] if top else 0.0,
                "top_mem_free": max((n.mem_free for n in h), default=0.0),
                "top_nodes": top,
                "util": round(1 - cur / cap, 3) if cap else 1.0,
                "trend": round(cur - prev_cpu_free, 2),
                "healthy_frac": round(len(h) / max(len(self.nodes), 1), 2)}


def digest_fits(d, demand):
    """Does the digest suggest *some* node in the zone fits the demand?"""
    return any(c >= demand["cpu"] - EPS and m >= demand["mem"] - EPS
               for c, m in d.get("top_nodes", []))


@dataclass
class Decision:
    node: Optional[str]
    path: str
    decision_source: str
    action: str = "place"
    demand: dict = field(default_factory=dict)   # what is actually allocated
    degrade_level: float = 1.0
    victims: List[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)  # orchestrator's belief, travels with the service
    note: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class ServiceInfo:
    """A running (or displaced) service as the orchestrator may see it.
    `profile` is the orchestrator's own belief from translation, not the
    hidden truth."""
    req_id: str
    origin_zone: str
    profile: dict
    demand: dict
    node: Optional[str] = None
    t_start: float = 0.0
    t_end: float = 0.0
    state: str = "PLACED"

    def view(self, now):
        return {"req_id": self.req_id,
                "service_type": self.profile.get("service_type"),
                "priority": self.profile.get("priority", "normal"),
                "remaining_s": round(max(self.t_end - now, 0.0), 1),
                "cpu": round(self.demand["cpu"], 3), "mem": round(self.demand["mem"], 3),
                "node": self.node}


class EdgeSimulation:
    def __init__(self, topo, wl, orch, tel, latency, catalog, fleet,
                 manifests, digest_s=5.0, policy_tick_s=120.0):
        from workload import lifetime_of
        self._lifetime_of = lifetime_of
        self.sim, self.topo, self.wl = Sim(), topo, wl
        self.orch, self.tel, self.lat = orch, tel, latency
        self.catalog, self.fleet, self.manifests = catalog, fleet, manifests
        self.digest_s, self.policy_tick_s = digest_s, policy_tick_s
        self.nodes, self.zones = {}, {}
        for z in topo["zones"]:
            zn = {}
            for spec in z["nodes"]:
                n = Node(spec, z["zone_id"])
                self.nodes[n.node_id] = zn[n.node_id] = n
            self.zones[z["zone_id"]] = ZoneRuntime(z["zone_id"], zn)
        self.zone_ids = [z["zone_id"] for z in topo["zones"]]
        self._z2g = {z["zone_id"]: z["zone_to_global_ms"] for z in topo["zones"]}
        self.services = {}               # req_id -> ServiceInfo
        self.truth = {r["req_id"]: r for r in wl["requests"]}
        self.digests, self._prev = {}, {z: 0.0 for z in self.zone_ids}
        self._pending = 0
        self.probe_count = 0
        self.oracle_calls = 0
        self.peak_util = 0.0
        if hasattr(orch, "attach"):
            orch.attach(self, latency)

    # ------------------------------------------------------------------
    # ClusterView API (what orchestrators may call)
    # ------------------------------------------------------------------
    @property
    def now(self):
        return self.sim.t

    def zone_table(self, zid):
        return self.zones[zid].table()

    def solve(self, zid, demand, profile, policy=None):
        self.probe_count += 1
        return solve_placement(demand, profile, self.zones[zid].table(), policy)

    def running(self, zid):
        out = []
        for n in self.zones[zid].nodes.values():
            if not n.healthy:
                continue
            for rid in n.services:
                svc = self.services.get(rid)
                if svc and svc.state == "PLACED":
                    out.append(svc.view(self.now))
        return out

    def node_zone(self, nid):
        return self.nodes[nid].zone_id

    def node_fits_after_evict(self, nid, victims, demand):
        n = self.nodes[nid]
        if not n.healthy:
            return False
        c = sum(n.services[v][0] for v in victims if v in n.services)
        m = sum(n.services[v][1] for v in victims if v in n.services)
        return n.cpu_free + c >= demand["cpu"] - EPS and n.mem_free + m >= demand["mem"] - EPS

    def manifest(self, req_id, resolved_type):
        r = self.truth.get(req_id)
        return self.manifests.demand(req_id, resolved_type, r["truth"] if r else None)

    def oracle_profile(self, req_id):
        """Ground-truth profile. ONLY for baselines that declare
        `oracle = True` (greedy_oracle: a placement-policy reference that is
        handed perfect, free translation). Every use is counted."""
        self.oracle_calls += 1
        tr = self.truth[req_id]["truth"]
        return {f: tr[f] for f in ("service_type", "latency_class", "data_locality", "priority")}

    def rtt(self, a, b):
        from scenario import rtt
        return rtt(self.topo, a, b)

    def zone_to_global_ms(self, zid):
        return self._z2g[zid]

    # ------------------------------------------------------------------
    # event plumbing
    # ------------------------------------------------------------------
    def _at_real(self, t, fn, *a):
        self._pending += 1

        def w(*args):
            self._pending -= 1
            fn(*args)
        self.sim.at(t, w, *a)

    def _clock(self, t):
        self.tel.now = t

    def run(self):
        for r in self.wl["requests"]:
            self._at_real(r["t_s"], self._arrival, r)
        for e in self.wl["events"]:
            if e["type"] == "node_failure":
                self._at_real(e["t_s"], self._fail, e)
            elif e["type"] == "register_service":
                self._at_real(e["t_s"], self._register, e)
        self.sim.at(0.0, self._digest_tick)
        self.sim.at(self.policy_tick_s, self._policy_tick)
        self.sim.run(on_step=self._clock)
        return self.tel

    # ------------------------------------------------------------------
    def _arrival(self, r):
        t = self.now
        dev = self.fleet[r["device_id"]]
        req = dev.send(r["req_id"], r["intent_text"], t)
        assert req.zone_id == r["origin_zone"], (req.zone_id, r["origin_zone"])
        tr = r["truth"]
        rec = self.tel.new_request(
            req_id=req.req_id, device_id=dev.device_id, origin_zone=req.zone_id,
            send_time=t, service_type_true=tr["service_type"],
            intent_text=req.text, phase=r["phase"],
            priority_true=tr["priority"], locality_true=tr["data_locality"])
        rec.add_latency("transport_in", self.lat.transport_in(dev.transport_ms))
        dec = self.orch.handle(t, req, rec)
        self._record_decision(rec, dec)
        if dec.node is None:
            self.tel.transition(rec.req_id, t, "REJECTED", "arrival")
            self.tel.finalize(rec.req_id, "rejected")
            return
        svc = ServiceInfo(req_id=rec.req_id, origin_zone=req.zone_id,
                          profile=dict(dec.profile),
                          demand=dict(dec.demand))
        life = self._lifetime_of(r, self.wl)
        self._commit(t, svc, dec, life, note="arrival", depth=0)

    def _record_decision(self, rec, dec):
        # ground-truth scoring happens here, in the simulator - agents never
        # see the truth
        tr = self.truth[rec.req_id]["truth"]
        prof = dec.profile or {}
        if prof.get("service_type") is not None:
            rec.service_type_resolved = prof["service_type"]
            rec.service_type_correct = prof["service_type"] == tr["service_type"]
            if all(f in prof for f in ("latency_class", "data_locality", "priority")):
                rec.translation_correct = all(prof[f] == tr[f] for f in
                                              ("service_type", "latency_class",
                                               "data_locality", "priority"))
        if rec.translation_source is None:
            rec.translation_source = "none"
        rec.path = rec.path or dec.path
        rec.decision_source = rec.decision_source or dec.decision_source
        rec.action = dec.action if dec.node is not None else "reject"
        if dec.node is not None:
            rec.node_final = dec.node
            rec.zone_final = self.nodes[dec.node].zone_id
            rec.cross_zone = rec.zone_final != rec.origin_zone
            rec.degraded = dec.degrade_level < 1.0 - 1e-9
            rec.degrade_level = float(dec.degrade_level)
            rec.victims = list(dec.victims)
            rec.cpu_alloc, rec.mem_alloc = dec.demand["cpu"], dec.demand["mem"]
            if rec.cross_zone:
                # request/deployment forwarded origin -> target zone: never 0
                rec.add_latency("cross_zone", self.lat.cross_zone(rec.origin_zone,
                                                                  rec.zone_final))

    def _commit(self, t, svc, dec, lifetime, note, depth):
        """Single choke point for executing a placement: evict victims
        first (so the freed capacity exists), allocate, schedule
        completion, then try to migrate each victim through the
        orchestrator's replan() (which may pre-empt again, bounded)."""
        evicted = [self._evict(t, v, by=svc.req_id,
                               by_priority=svc.profile.get("priority"))
                   for v in dec.victims]
        evicted = [e for e in evicted if e is not None]
        n = self.nodes[dec.node]
        n.alloc(svc.req_id, dec.demand["cpu"], dec.demand["mem"])
        self.peak_util = max(self.peak_util, n.cpu_used / n.cpu_cap)
        svc.node, svc.demand, svc.state = dec.node, dict(dec.demand), "PLACED"
        svc.t_start, svc.t_end = t, t + lifetime
        self.services[svc.req_id] = svc
        rec = self.tel.requests[svc.req_id]
        rec.add_latency("deployment", self.lat.deployment(n.device_class))
        self.tel.transition(svc.req_id, t, "PLACED", note)
        if note != "arrival":
            rec.node_final, rec.zone_final = dec.node, n.zone_id
        self._at_real(svc.t_end, self._complete, svc.req_id, svc.t_end)
        for ev in evicted:
            self._migrate(t, ev, depth + 1)

    def _evict(self, t, victim_id, by, by_priority=None):
        svc = self.services.get(victim_id)
        if svc is None or svc.state != "PLACED":
            return None
        n = self.nodes[svc.node]
        if victim_id not in n.services:
            return None
        n.release(victim_id)
        svc.state = "EVICTED"
        remaining = max(svc.t_end - t, 1.0)
        self.tel.log_event(t, "preempt", victim=victim_id, by=by, node=n.node_id,
                           victim_priority=svc.profile.get("priority"),
                           by_priority=by_priority)
        return (svc, remaining)

    def _migrate(self, t, evicted, depth):
        svc, remaining = evicted
        dec, scratch = None, _scratch(svc.req_id)
        if depth <= MAX_MIGRATION_DEPTH:
            dec = self.orch.replan(t, svc, scratch, reason="preempted")
        if dec is None or dec.node is None:
            svc.state = "DISPLACED"
            self.tel.transition(svc.req_id, t, "DISPLACED", "preempted")
            self.tel.finalize(svc.req_id, "preempt_unmigrated")
            self.tel.log_event(t, "victim_lost", req=svc.req_id)
            return
        self.tel.log_event(t, "migrate", req=svc.req_id, to_node=dec.node,
                           path=dec.path, cause="preempted",
                           replan_latency_ms=round(scratch.total_latency_ms, 3))
        self._commit(t, svc, dec, remaining, note="migrated", depth=depth)

    def _complete(self, rid, t_end):
        svc = self.services.get(rid)
        if svc is None or svc.state != "PLACED" or abs(svc.t_end - t_end) > 1e-6:
            return                        # superseded by eviction/failure/migration
        n = self.nodes[svc.node]
        if rid in n.services:
            n.release(rid)
        svc.state = "COMPLETED"
        self.tel.transition(rid, self.now, "COMPLETED", "lifetime_end")
        self.tel.finalize(rid, "completed")

    def _fail(self, ev):
        t = self.now
        n = self.nodes[ev["node_id"]]
        if not n.healthy:
            return
        lost = n.fail()
        self.tel.log_event(t, "node_failure", node=n.node_id, zone=n.zone_id,
                           device_class=n.device_class, n_displaced=len(lost),
                           recover_t=ev["recover_t_s"], forced=ev.get("forced", False))
        if hasattr(self.orch, "on_failure"):
            self.orch.on_failure(t, n.zone_id)
        for rid in lost:
            svc = self.services[rid]
            remaining = max(svc.t_end - t, 1.0)
            svc.state = "INTERRUPTED"
            scratch = _scratch(rid)
            dec = self.orch.replan(t, svc, scratch, reason="node_failure")
            if dec is not None and dec.node is not None:
                self.tel.log_event(t, "migrate", req=rid, to_node=dec.node,
                                   path=dec.path, cause="node_failure",
                                   replan_latency_ms=round(scratch.total_latency_ms, 3))
                self._commit(t, svc, dec, remaining, note="replanned", depth=0)
            else:
                svc.state = "DISPLACED"
                self.tel.transition(rid, t, "DISPLACED", "node_failure")
                self.tel.finalize(rid, "node_loss")
        self._at_real(ev["recover_t_s"], self._recover, n.node_id)

    def _recover(self, nid):
        self.nodes[nid].recover()
        self.tel.log_event(self.now, "node_recover", node=nid)

    def _register(self, ev):
        self.catalog.register(ev["name"], ev["spec"], self.now)
        self.tel.log_event(self.now, "register_service", name=ev["name"])
        if hasattr(self.orch, "on_catalog_update"):
            self.orch.on_catalog_update(self.now, ev["name"])

    def _digest_tick(self):
        t = self.now
        for zid, zr in self.zones.items():
            d = zr.digest(t, self._prev[zid])
            self._prev[zid] = d["cpu_free"]
            self.digests[zid] = d
        if hasattr(self.orch, "on_digest_tick"):
            self.orch.on_digest_tick(t, {k: dict(v) for k, v in self.digests.items()})
        utils = [1 - n.cpu_free / n.cpu_cap for n in self.nodes.values() if n.healthy]
        stats = self.orch.stats() if hasattr(self.orch, "stats") else {}
        self.tel.snapshot(t, util_zone=[self.digests[z]["util"] for z in self.zone_ids],
                          util_node_var=round(float(_var(utils)), 5),
                          running=sum(len(n.services) for n in self.nodes.values()),
                          probes=self.probe_count, **stats)
        if self._pending > 0:
            self.sim.at(t + self.digest_s, self._digest_tick)

    def _policy_tick(self):
        if hasattr(self.orch, "on_policy_tick"):
            self.orch.on_policy_tick(self.now)
        if self._pending > 0:
            self.sim.at(self.now + self.policy_tick_s, self._policy_tick)

    def check_invariants(self):
        for n in self.nodes.values():
            n._check()
            used = sum(c for c, _ in n.services.values())
            if abs(used - n.cpu_used) > 1e-4:
                raise CapacityViolation(f"{n.node_id}: bookkeeping drift")
        return True


def _scratch(req_id):
    from orchestrator import ScratchRecord
    return ScratchRecord(req_id)


def _var(xs):
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


class Manifests:
    """Service registry lookup ("deployment manifest"): once a request's
    service type has been resolved, its container resource request is
    known at placement time.

    * resolved type == true type -> the request's own resource request
      (per-request size drawn from the Alibaba distribution);
    * resolved to a different registered type -> that type's median
      manifest (the wrong service gets deployed: translation error cost);
    * unregistered / novel -> a generic manifest (median over the
      catalog's registered types).
    This is the only place hidden truth touches a decision, and only as
    the size of whatever the orchestrator itself chose to deploy."""

    def __init__(self, catalog, resources):
        self.catalog, self.resources = catalog, resources

    def demand(self, req_id, resolved_type, truth):
        if truth is not None and resolved_type == truth["service_type"]:
            return {"cpu": float(truth["cpu"]), "mem": float(truth["mem"]),
                    "accel_pref": bool(truth.get("accel_pref"))}
        if resolved_type and self.catalog.known(resolved_type):
            spec = self.catalog.types[resolved_type]
            c, m = self.resources.median_size(spec)
            return {"cpu": c, "mem": m, "accel_pref": bool(spec.get("accel_pref"))}
        sizes = [self.resources.median_size(s) for s in self.catalog.types.values()]
        return {"cpu": float(sorted(c for c, _ in sizes)[len(sizes) // 2]),
                "mem": float(sorted(m for _, m in sizes)[len(sizes) // 2]),
                "accel_pref": False}


def realtime(profile):
    return profile.get("latency_class") == "realtime"


def lower_priority(a, b):
    """True if priority a is strictly lower than b."""
    return PRIORITY_RANK.get(a, 1) < PRIORITY_RANK.get(b, 1)


__all__ = ["solve_placement", "Sim", "Node", "ZoneRuntime", "EdgeSimulation",
           "Decision", "ServiceInfo", "Manifests", "digest_fits", "EPS",
           "CapacityViolation", "latency_class_of", "lower_priority"]
