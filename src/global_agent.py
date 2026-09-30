"""Tier 3 - the global agent.

Escalation chain for a request its zone agent could not place locally:

  1. episodic (similar-case) memory  - no LLM call if a good match exists
  2. procedural rules                - distilled from episodic memory
  3. cross-zone digest scan          - deterministic, latency-aware
  4. ONE structured LLM decision     - {action: place|preempt|degrade|reject,
                                        zone, victim, degrade_level, reason}
     verified against the live capacity solver; one retry with the
     verification error as feedback; then
  5. the deterministic rule chain    - place -> pre-empt -> degrade -> reject

Old-repo flaws fixed here:
  #4 "similar case" meant only same service type -> CaseMemory keys on
     (service_type, latency_class, priority, cpu-size bucket, origin zone),
     stores the chosen zone/node/action and the free capacity seen at the
     time, and re-checks the candidate against the live digest (or a
     probe, when digests are off) before reuse, falling through if stale.
     Negative cases (zones that failed to fit) are stored too and avoided.
  #5 two separate LLM calls (zone choice, then eviction choice) -> one call.
  #6 a fixed 60%/80% degradation rule the LLM never saw -> an explicit
     menu (100/80/60/40% of requested CPU+mem, never below the service
     type's floor) the LLM chooses from inside the same call.

Every mechanism can be switched off independently (ablations.py), and
each one's cost (probes, LLM calls, tokens, latency) lands in telemetry.
"""
import json
import math
from collections import Counter, defaultdict, deque

from latency_model import measure
from llm_client import LLMUnavailable
from scenario import PRIORITY_RANK
from sim_engine import Decision, digest_fits

DEGRADE_LEVELS = (0.8, 0.6, 0.4)          # 1.0 = no degradation
MAX_XZ_CAND = 3                          # digest-scan probes per escalation
MAX_PREEMPT_OPTIONS = 6
MAX_PLACE_OPTIONS = 6
MAX_FEAS_ZONES = 4                        # zones checked in the pre-verification round trip
MEMORY_CANDIDATES = 3                     # remembered cases tried per escalation
PREEMPT_MIN_PRIORITY = "high"            # only high/critical requests may pre-empt
RULE_MIN_EPS, RULE_MIN_FRAC = 5, 0.6
EPISODIC_PER_KEY = 10
NEGATIVE_TTL_S = 60.0

DECIDE_SYS = """You are the global coordinator of a three-tier edge orchestrator. A service
request could not be placed by its zone agent. Choose exactly ONE action.
Respond with ONLY a JSON object:
{"action": "place"|"preempt"|"degrade"|"reject", "zone": "<zone id or null>",
 "victim": "<req_id or null>", "degrade_level": <number or null>, "reason": "<one sentence>"}
Rules:
- "place": zone must be one of options.place (zones whose capacity digest suggests a node fits).
- "preempt": victim must be one of options.preempt (strictly lower-priority running services whose
  eviction frees enough room on their node); zone = that victim's zone. The victim is migrated
  elsewhere if possible, otherwise it is lost.
- "degrade": run the request with a fraction of its CPU/memory; (zone, degrade_level) must be an
  entry of options.degrade. Levels below the service's floor are never offered.
- "reject": when no option is worth taking.
Guidance: placing without harm beats pre-emption or degradation; latency-sensitive (realtime)
services should stay close (low rtt_ms); prefer evicting services that are low priority, nearly
finished and interruption-tolerant; degrade only services that tolerate reduced resources;
avoid zones listed in memory_hints.failed_zones."""

RULE_SYS = """Author ONE placement rule from episodic evidence. Respond ONLY with JSON:
{"condition": {"service_type": "<type>"}, "action": {"prefer_zone": "<zone_id>"},
 "rationale": "<one sentence>"}
The rule should reflect the zone where this service type was most often placed successfully."""


def cpu_bucket(cpu):
    return int(math.floor(math.log2(max(cpu, 0.0625) * 4)))


class CaseMemory:
    """Episodic similar-case memory with negative cases."""

    def __init__(self):
        self.pos = defaultdict(lambda: deque(maxlen=EPISODIC_PER_KEY))
        self.neg = {}                     # (key, zone) -> t
        self.n_stale = 0
        self.n_reused = 0

    @staticmethod
    def key(profile, demand, origin):
        return (profile.get("service_type"), profile.get("latency_class"),
                profile.get("priority"), cpu_bucket(demand["cpu"]), origin)

    def record(self, key, zone, node, action, level, victim_priority, free_seen, t):
        self.pos[key].append({"zone": zone, "node": node, "action": action,
                              "level": level, "victim_priority": victim_priority,
                              "free_seen": free_seen, "t": round(t, 1)})
        self.neg.pop((key, zone), None)

    def record_negative(self, key, zone, t):
        self.neg[(key, zone)] = t

    def is_negative(self, key, zone, t):
        tn = self.neg.get((key, zone))
        return tn is not None and t - tn <= NEGATIVE_TTL_S

    def failed_zones(self, key, t):
        return sorted(z for (k, z), tn in self.neg.items()
                      if k == key and t - tn <= NEGATIVE_TTL_S)

    def candidates(self, key, allowed, t, k=2):
        """Exact-key cases first (most recent first), then similar keys:
        same type/latency/priority, cpu bucket within 1, any origin -
        ranked by (same origin, bucket distance, recency)."""
        out, seen = [], set()
        for c in reversed(self.pos.get(key, ())):
            if c["zone"] in allowed and c["zone"] not in seen \
                    and not self.is_negative(key, c["zone"], t):
                out.append(c)
                seen.add(c["zone"])
        if len(out) < k:
            sims = []
            for k2, cases in self.pos.items():
                if k2 == key or k2[:3] != key[:3] or abs(k2[3] - key[3]) > 1:
                    continue
                for c in cases:
                    if c["zone"] in allowed and c["zone"] not in seen \
                            and not self.is_negative(key, c["zone"], t):
                        sims.append((k2[4] != key[4], abs(k2[3] - key[3]), -c["t"], c))
            for *_, c in sorted(sims, key=lambda x: x[:3]):
                if c["zone"] not in seen:
                    out.append(c)
                    seen.add(c["zone"])
        return out[:k]

    def size(self):
        return sum(len(v) for v in self.pos.values())

    def by_type(self):
        agg = defaultdict(Counter)
        for k, cases in self.pos.items():
            for c in cases:
                agg[k[0]][c["zone"]] += 1
        return agg


class ProceduralStore:
    def __init__(self):
        self.rules = {}

    def match(self, service_type):
        return self.rules.get(service_type)

    def add(self, rule):
        self.rules[rule["condition"]["service_type"]] = rule


class GlobalAgent:
    def __init__(self, llm, catalog, zone_ids, *, use_memory=True, use_digest=True,
                 use_llm=True, use_preempt_degrade=True, use_procedural=True,
                 stale_s=15.0):
        self.llm, self.catalog, self.zone_ids = llm, catalog, list(zone_ids)
        self.use_memory, self.use_digest, self.use_llm = use_memory, use_digest, use_llm
        self.use_pd, self.use_procedural = use_preempt_degrade, use_procedural and use_memory
        self.stale_s = stale_s
        self.mem, self.proc = CaseMemory(), ProceduralStore()
        self.digests = {}
        self.counters = Counter()
        self._skip_place = set()
        self._no_options = False
        self._opt_nodes = {}

    # --- cluster-state inputs -----------------------------------------------
    def on_digest_tick(self, t, digests):
        if self.use_digest:
            self.digests = digests

    def _fresh_digest(self, zone, t):
        d = self.digests.get(zone)
        return d if d is not None and t - d["t"] <= self.stale_s else None

    # --- probes (each is a global->zone round trip) ----------------------
    def _probe(self, view, rec, zone, demand, profile, policy=None):
        self.counters["probes"] += 1
        rec.add_latency("escalation", view.zone_to_global_ms(zone))
        nid, ms = measure(view.solve, zone, demand, profile, policy)
        rec.add_latency("decision", ms)
        return nid

    # --- main entry ---------------------------------------------------------
    def escalate(self, t, view, rid, origin, profile, demand, allowed, rec,
                 policies=None, charge_hop=True, tried_local=False):
        """tried_local: the zone agent just failed a *live* full-size solve
        in `origin`, so a full-size placement there is not re-offered
        (a digest may be up to DIGEST_S stale); pre-emption / degradation
        in the origin zone remain possible."""
        policies = policies or {}
        self._skip_place = {origin} if tried_local else set()
        self.counters["escalations"] += 1
        if charge_hop:
            rec.add_latency("escalation", view.zone_to_global_ms(origin))
        key = CaseMemory.key(profile, demand, origin)
        local_only = allowed == [origin]

        for step in (self._from_memory, self._from_rules, self._from_digest):
            dec = step(t, view, rid, origin, profile, demand, allowed, rec, key, policies)
            if dec is not None:
                return self._remember(key, dec, t, profile)

        if self.use_llm:
            self._no_options = False
            dec = self._from_llm(t, view, rid, origin, profile, demand, allowed, rec, key,
                                 policies, local_only)
            if dec is not None:
                return self._remember(key, dec, t, profile)   # verified place/preempt/degrade/reject
            if self._no_options and self.use_digest:
                # nothing feasible on the current state: the rule chain would
                # re-issue the same queries and find nothing either
                return Decision(node=None, path="unresolved", decision_source="rule",
                                action="reject", demand=dict(demand), profile=dict(profile))
        dec = self._rule_chain(t, view, rid, origin, profile, demand, allowed, rec, key,
                               policies, local_only)
        return self._remember(key, dec, t, profile)

    def _remember(self, key, dec, t, profile):
        if self.use_memory and dec.node is not None:
            zone = dec.meta.get("zone")
            d = self.digests.get(zone, {})
            self.mem.record(key, zone, dec.node, dec.action, dec.degrade_level,
                            dec.meta.get("victim_priority"), d.get("cpu_free"), t)
        return dec

    def _dec(self, view, node, path, source, profile, demand, action="place",
             level=1.0, victims=(), victim_priority=None):
        return Decision(node=node, path=path, decision_source=source, action=action,
                        demand={**demand, "cpu": round(demand["cpu"] * level, 4),
                                "mem": round(demand["mem"] * level, 4)},
                        degrade_level=level, victims=list(victims), profile=dict(profile),
                        meta={"zone": view.node_zone(node) if node else None,
                              "victim_priority": victim_priority})

    # --- 1. episodic memory ---------------------------------------------------
    def _from_memory(self, t, view, rid, origin, profile, demand, allowed, rec, key, pol):
        if not self.use_memory:
            return None
        cands, ms = measure(self.mem.candidates, key, allowed, t, MEMORY_CANDIDATES)
        rec.add_latency("decision", ms)
        for c in cands:
            zone, level = c["zone"], c.get("level", 1.0) or 1.0
            if c["action"] == "place" and zone in self._skip_place:
                continue
            want = {**demand, "cpu": demand["cpu"] * level, "mem": demand["mem"] * level} \
                if c["action"] == "degrade" else demand
            if self.use_digest and c["action"] != "preempt":
                d = self._fresh_digest(zone, t)
                if d is None or not digest_fits(d, want):
                    self.mem.n_stale += 1                  # stale: capacity since consumed
                    self.counters["memory_stale"] += 1
                    continue
            if c["action"] == "preempt":
                if not self.use_pd:
                    continue
                v = self._pick_victim(view, rec, zone, profile, demand,
                                      max_priority=c.get("victim_priority"))
                if v is None:
                    self.counters["memory_stale"] += 1
                    continue
                self.mem.n_reused += 1
                return self._dec(view, v["node"], "episodic", "memory", profile, demand,
                                 action="preempt", victims=[v["req_id"]],
                                 victim_priority=v["priority"])
            if c["action"] == "degrade" and not self.use_pd:
                continue
            nid = self._probe(view, rec, zone, demand, profile, pol.get(zone))
            action, lvl = "place", 1.0
            if nid is None and c["action"] == "degrade":
                nid = self._probe(view, rec, zone, want, profile, pol.get(zone))
                action, lvl = "degrade", level
            if nid is None:
                self.mem.record_negative(key, zone, t)
                self.counters["memory_stale"] += 1
                continue
            self.mem.n_reused += 1
            return self._dec(view, nid, "episodic", "memory", profile, demand,
                             action=action, level=lvl)
        return None

    # --- 2. procedural rules -------------------------------------------------------
    def _from_rules(self, t, view, rid, origin, profile, demand, allowed, rec, key, pol):
        if not self.use_procedural:
            return None
        rule = self.proc.match(profile.get("service_type"))
        if not rule:
            return None
        zone = rule["action"]["prefer_zone"]
        if zone not in allowed or zone in self._skip_place or self.mem.is_negative(key, zone, t):
            return None
        if self.use_digest:
            d = self._fresh_digest(zone, t)
            if d is None or not digest_fits(d, demand):
                return None
        nid = self._probe(view, rec, zone, demand, profile, pol.get(zone))
        if nid is None:
            self.mem.record_negative(key, zone, t)
            return None
        return self._dec(view, nid, "procedural", "memory", profile, demand)

    # --- 3. digest scan ------------------------------------------------------------
    def _from_digest(self, t, view, rid, origin, profile, demand, allowed, rec, key, pol):
        if not self.use_digest:
            return None

        def scan():
            c = []
            for z in allowed:
                if z in self._skip_place:
                    continue
                d = self._fresh_digest(z, t)
                if d is not None and digest_fits(d, demand) and \
                        not self.mem.is_negative(key, z, t):
                    c.append((view.rtt(origin, z), -d["top_cpu_free"], z))
            return [z for *_, z in sorted(c)]
        cands, ms = measure(scan)
        rec.add_latency("decision", ms)
        for z in cands[:MAX_XZ_CAND]:
            nid = self._probe(view, rec, z, demand, profile, pol.get(z))
            if nid is not None:
                return self._dec(view, nid, "digest", "digest", profile, demand)
            self.mem.record_negative(key, z, t)
        return None

    # --- options offered to the LLM (and used by the rule chain) --------------
    def _preempt_allowed(self, profile):
        return self.use_pd and PRIORITY_RANK.get(profile.get("priority"), 1) >= \
            PRIORITY_RANK[PREEMPT_MIN_PRIORITY]

    def _victims(self, view, rec, zones, profile, demand, max_priority=None, charge=True):
        """Running, strictly-lower-priority services whose eviction alone
        frees enough room on their node. Querying a zone's running list is
        a round trip; zones are queried in parallel (max RTT charged)."""
        if not zones:
            return []
        if charge:
            rec.add_latency("escalation", max(view.zone_to_global_ms(z) for z in zones))
        pr = PRIORITY_RANK.get(profile.get("priority"), 1)
        cap = PRIORITY_RANK.get(max_priority, 99) if max_priority else 99
        out = []
        for z in zones:
            for v in view.running(z):
                vr = PRIORITY_RANK.get(v["priority"], 1)
                if vr < pr and vr <= cap and \
                        view.node_fits_after_evict(v["node"], [v["req_id"]], demand):
                    out.append({**v, "zone": z})
        out.sort(key=lambda v: (PRIORITY_RANK.get(v["priority"], 1), v["remaining_s"]))
        return out

    def _pick_victim(self, view, rec, zone, profile, demand, max_priority=None):
        vs = self._victims(view, rec, [zone], profile, demand, max_priority)
        return vs[0] if vs else None

    def _options(self, t, view, rec, origin, profile, demand, allowed, key):
        """Options offered to the LLM, *pre-verified*: one parallel feasibility
        round trip to the nearest promising zones asks each zone's solver
        whether the request fits at full size, else at the highest degradation
        level the service allows, and (for high/critical requests) which lower-
        priority running services could be evicted. The LLM therefore only
        chooses among actions that are feasible right now, so a failed
        verification means a genuine model error, not a stale digest."""
        near = sorted(allowed, key=lambda z: view.rtt(origin, z))
        floor = self.catalog.degrade_floor(profile.get("service_type"))
        levels = [1.0] + [l for l in DEGRADE_LEVELS if l + 1e-9 >= floor] if self.use_pd else [1.0]
        lowest = {"cpu": demand["cpu"] * levels[-1], "mem": demand["mem"] * levels[-1]}
        cand = []
        for z in near:
            if self.use_digest:
                d = self._fresh_digest(z, t)
                if d is None or not digest_fits(d, lowest):
                    continue                     # not even the smallest level can fit
            cand.append(z)
            if len(cand) >= MAX_FEAS_ZONES:
                break
        want_victims = self._preempt_allowed(profile)
        victim_zones = near[:4] if want_victims else []
        queried = list(dict.fromkeys(cand + victim_zones))
        if queried:                              # one parallel round trip
            rec.add_latency("escalation", max(view.zone_to_global_ms(z) for z in queried))
        self._opt_nodes = {}
        place, degrade = [], []

        def feasibility():
            for z in cand:
                self.counters["probes"] += 1
                for lvl in levels:
                    if lvl == 1.0 and z in self._skip_place:
                        continue
                    want = {**demand, "cpu": demand["cpu"] * lvl, "mem": demand["mem"] * lvl}
                    nid = view.solve(z, want, profile)
                    if nid is not None:
                        if lvl == 1.0:
                            place.append(z)
                        else:
                            degrade.append({"zone": z, "level": lvl})
                        self._opt_nodes[(z, lvl)] = nid
                        break                    # lower levels are dominated
        _, ms = measure(feasibility)
        rec.add_latency("decision", ms)
        preempt = self._victims(view, rec, victim_zones, profile, demand,
                                charge=False)[:MAX_PREEMPT_OPTIONS] if want_victims else []
        return {"place": place[:MAX_PLACE_OPTIONS],
                "preempt": [{"victim": v["req_id"], "zone": v["zone"], "node": v["node"],
                             "priority": v["priority"], "service_type": v["service_type"],
                             "remaining_s": v["remaining_s"], "cpu": v["cpu"], "mem": v["mem"]}
                            for v in preempt],
                "degrade": degrade[:MAX_PLACE_OPTIONS]}

    # --- 4. one structured LLM call -------------------------------------------------
    def _prompt(self, t, view, origin, profile, demand, allowed, options, key, feedback):
        zones = {}
        for z in allowed:
            info = {"rtt_ms": round(view.rtt(origin, z), 2)}
            d = self._fresh_digest(z, t) if self.use_digest else None
            if d is not None:
                info.update(cpu_free=d["cpu_free"], top_nodes=d["top_nodes"],
                            util=d["util"], healthy_frac=d["healthy_frac"])
            zones[z] = info
        body = {"request": {"profile": profile, "cpu": round(demand["cpu"], 3),
                            "mem": round(demand["mem"], 3), "origin_zone": origin,
                            "degrade_floor": self.catalog.degrade_floor(profile.get("service_type"))},
                "zones": zones, "options": options,
                "memory_hints": {"failed_zones": self.mem.failed_zones(key, t)
                                 if self.use_memory else []}}
        if feedback:
            body["feedback"] = feedback
        return json.dumps(body, sort_keys=True)

    def _verify(self, out, options, view, rec, profile, demand, pol, local_only):
        """Returns (Decision or None, error string)."""
        a = out.get("action")
        if a == "place":
            z = out.get("zone")
            if z not in options["place"]:
                return None, f"zone {z!r} is not in options.place"
            nid = self._opt_nodes.get((z, 1.0))
            if nid is None or not view.node_fits_after_evict(nid, [], demand):
                return None, f"zone {z} has no node that fits cpu={demand['cpu']:.2f} mem={demand['mem']:.2f}"
            return self._dec(view, nid, "llm", "llm", profile, demand), ""
        if a == "preempt":
            v = next((o for o in options["preempt"] if o["victim"] == out.get("victim")), None)
            if v is None:
                return None, f"victim {out.get('victim')!r} is not in options.preempt"
            if not view.node_fits_after_evict(v["node"], [v["victim"]], demand):
                return None, f"evicting {v['victim']} no longer frees enough room"
            return self._dec(view, v["node"], "preempt_local" if local_only else "preempt",
                             "llm", profile, demand, action="preempt", victims=[v["victim"]],
                             victim_priority=v["priority"]), ""
        if a == "degrade":
            try:
                lvl = float(out.get("degrade_level"))
            except (TypeError, ValueError):
                return None, "degrade_level missing or not a number"
            z = out.get("zone")
            ok = any(o["zone"] == z and abs(o["level"] - lvl) < 1e-6 for o in options["degrade"])
            if not ok:
                return None, f"(zone={z}, level={lvl}) is not in options.degrade"
            want = {**demand, "cpu": demand["cpu"] * lvl, "mem": demand["mem"] * lvl}
            nid = next((n for (zz, l), n in self._opt_nodes.items()
                        if zz == z and abs(l - lvl) < 1e-6), None)
            if nid is None or not view.node_fits_after_evict(nid, [], want):
                return None, f"zone {z} cannot fit the request even at level {lvl}"
            return self._dec(view, nid, "llm_degraded", "llm", profile, demand,
                             action="degrade", level=lvl), ""
        if a == "reject":
            if options["place"]:
                return None, "reject is not allowed while options.place is non-empty"
            return Decision(node=None, path="unresolved", decision_source="llm",
                            action="reject", demand=dict(demand), profile=dict(profile)), ""
        return None, f"unknown action {a!r}"

    def _from_llm(self, t, view, rid, origin, profile, demand, allowed, rec, key, pol,
                  local_only):
        options = self._options(t, view, rec, origin, profile, demand, allowed, key)
        if not (options["place"] or options["preempt"] or options["degrade"]):
            self.counters["llm_skipped_no_options"] += 1
            self._no_options = True
            return None
        rec.reached_llm_stage = True
        self.counters["llm_stage"] += 1
        feedback = ""
        for attempt in range(2):
            user = self._prompt(t, view, origin, profile, demand, allowed, options, key,
                                feedback)
            try:
                res = self.llm.ask(DECIDE_SYS, user, kind="decide", req_id=rid)
            except LLMUnavailable:
                self.counters["llm_unavailable"] += 1
                return None
            rec.add_latency("decision", res.sim_ms)
            dec, err = self._verify(res.data, options, view, rec, profile, demand, pol,
                                    local_only)
            if dec is not None:
                dec.decision_source = "llm_fresh" if res.source == "fresh" else "llm_cached_disk"
                rec.llm_verified = True
                rec.llm_retries = attempt
                self.counters[f"llm_{dec.action}"] += 1
                return dec
            feedback = f"Your previous answer {json.dumps(res.data)[:200]} was rejected: {err}."
            self.counters["llm_invalid"] += 1
        rec.llm_verified = False
        rec.llm_retries = 1
        return None

    # --- 5. deterministic rule chain --------------------------------------------------
    def _rule_chain(self, t, view, rid, origin, profile, demand, allowed, rec, key, pol,
                    local_only):
        rec.fallback_used = rec.reached_llm_stage or rec.fallback_used
        self.counters["rule_chain"] += 1
        near = sorted(allowed, key=lambda z: view.rtt(origin, z))
        probed = set()
        if not self.use_digest:
            # no digests: the chain has to probe blind, nearest first
            for z in [z for z in near if z not in self._skip_place][:MAX_XZ_CAND + 1]:
                probed.add(z)
                nid = self._probe(view, rec, z, demand, profile, pol.get(z))
                if nid:
                    return self._dec(view, nid, "digest" if z != origin else "local",
                                     "rule", profile, demand)
        if self.use_pd:
            if self._preempt_allowed(profile):
                vs = self._victims(view, rec, near[:4], profile, demand)
                if vs:
                    v = vs[0]
                    return self._dec(view, v["node"], "preempt_local" if local_only else "preempt",
                                     "rule", profile, demand, action="preempt",
                                     victims=[v["req_id"]], victim_priority=v["priority"])
            floor = self.catalog.degrade_floor(profile.get("service_type"))
            for lvl in DEGRADE_LEVELS:
                if lvl + 1e-9 < floor:
                    break
                want = {**demand, "cpu": demand["cpu"] * lvl, "mem": demand["mem"] * lvl}
                for z in near[:3]:
                    if self.use_digest:
                        d = self._fresh_digest(z, t)
                        if d is None or not digest_fits(d, want):
                            continue
                    nid = self._probe(view, rec, z, want, profile, pol.get(z))
                    if nid:
                        return self._dec(view, nid, "llm_degraded", "rule", profile, demand,
                                         action="degrade", level=lvl)
        return Decision(node=None, path="unresolved", decision_source="rule",
                        action="reject", demand=dict(demand), profile=dict(profile))

    # --- policy tick: author procedural rules -------------------------------------------
    def on_policy_tick(self, t, rec_llm_agent="global"):
        if not self.use_procedural:
            return
        for st, zc in self.mem.by_type().items():
            if self.proc.match(st):
                continue
            n = sum(zc.values())
            zone, cnt = zc.most_common(1)[0]
            frac = cnt / max(n, 1)
            if cnt < RULE_MIN_EPS or frac < RULE_MIN_FRAC:
                continue
            rule = None
            if self.use_llm:
                try:
                    res = self.llm.ask(RULE_SYS, json.dumps(
                        {"service_type": st, "successful_zone_counts": dict(zc)},
                        sort_keys=True), kind="rule_author", req_id=None)
                    rule = res.data
                except LLMUnavailable:
                    rule = None
            if not isinstance(rule, dict) or \
                    (rule.get("action") or {}).get("prefer_zone") != zone:
                self.counters["rule_llm_overridden"] += 1
                rule = {"condition": {"service_type": st}, "action": {"prefer_zone": zone},
                        "rationale": "episodic majority (validated)"}
            rule["condition"] = {"service_type": st}
            rule.update(confidence=round(frac, 3), provenance={"episodes": cnt, "t": round(t, 1)})
            self.proc.add(rule)
            self.counters["rules_authored"] += 1

    def stats(self):
        return {"memory_cases": self.mem.size(), "memory_rules": len(self.proc.rules),
                "memory_stale": self.mem.n_stale, "memory_reused": self.mem.n_reused,
                **{f"g_{k}": v for k, v in self.counters.items()}}
