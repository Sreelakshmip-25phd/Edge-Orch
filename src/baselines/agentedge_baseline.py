"""AgentEdge-style baseline: a four-role sequential LLM pipeline with a
simulate-before-execute validator (mirroring ActSimCrit's
simulate-then-critic loop), adapted to this environment.

  1. intent role          LLM: NL intent -> {service_type, latency_class,
                          data_locality, priority}             kind="intent"
  2. observability role   LLM: raw zone digests -> situation report
                          {candidate_zones, hotspots, summary}  kind="observe"
  3. planning role        LLM: -> {action place|preempt|degrade|reject,
                          zone, victim, degrade_level}          kind="plan"
  4. infra-action role    LLM: plan + the target zone's live node table ->
                          concrete {zone, node, victim, degrade_level}  kind="act"
  5. validator            the deterministic capacity solver/constraint check
                          stands in for the digital twin: the concrete action
                          is simulated first and only committed if feasible
                          (and locality-respecting); otherwise the failure
                          goes back to the planning role as critic feedback
                          (at most MAX_REPLANS re-plans).

No memory and no cache: the cost-reduction mechanism here is "simulation
catches bad plans", not "fewer LLM calls", so its LLM calls per request are
expected to stay flat over time (>= 4 per request) - metrics.py shows this
next to the proposed system's falling curve.
"""
from global_agent import OBJECTIVE
from llm_client import LLMUnavailable
from scenario import PRIORITY_RANK
from sim_engine import Decision
from zone_agent import resolve_profile, slm_system_prompt

from .common import ToolEnv, dumps, finish_decision

MAX_REPLANS = 2
DEGRADE_LEVELS = (0.8, 0.6, 0.4)

OBSERVE_SYS = """You are the observability agent of an edge orchestrator. Summarise the cluster
state for the planner. Respond with ONLY a JSON object:
{"candidate_zones": [<up to 4 zone ids best suited to host the request, best first>],
 "hotspots": [<zone ids that are overloaded or unhealthy>], "summary": "<one sentence>"}"""

PLAN_SYS = """You are the planning agent of an edge orchestrator. Decide how to serve the request.
""" + OBJECTIVE + """
Respond with ONLY a JSON object:
{"action": "place"|"preempt"|"degrade"|"reject", "zone": "<zone id>", "victim": "<req_id or null>",
 "degrade_level": <1.0|0.8|0.6|0.4>, "reason": "<one sentence>"}
Only allowed_zones may be used. A victim must come from preempt_candidates. The degrade level
may not go below degrade_floor. If feedback lists earlier plans that failed validation, do not
repeat them."""

ACT_SYS = """You are the infrastructure-action agent of an edge orchestrator. Turn the plan into
one concrete deployment on a specific node of the planned zone. Respond with ONLY a JSON object:
{"zone": "<zone id>", "node": "<node id>", "victim": "<req_id or null>", "degrade_level": <number>}"""


class AgentEdgeBaseline:
    name = "agentedge"
    uses_llm = True

    def __init__(self, *, catalog, embedder, make_llm, **_):
        self.catalog, self.emb = catalog, embedder
        self.llm = make_llm("llm", "agentedge")
        self.view, self.digests = None, {}
        self.counters = {"validations": 0, "validation_failures": 0, "replans": 0}

    def attach(self, view, latency):
        self.view = view

    def on_digest_tick(self, t, digests):
        self.digests = digests

    def _ask(self, sys, body, kind, rid, rec, component="decision"):
        res = self.llm.ask(sys, body if isinstance(body, str) else dumps(body),
                           kind=kind, req_id=rid)
        rec.add_latency(component, res.sim_ms)
        rec.reached_llm_stage = True
        return res

    def _pipeline(self, t, rid, origin, prof, rec, env, demand):
        allowed = [origin] if prof.get("data_locality") == "zone_local" else list(self.view.zone_ids)
        zones = {z: {"rtt_ms": round(self.view.rtt(origin, z), 2),
                     **{k: d[k] for k in ("cpu_free", "top_nodes", "util", "healthy_frac")}}
                 for z, d in self.digests.items() if z in allowed and t - d["t"] <= 15.0}
        req = {"profile": prof, "cpu": round(demand["cpu"], 3), "mem": round(demand["mem"], 3),
               "origin_zone": origin}
        obs = self._ask(OBSERVE_SYS, {"request": req, "zones": zones}, "observe", rid, rec).data
        pre = []
        if PRIORITY_RANK.get(prof.get("priority"), 1) >= PRIORITY_RANK["high"]:
            cz = [z for z in (obs.get("candidate_zones") or []) if z in allowed][:3] or [origin]
            rec.add_latency("escalation", max(self.view.zone_to_global_ms(z) for z in cz))
            for z in cz:
                for v in self.view.running(z):
                    if PRIORITY_RANK.get(v["priority"], 1) < PRIORITY_RANK.get(prof["priority"], 1) \
                            and self.view.node_fits_after_evict(v["node"], [v["req_id"]], demand):
                        pre.append({"victim": v["req_id"], "zone": z, "node": v["node"],
                                    "priority": v["priority"], "remaining_s": v["remaining_s"],
                                    "service_type": v["service_type"]})
            pre = sorted(pre, key=lambda v: (PRIORITY_RANK.get(v["priority"], 1),
                                             v["remaining_s"]))[:6]
        floor = self.catalog.degrade_floor(prof.get("service_type")) \
            if self.catalog.known(prof.get("service_type")) else 0.6
        feedback, src = [], None
        for attempt in range(MAX_REPLANS + 1):
            if attempt:
                self.counters["replans"] += 1
            plan_res = self._ask(PLAN_SYS, {"request": req, "observability": obs, "zones": zones,
                                            "allowed_zones": allowed, "preempt_candidates": pre,
                                            "degrade_floor": floor, "feedback": feedback},
                                 "plan", rid, rec)
            src = src or plan_res.source
            plan = plan_res.data
            if plan.get("action") == "reject" or plan.get("zone") not in self.view.zone_ids:
                if plan.get("action") == "reject":
                    return None, src
                feedback.append({"plan": plan, "error": "zone missing or unknown"})
                continue
            z = plan["zone"]
            env._rtt(z)
            table = [{"node": n, "cpu_free": round(r["cpu_free"], 2),
                      "mem_free": round(r["mem_free"], 2), "class": r["device_class"]}
                     for n, r in sorted(self.view.zone_table(z).items()) if r["healthy"]]
            lvl = plan.get("degrade_level") or 1.0
            try:
                lvl = float(lvl)
            except (TypeError, ValueError):
                lvl = 1.0
            vnode = next((v["node"] for v in pre if v["victim"] == plan.get("victim")), None)
            act = self._ask(ACT_SYS, {"plan": plan, "zone_nodes": table, "victim_node": vnode,
                                      "demand": {"cpu": round(demand["cpu"] * lvl, 3),
                                                 "mem": round(demand["mem"] * lvl, 3)}},
                            "act", rid, rec).data
            # --- simulate before execute ------------------------------------
            self.counters["validations"] += 1
            err = None
            if act.get("zone") not in allowed:
                err = f"zone {act.get('zone')} violates data locality / allowed_zones"
            else:
                victim = act.get("victim") if plan.get("action") == "preempt" else None
                if victim and victim not in {v["victim"] for v in pre}:
                    err = f"victim {victim} is not an allowed pre-emption candidate"
                else:
                    r = env.try_place(act.get("zone"), act.get("node"), prof["service_type"],
                                      evict=victim, degrade_level=act.get("degrade_level", lvl))
                    if not r.get("ok"):
                        err = r.get("reason", "infeasible")
            if err is None and env.decision is not None:
                env.decision.profile = dict(prof)
                env.decision.decision_source = "llm_fresh" if src == "fresh" else "llm_cached_disk"
                return env.decision, src
            self.counters["validation_failures"] += 1
            feedback.append({"plan": plan, "action": act, "validator": err})
        return None, src

    def handle(self, t, req, rec):
        rid = req.req_id
        try:
            ires = self._ask(slm_system_prompt(self.catalog), "INPUT: " + req.text, "intent",
                             rid, rec, component="translation")
        except LLMUnavailable:
            rec.translation_source = "none"
            return finish_decision(rec, Decision(node=None, path="unresolved", decision_source="none",
                                                 action="reject", demand={"cpu": 0.0, "mem": 0.0}),
                                   req.zone_id)
        rec.translation_source = "llm_fresh" if ires.source == "fresh" else "llm_cached_disk"
        prof, rkind = resolve_profile(ires.data, self.catalog, self.emb)
        rec.type_resolution = rkind
        demand = self.view.manifest(rid, prof["service_type"])
        env = ToolEnv(self.view, self.catalog, rid, req.zone_id, rec)
        try:
            dec, src = self._pipeline(t, rid, req.zone_id, prof, rec, env, demand)
        except LLMUnavailable:
            dec, src = None, None
        if dec is None:
            dec = Decision(node=None, path="unresolved",
                           decision_source={"fresh": "llm_fresh",
                                            "cached_disk": "llm_cached_disk"}.get(src, "none"),
                           action="reject", demand=demand, profile=prof)
        return finish_decision(rec, dec, req.zone_id)

    def replan(self, t, svc, scratch, reason=""):
        env = ToolEnv(self.view, self.catalog, svc.req_id, svc.origin_zone, scratch,
                      demand_override=svc.demand, priority_override=svc.profile.get("priority"))
        try:
            dec, _ = self._pipeline(t, svc.req_id, svc.origin_zone, svc.profile, scratch, env,
                                    dict(svc.demand))
        except LLMUnavailable:
            return None
        return dec

    def stats(self):
        return dict(self.counters)
