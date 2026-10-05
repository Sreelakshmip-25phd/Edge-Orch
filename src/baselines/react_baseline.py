"""ReAct (Yao et al., ICLR 2023) adapted to this environment.

Reimplements the mechanism, not the paper's benchmark: an LLM interleaves
a free-text thought with one tool call per step over the *same* cluster
the proposed system manages - read_digest(zone), read_running_services(zone),
try_place(zone, node, service_type[, evict, degrade_level]) - until a
try_place succeeds, it calls finish, or the step cap is hit. The step cap
is 7 (common.STEP_BUDGET; the top of the 5-7 range the brief specifies -
the quick pilot's 6 left too few steps after the reads).

No memory, no cache: every request reasons from scratch, including
translation (the model names the service_type inside try_place). Every
step is one LLM call, logged with kind="react_step".
"""
from llm_client import LLMUnavailable
from sim_engine import Decision

from .common import (STEP_BUDGET, TOOLS_DOC, ToolEnv, catalog_sizes, dumps, finish_decision,
                     request_block)

MAX_STEPS = STEP_BUDGET

REACT_SYS = """You are an orchestration agent placing a service request on an edge cluster.
Work step by step. At each step think briefly, then call ONE tool.
""" + TOOLS_DOC + """
Respond with ONLY a JSON object: {"thought": "<short reasoning>", "tool": "<tool name>", "args": {...}}"""


class ReActBaseline:
    name = "react"
    uses_llm = True

    def __init__(self, *, catalog, make_llm, resources, max_steps=MAX_STEPS, **_):
        self.catalog, self.resources = catalog, resources
        self.llm = make_llm("llm", "react")
        self.max_steps = max_steps
        self.view = None
        self.counters = {"steps": 0, "tool_errors": 0}

    def attach(self, view, latency):
        self.view = view

    def _loop(self, rid, origin, text, rec, env):
        history, first_src = [], None
        sizes = catalog_sizes(self.catalog, self.resources)
        base = {"request": request_block(self.catalog, sizes, text, origin, self.view),
                "catalog_sizes": sizes}
        for _ in range(self.max_steps):
            try:
                res = self.llm.ask(REACT_SYS, dumps({**base, "history": history}),
                                   kind="react_step", req_id=rid)
            except LLMUnavailable:
                break
            rec.reached_llm_stage = True
            first_src = first_src or res.source
            rec.add_latency("decision", res.sim_ms)
            self.counters["steps"] += 1
            a = res.data
            tool, args = a.get("tool"), a.get("args", {})
            if tool == "finish":
                break
            obs = env.run_tool(tool, args)
            if isinstance(obs, dict) and obs.get("error"):
                self.counters["tool_errors"] += 1
            history.append({"thought": str(a.get("thought", ""))[:300],
                            "action": {"tool": tool, "args": args}, "observation": obs})
            if env.decision is not None:
                env.decision.decision_source = "llm_fresh" if res.source == "fresh" \
                    else "llm_cached_disk"
                return env.decision, first_src
        return None, first_src

    def handle(self, t, req, rec):
        env = ToolEnv(self.view, self.catalog, req.req_id, req.zone_id, rec)
        dec, src = self._loop(req.req_id, req.zone_id, req.text, rec, env)
        rec.translation_source = {"fresh": "llm_fresh", "cached_disk": "llm_cached_disk"}.get(src, "none")
        if dec is None:
            dec = Decision(node=None, path="unresolved",
                           decision_source="llm_fresh" if src else "none", action="reject",
                           demand={"cpu": 0.0, "mem": 0.0})
        return finish_decision(rec, dec, req.zone_id)

    def replan(self, t, svc, scratch, reason=""):
        st = svc.profile.get("service_type") or "unknown"
        text = (f"Re-place an already-running {st} service (priority "
                f"{svc.profile.get('priority')}) that was displaced ({reason}); "
                f"use service_type {st}.")
        env = ToolEnv(self.view, self.catalog, svc.req_id, svc.origin_zone, scratch,
                      demand_override=svc.demand, priority_override=svc.profile.get("priority"))
        dec, _ = self._loop(svc.req_id, svc.origin_zone, text, scratch, env)
        if dec is not None:
            dec.profile = dict(svc.profile)
        return dec

    def stats(self):
        return dict(self.counters)
