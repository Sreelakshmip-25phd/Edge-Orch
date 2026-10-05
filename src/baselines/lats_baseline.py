"""LATS - Language Agent Tree Search (Zhou et al., ICML 2024) adapted to
this environment.

Same tool space as ReAct, wrapped in Monte Carlo Tree Search:
  selection     UCT over tree nodes (a node = a partial trajectory)
  expansion     sample n candidate next actions from the LLM
                (temperature 0.7, n independent calls: kind="lats_expand")
  evaluation    value = LAMBDA * LLM self-score (kind="lats_value", 0-10,
                one call per distinct child) + (1 - LAMBDA) * self-consistency
                (fraction of the n samples proposing that same action)
  simulation    the chosen child's action is executed for real: reads are
                side-effect free observations; a try_place is terminal
                (reward 1 on success - it commits - else 0)
  backprop      reward/value propagated to the root
  reflection    a failed try_place triggers kind="lats_reflect"; the
                reflection is added to every later prompt of this search
  re-expansion  when every child of a node has ended in failure, the node
                is expanded again - n new samples whose prompt now carries
                the reflections - instead of the search giving up (the
                quick pilot stopped as soon as the root's first children
                had all failed). Candidates identical to an action already
                tried at that node are dropped; a node whose re-expansion
                yields nothing new is closed.
Budget: at most K rollouts (default 20) and depth common.STEP_BUDGET (7);
the search stops at the first successful placement or when the root is
closed. No memory, no cache across requests.
This is expected - and logged - to be the most token-expensive baseline.
"""
import math

from llm_client import LLMUnavailable
from sim_engine import Decision

from .common import (STEP_BUDGET, TOOLS_DOC, ToolEnv, catalog_sizes, dumps, finish_decision,
                     request_block)

N_SAMPLES, MAX_ROLLOUTS, MAX_DEPTH = 3, 20, STEP_BUDGET
LAMBDA, C_UCT = 0.8, 1.0

EXPAND_SYS = """You are an orchestration agent placing a service request on an edge cluster,
exploring alternatives. Given the trajectory so far (and reflections on failed attempts),
propose the single best NEXT tool call.
""" + TOOLS_DOC + """
Respond with ONLY a JSON object: {"thought": "<short reasoning>", "tool": "<tool name>", "args": {...}}"""

VALUE_SYS = """You evaluate a candidate next step of an orchestration agent that is placing a
service request on an edge cluster. Score from 0 (useless or invalid) to 10 (will very likely
lead to a correct, feasible placement that respects data locality and priorities).
Respond with ONLY a JSON object: {"score": <0-10>, "why": "<short>"}"""

REFLECT_SYS = """An orchestration agent's placement attempt failed. In one or two sentences,
state what went wrong and what to do differently. Respond with ONLY a JSON object:
{"reflection": "<text>"}"""


class _Node:
    __slots__ = ("history", "action", "parent", "children", "visits", "value",
                 "terminal", "reward", "expanded", "simulated", "prior", "n_expansions")

    def __init__(self, history, action=None, parent=None):
        self.history, self.action, self.parent = history, action, parent
        self.children, self.visits, self.value = [], 0, 0.0
        self.terminal, self.reward, self.expanded = False, 0.0, False
        self.simulated, self.prior = False, 0.0
        self.n_expansions = 0

    def dead(self):
        """Every child tried and ended without a placement."""
        return bool(self.children) and all(c.simulated and c.terminal and c.reward <= 0.0
                                           for c in self.children)

    def uct(self, n_parent):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + C_UCT * math.sqrt(math.log(max(n_parent, 1)) / self.visits)


def _akey(a):
    return dumps({"tool": a.get("tool"), "args": a.get("args", {})})


class LATSBaseline:
    name = "lats"
    uses_llm = True

    def __init__(self, *, catalog, make_llm, resources, n_samples=N_SAMPLES,
                 max_rollouts=MAX_ROLLOUTS, **_):
        self.catalog, self.resources = catalog, resources
        self.llm = make_llm("llm", "lats")
        self.n, self.k = n_samples, max_rollouts
        self.view = None
        self.counters = {"rollouts": 0, "expansions": 0, "re_expansions": 0, "reflections": 0}

    def attach(self, view, latency):
        self.view = view

    def _ask(self, sys, body, kind, rid, rec, temperature=0.0, sample_idx=0):
        res = self.llm.ask(sys, dumps(body), kind=kind, req_id=rid,
                           temperature=temperature, sample_idx=sample_idx)
        rec.add_latency("decision", res.sim_ms)
        rec.reached_llm_stage = True
        return res

    def _expand(self, node, base, reflections, rid, rec):
        """Adds the distinct new candidates among n samples; returns
        (first response source, number of children added)."""
        samples, src = [], None
        first = node.n_expansions * self.n          # fresh sample indices on re-expansion
        node.n_expansions += 1
        for i in range(first, first + self.n):
            try:
                res = self._ask(EXPAND_SYS, {**base, "history": node.history,
                                             "reflections": reflections},
                                "lats_expand", rid, rec, temperature=0.7, sample_idx=i)
            except LLMUnavailable:
                continue
            src = src or res.source
            samples.append(res.data)
        node.expanded = True
        self.counters["expansions"] += 1
        if node.n_expansions > 1:
            self.counters["re_expansions"] += 1
        tried = {_akey(c.action) for c in node.children}
        groups = {}
        for a in samples:
            if _akey(a) not in tried:
                groups.setdefault(_akey(a), []).append(a)
        for acts in groups.values():
            a = acts[0]
            try:
                v = self._ask(VALUE_SYS, {**base, "history": node.history,
                                          "candidate": {"tool": a.get("tool"),
                                                        "args": a.get("args", {})}},
                              "lats_value", rid, rec).data
                score = float(v.get("score", 0)) / 10.0
            except (LLMUnavailable, TypeError, ValueError, AttributeError):
                score = 0.0
            child = _Node(node.history, action=a, parent=node)
            child.prior = LAMBDA * max(0.0, min(score, 1.0)) + \
                (1 - LAMBDA) * len(acts) / len(samples)
            node.children.append(child)
        return src, len(groups)

    def _search(self, rid, origin, text, rec, env):
        sizes = catalog_sizes(self.catalog, self.resources)
        base = {"request": request_block(self.catalog, sizes, text, origin, self.view),
                "catalog_sizes": sizes}
        root, reflections, first_src = _Node([]), [], None
        root.simulated = True
        for _ in range(self.k):
            self.counters["rollouts"] += 1
            node = root
            while True:
                if not node.simulated:                        # new leaf: simulate it
                    reward = self._simulate(node, env, reflections, base, rid, rec)
                    break
                if len(node.history) >= MAX_DEPTH and not node.terminal:
                    node.terminal, node.reward = True, 0.0    # step budget used up: a failure
                if node.terminal:
                    reward = node.reward
                    break
                if not node.expanded or node.dead():          # (re-)expand, then simulate best new child
                    src, added = self._expand(node, base, reflections, rid, rec)
                    first_src = first_src or src
                    if not added:                             # nothing new to try here: close it
                        node.terminal, node.reward, reward = True, 0.0, 0.0
                        break
                    continue
                pending = [c for c in node.children if not c.simulated]
                live = [c for c in node.children if not (c.terminal and c.reward <= 0.0)]
                node = max(pending, key=lambda c: c.prior) if pending else \
                    max(live, key=lambda c: c.uct(node.visits))
            if env.decision is not None:
                env.decision.decision_source = "llm_fresh" if first_src == "fresh" \
                    else "llm_cached_disk"
                return env.decision, first_src
            self._backprop(node, reward)
            if root.terminal:
                break
        return None, first_src

    def _simulate(self, child, env, reflections, base, rid, rec):
        child.simulated = True
        a = child.action or {}
        tool, args = a.get("tool"), a.get("args", {})
        if tool == "finish" or tool is None:
            child.terminal, child.reward = True, 0.0
            return 0.0
        obs = env.run_tool(tool, args)
        child.history = child.history + [{"thought": str(a.get("thought", ""))[:300],
                                          "action": {"tool": tool, "args": args},
                                          "observation": obs}]
        if tool == "try_place":
            child.terminal = True
            if isinstance(obs, dict) and obs.get("ok"):
                child.reward = 1.0
                return 1.0
            child.reward = 0.0
            try:
                r = self._ask(REFLECT_SYS, {**base, "failed_trajectory": child.history},
                              "lats_reflect", rid, rec).data
                reflections.append(str(r.get("reflection", ""))[:300])
                self.counters["reflections"] += 1
            except LLMUnavailable:
                pass
            return 0.0
        return child.prior                     # non-terminal read: value estimate

    @staticmethod
    def _backprop(node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def handle(self, t, req, rec):
        env = ToolEnv(self.view, self.catalog, req.req_id, req.zone_id, rec)
        dec, src = self._search(req.req_id, req.zone_id, req.text, rec, env)
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
        dec, _ = self._search(svc.req_id, svc.origin_zone, text, scratch, env)
        if dec is not None:
            dec.profile = dict(svc.profile)
        return dec

    def stats(self):
        return dict(self.counters)
