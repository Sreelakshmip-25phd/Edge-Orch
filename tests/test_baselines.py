import json

import pytest

from helpers import mock_env
from orchestrator import ScratchRecord
from sim_engine import Decision

BASELINES = ["greedy_oracle", "rule_based", "react", "lats", "agentedge", "core"]


@pytest.fixture(scope="module")
def env():
    return mock_env(n_requests=120, lifetime_scale=60.0)


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_runs_and_is_nontrivial(env, name):
    run, wl, _ = env
    tel, orch = run(name)
    R = list(tel.requests.values())
    assert len(R) == len(wl["requests"])
    acc = sum(r.accepted for r in R)
    assert 0 < acc, f"{name} accepted nothing"
    if getattr(orch, "uses_llm", False):
        assert tel.llm_calls, f"{name} never called a model"


def test_llm_call_cost_ordering(env):
    run, wl, _ = env
    per_req = {}
    for name in ("react", "lats", "agentedge"):
        tel, _ = run(name)
        per_req[name] = len(tel.llm_calls) / len(tel.requests)
    assert per_req["agentedge"] >= 4.0          # four roles, every request
    assert per_req["lats"] > per_req["react"]   # tree search costs more than a single chain


def test_lats_re_expands_with_reflections_after_all_children_fail(monkeypatch):
    import baselines.lats_baseline as L
    from llm_client import LLMResult
    monkeypatch.setattr(L, "request_block", lambda *a: {})
    monkeypatch.setattr(L, "catalog_sizes", lambda *a: {})

    class FakeLLM:
        def __init__(self):
            self.expand_prompts = []

        def ask(self, system, user, *, kind, req_id=None, temperature=0.0, sample_idx=0, **_):
            if kind == "lats_expand":
                self.expand_prompts.append(json.loads(user))
                node = f"bad{sample_idx}" if sample_idx < 3 else "good"
                out = {"tool": "try_place", "args": {"zone": "z0", "node": node}}
            elif kind == "lats_value":
                out = {"score": 5}
            else:
                out = {"reflection": "that node was full; try another"}
            return LLMResult(data=out, source="fresh", tokens_in=1, tokens_out=1,
                             wall_ms=1.0, sim_ms=1.0)

    class FakeEnv:
        decision = None

        def run_tool(self, tool, args):
            if args.get("node") == "good":
                self.decision = Decision(node="good", path="local", decision_source="llm_fresh")
                return {"ok": True}
            return {"ok": False, "reason": "full"}

    llm = FakeLLM()
    lats = L.LATSBaseline(catalog=None, make_llm=lambda role, agent: llm, resources=None)
    dec, _ = lats._search("r1", "z0", "text", ScratchRecord("r1"), FakeEnv())
    assert dec is not None and dec.node == "good"
    assert lats.counters["re_expansions"] == 1
    assert lats.counters["reflections"] == 3
    assert len(llm.expand_prompts[-1]["reflections"]) == 3     # re-expansion saw the reflections
