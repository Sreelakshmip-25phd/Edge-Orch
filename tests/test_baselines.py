import pytest

from helpers import mock_env

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


def test_offline_bound_is_an_upper_bound(env):
    from baselines.optimal_solver import window_bound
    from workload import lifetime_of
    run, wl, topo = env
    tel, _ = run("greedy_oracle")
    reqs = wl["requests"][:60]
    b = window_bound(topo, reqs, [lifetime_of(r, wl) for r in reqs], time_limit_s=20)
    done = sum(tel.requests[r["req_id"]].end_cause == "completed" for r in reqs)
    assert b["kind"] in ("ilp", "lp_relaxation")
    assert b["bound"] + 1e-6 >= done
