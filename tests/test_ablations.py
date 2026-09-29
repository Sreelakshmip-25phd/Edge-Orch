import pytest

import ablations
from helpers import HashEmbedder, ScriptedLLM, catalog, mock_env, tiny_topology

import numpy as np

FULL_FLAGS = ("use_cache", "use_memory", "use_digest", "use_llm", "use_preempt_degrade",
              "use_zone_tier", "use_cross_zone", "use_procedural")


def _make(name):
    return ablations.make(name, catalog=catalog(), embedder=HashEmbedder(), topo=tiny_topology(),
                          make_llm=lambda role, agent: ScriptedLLM(role), threshold=0.8,
                          rng=np.random.default_rng(0))


@pytest.mark.parametrize("name", sorted(ablations.DISABLES))
def test_each_ablation_disables_exactly_what_it_claims(name):
    o = _make(name)
    off = {f for f in FULL_FLAGS if not o.flags[f]}
    assert off == set(ablations.DISABLES[name]), (name, off)
    ga = o.global_
    assert ga.use_memory == o.flags["use_memory"]
    assert ga.use_digest == o.flags["use_digest"]
    assert ga.use_pd == o.flags["use_preempt_degrade"]
    for za in o._agents():
        assert za.use_cache == o.flags["use_cache"]
    assert bool(o.zones) == o.flags["use_zone_tier"]


def test_no_digest_keeps_memory_and_no_memory_keeps_digest():
    run, _, _ = mock_env(n_requests=160, lifetime_scale=80.0)
    tel_nd, o_nd = run("no_digest")
    assert o_nd.global_.digests == {}                        # digest really off...
    assert o_nd.global_.mem.size() > 0                       # ...memory really on
    assert not any(r.path == "digest" and r.decision_source == "digest"
                   for r in tel_nd.requests.values())
    tel_nm, o_nm = run("no_memory")
    assert o_nm.global_.mem.size() == 0                      # memory really off...
    assert o_nm.global_.digests                              # ...digest really on
    assert not any(r.path in ("episodic", "procedural") for r in tel_nm.requests.values())
    tel_nc, o_nc = run("no_intent_cache")
    assert not any(r.translation_source == "cache" for r in tel_nc.requests.values())
    tel_nz, o_nz = run("no_zone_tier")
    assert not any(r.path == "local" for r in tel_nz.requests.values())
    tel_ncz, _ = run("no_cross_zone")
    assert not any(r.cross_zone for r in tel_ncz.requests.values())
    tel_npd, _ = run("no_preempt_degrade")
    assert not any(r.action in ("preempt", "degrade") for r in tel_npd.requests.values())
