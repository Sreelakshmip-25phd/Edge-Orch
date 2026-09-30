import numpy as np

from helpers import HashEmbedder, ScriptedLLM, catalog
from scenario import ground_truth_profile
from telemetry import Telemetry
from zone_agent import IntentLibrary, ZoneAgent, threshold_sweep


def _agent(slm, threshold=0.8, shadow_rate=0.0, seed_entries=()):
    return ZoneAgent("z0", catalog(), HashEmbedder(), slm, threshold=threshold,
                     shadow_rate=shadow_rate, rng=np.random.default_rng(0),
                     seed_entries=seed_entries)


def _rec(tel, rid, text):
    return tel.new_request(req_id=rid, device_id="d", origin_zone="z0", send_time=0.0,
                           service_type_true="video_analytics", intent_text=text)


VA = ground_truth_profile("video_analytics")


def test_cache_starts_empty():
    za = _agent(ScriptedLLM("slm"))
    assert len(za.lib) == 0
    assert za.stats()["cache_entries"] == 0


def test_heldout_paraphrase_below_threshold_misses_and_calls_slm():
    slm = ScriptedLLM("slm", script={"translate": [dict(VA), dict(VA)]})
    za = _agent(slm, threshold=0.8)
    tel = Telemetry()
    p1 = za.translate(0.0, "r1", "analyse stadium camera feeds in real time", _rec(tel, "r1", "a"))
    assert p1["service_type"] == "video_analytics"
    assert tel.requests["r1"].translation_source == "slm_fresh"
    # held-out paraphrase: little word overlap -> similarity below threshold -> miss
    r2 = _rec(tel, "r2", "b")
    za.translate(1.0, "r2", "detect incidents on surveillance streams with low latency", r2)
    assert r2.translation_source == "slm_fresh"
    assert r2.cache_similarity < 0.8
    assert [c["kind"] for c in slm.calls] == ["translate", "translate"]
    # an exact repeat is an earned hit (from the first translation of real traffic)
    r3 = _rec(tel, "r3", "c")
    za.translate(2.0, "r3", "analyse stadium camera feeds in real time", r3)
    assert r3.translation_source == "cache" and len(slm.calls) == 2


def test_shadow_check_flags_and_repairs_wrong_cached_translation():
    wrong = ground_truth_profile("iot_aggregator")          # a poisoned cache entry
    slm = ScriptedLLM("slm", script={"shadow_check": [dict(VA)]})
    text = "analyse stadium camera feeds in real time"
    za = _agent(slm, threshold=0.8, shadow_rate=1.0, seed_entries=[(text, wrong)])
    near = text + " please"                                   # a near-match, not exact
    tel = Telemetry()
    r = _rec(tel, "r1", near)
    used = za.translate(0.0, "r1", near, r)
    assert used["service_type"] == "iot_aggregator"          # the request used the cache...
    assert r.shadow_checked and r.shadow_agree is False      # ...the audit disagreed
    assert za.counters["shadow_disagree"] == 1
    r2 = _rec(tel, "r2", near)
    za.shadow_rate = 0.0
    assert za.translate(1.0, "r2", near, r2)["service_type"] == "video_analytics"   # repaired


def test_exact_repeats_are_not_audited_and_trust_decays_audits():
    slm = ScriptedLLM("slm", default=lambda k, s, u: dict(VA))
    text = "analyse stadium camera feeds in real time"
    za = _agent(slm, threshold=0.8, shadow_rate=1.0, seed_entries=[(text, VA)])
    tel = Telemetry()
    za.translate(0.0, "r0", text, _rec(tel, "r0", text))
    assert za.counters["shadow_checks"] == 0                  # exact repeat: nothing to audit
    for i, extra in enumerate(("now", "today", "asap"), start=1):   # near-matches build trust
        near = f"{text} {extra}"
        za.translate(float(i), f"r{i}", near, _rec(tel, f"r{i}", near))
    assert za.lib.trust[0] >= 2


def test_lru_cap_evicts_least_recently_used():
    lib = IntentLibrary(HashEmbedder(), threshold=0.99, cap=2)
    lib.add("alpha beta", VA)
    lib.add("gamma delta", VA)
    assert lib.lookup("alpha beta")[0] is not None           # touch alpha
    lib.add("epsilon zeta", VA)                              # evicts gamma (LRU)
    assert lib.lookup("gamma delta")[0] is None
    assert lib.lookup("alpha beta")[0] is not None and lib.evictions == 1


def test_open_vocab_label_resolves_or_stays_novel():
    slm = ScriptedLLM("slm", script={"translate": [
        {"service_type": "crowd monitoring", "latency_class": "realtime",
         "data_locality": "zone_local", "priority": "critical"}]})
    za = _agent(slm)
    tel = Telemetry()
    r = _rec(tel, "r1", "x")
    p = za.translate(0.0, "r1", "count people at the gates", r)
    # crowd_safety isn't registered at t=0: must not be forced into a known type
    assert r.type_resolution in ("novel", "nearest")
    if r.type_resolution == "novel":
        assert p["service_type"].startswith("novel:")


def test_threshold_sweep_counts_false_hits_on_unseen_types():
    emb = HashEmbedder()
    train = {"video_analytics": ["run video analytics on the {place} cameras"],
             "crowd_safety": []}
    held = {"video_analytics": ["run video analytics on the {place} cameras please"],
            "crowd_safety": ["run crowd analytics on the {place} cameras"]}
    sw = threshold_sweep(emb, train, held, {}, ["stadium"], thresholds=[0.3, 0.99])
    low, high = sw["rows"]
    assert low["false_hit_rate_unseen"] == 1.0 and high["false_hit_rate_unseen"] == 0.0
