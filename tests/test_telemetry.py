import pytest

from telemetry import IncompleteRecordError, Telemetry


def _req(tel, rid="r1"):
    return tel.new_request(req_id=rid, device_id="d1", origin_zone="z0",
                           send_time=1.0, service_type_true="video_analytics",
                           intent_text="analyse the stadium cameras")


def _complete(tel, rid="r1"):
    rec = tel.requests[rid]
    rec.translation_source = "cache"
    rec.path = "local"
    rec.decision_source = "local"
    rec.add_latency("transport_in", 1.0)
    tel.transition(rid, 1.0, "PLACED", "arrival")
    tel.transition(rid, 50.0, "COMPLETED", "lifetime_end")
    tel.finalize(rid, "completed")
    return rec


def test_round_trip(tmp_path):
    tel = Telemetry({"system": "full", "seed": 0})
    _req(tel)
    _complete(tel)
    tel.log_llm_call(req_id="r1", agent="zone:z0", role="slm", kind="translate",
                     source="fresh", tokens_in=100, tokens_out=20, wall_ms=300)
    tel.log_event(5.0, "node_failure", node="n1")
    tel.snapshot(5.0, cache_entries=3)
    p = tmp_path / "t.jsonl.gz"
    tel.dump(str(p))
    back = Telemetry.load(str(p))
    assert back.meta == {"system": "full", "seed": 0}
    r = back.requests["r1"]
    assert r.state == "COMPLETED" and r.finalized
    assert r.tokens["slm"] == {"in": 100, "out": 20}
    assert r.calls["slm_fresh"] == 1
    assert back.llm_calls[0].source == "fresh"
    assert back.events[0].data == {"node": "n1"}
    assert back.snapshots[0].data["cache_entries"] == 3
    assert back.snapshots[0].data["slm_fresh"] == 1
    back.assert_complete()


def test_every_transition_captured():
    tel = Telemetry()
    _req(tel)
    rec = tel.requests["r1"]
    tel.transition("r1", 1.0, "PLACED", "arrival")
    tel.transition("r1", 9.0, "PLACED", "preempted")
    tel.transition("r1", 9.0, "PLACED", "migrated")
    tel.transition("r1", 20.0, "DISPLACED", "node_failure")
    states = [s for _, s, _ in rec.transitions]
    assert states == ["REQUESTED", "PLACED", "PLACED", "PLACED", "DISPLACED"]
    assert rec.n_interruptions == 2 and rec.n_migrations == 1
    assert rec.accepted


def test_fresh_vs_cached_disk_never_conflated():
    tel = Telemetry()
    _req(tel)
    tel.log_llm_call(req_id="r1", agent="global", role="llm", kind="decide",
                     source="fresh", tokens_in=500, tokens_out=40)
    tel.log_llm_call(req_id="r1", agent="global", role="llm", kind="decide",
                     source="cached_disk", tokens_in=500, tokens_out=40)
    rec = tel.requests["r1"]
    assert rec.calls["llm_fresh"] == 1 and rec.calls["llm_cached_disk"] == 1
    with pytest.raises(ValueError):
        tel.log_llm_call(req_id="r1", agent="g", role="llm", kind="decide",
                         source="cache")          # ambiguous label rejected


def test_partial_record_blocks_metrics():
    tel = Telemetry()
    _req(tel, "r1")
    _req(tel, "r2")
    _complete(tel, "r1")
    tel.requests["r2"].translation_source = "cache"   # half-filled
    with pytest.raises(IncompleteRecordError):
        tel.assert_complete()


def test_bad_vocabulary_rejected():
    tel = Telemetry()
    _req(tel)
    rec = tel.requests["r1"]
    rec.translation_source = "cache"
    rec.path = "teleport"
    rec.decision_source = "local"
    rec.add_latency("transport_in", 1.0)
    tel.transition("r1", 1.0, "REJECTED", "arrival")
    with pytest.raises(ValueError):
        tel.finalize("r1", "rejected")
    with pytest.raises(ValueError):
        rec.add_latency("python_wallclock", 3.0)
