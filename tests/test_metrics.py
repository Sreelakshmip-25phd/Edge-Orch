import pytest

import metrics
from telemetry import IncompleteRecordError, Telemetry


def _r(tel, rid, t, escalated, outcome):
    rec = tel.new_request(req_id=rid, device_id="d", origin_zone="z0", send_time=t,
                          service_type_true="traffic_monitor", intent_text="x", phase="A",
                          priority_true="normal", locality_true="any")
    rec.translation_source = "cache"
    rec.decision_source = "local"
    rec.escalated = escalated
    rec.service_type_correct = True
    rec.add_latency("transport_in", 1.0)
    if outcome == "rejected":
        rec.path = "unresolved"
        tel.transition(rid, t, "REJECTED", "arrival")
        tel.finalize(rid, "rejected")
        return rec
    rec.path = "digest" if escalated else "local"
    rec.zone_final = rec.origin_zone
    tel.transition(rid, t, "PLACED", "arrival")
    if outcome == "completed":
        tel.transition(rid, t + 10, "COMPLETED", "lifetime_end")
        tel.finalize(rid, "completed")
    else:
        tel.transition(rid, t + 5, "DISPLACED", "node_failure")
        tel.finalize(rid, "node_loss")
    return rec


def test_acceptance_completion_escalation_success_are_distinct():
    tel = Telemetry({"horizon_s": 100.0})
    _r(tel, "r1", 1, False, "completed")
    _r(tel, "r2", 2, True, "node_loss")
    _r(tel, "r3", 3, True, "rejected")
    _r(tel, "r4", 4, False, "rejected")
    _r(tel, "r5", 5, True, "completed")
    S = metrics.compute(tel, n_bins=4)["scalars"]
    assert S["acceptance_rate"] == pytest.approx(3 / 5)
    assert S["completion_rate"] == pytest.approx(2 / 5)
    assert S["escalation_success"] == pytest.approx(2 / 3)
    assert len({S["acceptance_rate"], S["completion_rate"], S["escalation_success"]}) == 3
    assert S["fail_node_loss"] == 1


def test_metrics_refuse_partial_records():
    tel = Telemetry({"horizon_s": 100.0})
    _r(tel, "r1", 1, False, "completed")
    tel.new_request(req_id="r2", device_id="d", origin_zone="z0", send_time=2.0,
                    service_type_true="traffic_monitor", intent_text="x")
    with pytest.raises(IncompleteRecordError):
        metrics.compute(tel)


def test_fresh_and_cached_disk_calls_reported_separately():
    tel = Telemetry({"horizon_s": 100.0})
    _r(tel, "r1", 1, False, "completed")
    tel.log_llm_call(req_id="r1", agent="z", role="slm", kind="translate", source="fresh",
                     tokens_in=100, tokens_out=10)
    tel.log_llm_call(req_id="r1", agent="z", role="slm", kind="translate", source="cached_disk",
                     tokens_in=100, tokens_out=10)
    S = metrics.compute(tel, n_bins=4)["scalars"]
    assert S["calls_slm_fresh"] == 1 and S["calls_slm_cached_disk"] == 1
    assert S["fresh_calls_per_req"] == 1.0 and S["invocations_per_req"] == 2.0
