import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "local_llm"))

from helpers import tiny_topology  # noqa: E402
from latency_model import LatencyModel  # noqa: E402


def test_calibration_file_round_trips_into_latency_model(monkeypatch, tmp_path):
    # PLACEHOLDER numbers from the mock provider - written only to a temp
    # file, never to src/latency_distributions.json
    import calibrate_latency as CL
    monkeypatch.setattr(CL, "provider_for", lambda label, url: {
        "name": "MOCK", "mock": True, "roles": None, "models": ["mock"]})
    monkeypatch.setattr(CL, "served_model", lambda url: "mock.gguf")
    out = tmp_path / "lat.json"
    CL.main(["--target", "s@http://x", "--target", "l@http://y", "--n", "6", "--repeats", "1",
             "--out", str(out)])
    cal = json.load(open(out))
    assert set(cal["models"]) == {"s", "l"}
    assert cal["models"]["s"]["kinds"]["translate"]["n"] > 0
    lm = LatencyModel.build(tiny_topology(), "real", str(out), None, 0, "s", "l")
    assert lm.mode == "calibrated"
    v = lm.llm_call("llm", "decide", wall_ms=999999.0)
    assert v in cal["models"]["l"]["kinds"]["decide"]["samples_ms"]
    assert lm.llm_call("llm", "lats_value", 1.0) in cal["models"]["l"]["kinds"]["decide"]["samples_ms"]


def test_missing_calibration_means_live_not_invented(tmp_path):
    lm = LatencyModel.build(tiny_topology(), "real", str(tmp_path / "absent.json"), None, 0, "s", "l")
    assert lm.mode == "live"
    assert lm.llm_call("slm", "translate", wall_ms=412.5) == 412.5


def test_cross_zone_and_escalation_components_are_topology_values():
    lm = LatencyModel(tiny_topology(), "placeholder")
    assert lm.cross_zone("z0", "z1") == 6.0 and lm.cross_zone("z0", "z0") == 0.0
    assert lm.escalation("z1") == 30.0
    assert lm.deployment("raspberry_pi_4") > 0
