"""Phase 5 - latency built from real components, never raw Python
wall-clock of the whole decision (old-repo flaw #7).

Per request, total latency = sum of:
  transport_in  device -> zone         topology intra_zone_ms (RIPE Atlas metro band)
  translation   cache hit              measured sentence-embedding time on this host
                                       (+ the measured cosine lookup)
                SLM call               sampled from the measured SLM distribution
  escalation    zone -> global -> zone  topology zone_to_global_ms (RIPE zone_to_cloud band),
                                       charged once per escalation
  decision      local / memory / digest measured compute time of that deterministic
                                       operation only (perf_counter around the solver /
                                       memory lookup / digest scan - not the whole call chain)
                LLM                    sampled from the measured LLM distribution, per call
                                       (a retry is another sample)
  cross_zone    origin <-> target zone  topology inter_zone_rtt_ms, whenever the final zone
                                       differs (and per cross-zone probe) - never silently 0
  deployment    container start         per device class (device_specs.py; measured if
                                       scripts/generate_device_calibration.py was run)

Where the LLM/SLM numbers come from, in priority order (recorded in every
run's metadata as `latency_source`):
  "calibrated"  src/latency_distributions.json, written by
                scripts/calibrate_latency.py on the GPU machine: empirical
                samples per (model, prompt kind), bootstrapped here.
  "live"        that file (or the active model's entry) is absent: each call
                is charged its own measured wall time (fresh network calls),
                or the wall time recorded when that response was first
                produced (on-disk cache hits). Real, but host-dependent.
  "placeholder" smoke runs with the mock LLM only. Numbers are NOT
                measurements; results are tagged so they can't be mistaken
                for real ones.
"""
import json
import math
import os
import time

import numpy as np

from device_specs import load_device_calibration


def measure(fn, *a, **kw):
    """Run fn, return (result, elapsed_ms) - for the deterministic
    decision components only."""
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000.0


def load_calibration(path):
    if not path or not os.path.exists(path):
        return None
    cal = json.load(open(path))
    if not cal.get("models"):
        return None
    return cal


class LatencyModel:
    def __init__(self, topo, mode, calibration=None, device_cal_path=None,
                 seed=0, slm_label=None, llm_label=None):
        if mode not in ("calibrated", "live", "placeholder"):
            raise ValueError(mode)
        self.topo, self.mode = topo, mode
        self.cal = calibration
        self.rng = np.random.default_rng(10_000 + seed)   # never shares a stream with decisions
        self.labels = {"slm": slm_label, "llm": llm_label}
        self.device = load_device_calibration(device_cal_path)
        self._embed = [0.0]

    @classmethod
    def build(cls, topo, llm_mode, calibration_path, device_cal_path, seed,
              slm_label, llm_label):
        if llm_mode == "mock":
            return cls(topo, "placeholder", None, device_cal_path, seed,
                       slm_label, llm_label)
        cal = load_calibration(calibration_path)
        have = cal is not None and all(
            lab in cal["models"] for lab in (slm_label, llm_label))
        return cls(topo, "calibrated" if have else "live", cal if have else None,
                   device_cal_path, seed, slm_label, llm_label)

    def describe(self):
        return {"latency_source": self.mode,
                "calibration_meta": (self.cal or {}).get("meta"),
                "deployment_is_assumption": {k: v["deploy_is_assumption"]
                                             for k, v in self.device.items()}}

    # --- transport ----------------------------------------------------------
    def transport_in(self, device_transport_ms):
        return float(device_transport_ms)

    def escalation(self, zone):
        z = next(z for z in self.topo["zones"] if z["zone_id"] == zone)
        return float(z["zone_to_global_ms"])

    def cross_zone(self, a, b):
        from scenario import rtt
        return rtt(self.topo, a, b)

    # --- translation ----------------------------------------------------------
    def set_embedding_samples(self, samples_ms):
        self._embed = list(samples_ms) or [0.0]

    def embed_ms(self):
        return float(self._embed[int(self.rng.integers(len(self._embed)))])

    # --- LLM / SLM ---------------------------------------------------------------
    def llm_call(self, role, kind, wall_ms):
        """Simulated latency of one SLM/LLM call of prompt `kind`."""
        if self.mode in ("live", "placeholder"):
            return float(wall_ms)
        entry = self.cal["models"][self.labels[role]]["kinds"]
        samples = entry.get(kind) or entry.get(_KIND_FALLBACK.get(kind, "")) \
            or next(iter(entry.values()))
        s = samples["samples_ms"]
        return float(s[int(self.rng.integers(len(s)))])

    # --- deployment ------------------------------------------------------------------
    def deployment(self, device_class):
        d = self.device[device_class]
        return float(d["deploy_ms_median"] * math.exp(
            self.rng.normal(0.0, d["deploy_ms_sigma"])))


# Prompt kinds that share a latency distribution when the calibration file
# only measured the canonical ones (translate for SLM-sized prompts, decide
# for LLM-sized structured decisions).
_KIND_FALLBACK = {
    "shadow_check": "translate", "intent": "translate", "zone_choice": "translate",
    "rule_author": "decide", "react_step": "decide", "lats_expand": "decide",
    "lats_value": "decide", "lats_reflect": "decide", "observe": "decide",
    "plan": "decide", "act": "decide", "critic": "decide", "translate_llm": "decide",
}


def measure_embedding_samples(embedder, texts, n=60):
    """Measured per-sentence encoding latency on this host (single
    sentence, no batching - that's what a cache lookup does)."""
    out = []
    for t in (texts * (n // max(len(texts), 1) + 1))[:n]:
        t0 = time.perf_counter()
        embedder.encode_uncached([t])
        out.append((time.perf_counter() - t0) * 1000.0)
    return out[5:] if len(out) > 10 else out     # drop warm-up
