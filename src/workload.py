"""Phase 2b / 6 - non-stationary workload generator.

"Dynamic" (the orchestrator reacts to current state) and "non-stationary"
(the *generating distribution* changes over time) are different
properties. This module is only about the second. Everything that drifts
does so gradually - there are no hard phase cuts:

  * arrival intensity   follows the Milan trace's diurnal curve (10-min
                        slots), blended with the real surge day in an
                        evening window via a raised-cosine ramp;
  * spatial mix         per-slot zone distribution from the trace (moves
                        through the day by itself), surge zone taken over
                        from the real surge day during the surge window;
  * service-type mix    each type's weight follows a logistic ramp centred
                        on its introduction hour, plus a gentle diurnal
                        modulation of the core types; a genuinely unseen
                        type (crowd_safety) appears mid-afternoon and is
                        registered in the catalog 30 min after its first
                        traffic; another (ev_charging) appears in the
                        evening and is never registered;
  * phrasing pool       the held-out template pool, the place names and
                        the OOD sentence pool all widen continuously;
  * devices             move between zones (scenario.MobilityModel) and
                        have phrasing habits, so a device that moves
                        carries phrasings the new zone's cache hasn't seen;
  * failures            several per run, sampled from the Google trace
                        renewal model (data_foundation.FailureModel).

A/B/C phase labels are kept only as reporting windows (A: 00-08h,
B: 08-16h, C: 16-24h of the simulated day).

The horizon is always mapped onto one full activity day: slot_s =
horizon_s / n_slots, so a short smoke horizon replays a compressed day.
"""
import gzip
import json
import math
import os

import numpy as np

from scenario import (PLACES_INITIAL, PLACES_LATER, SERVICE_TYPES,
                      MobilityModel, compress_trajectory, ground_truth_profile,
                      zone_distance_matrix)

PHASES = (("A", 0.0, 8.0), ("B", 8.0, 16.0), ("C", 16.0, 24.0))
SURGE_HALF_WIDTH_SLOTS = 12          # +-2h around the surge peak
HABIT_P = 0.6                        # device reuses one of its phrasings


def phase_of(hour):
    for name, lo, hi in PHASES:
        if lo <= hour < hi:
            return name
    return PHASES[-1][0]


def _logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def type_weights(hour):
    """Unnormalised weight of every service type at hour-of-day `hour`."""
    w = {}
    for st, s in SERVICE_TYPES.items():
        base = s["base_weight"]
        if s["intro_h"] is None:
            # gentle diurnal drift of the core mix: camera/AR types busier in
            # the evening, sensor batch jobs busier at night
            phase = 2 * math.pi * (hour - 14.0) / 24.0
            mod = 1.0 + (0.3 * math.sin(phase) if s["accel_pref"]
                         else (-0.25 * math.sin(phase)
                               if s["latency_ms"] >= 500 else 0.0))
            w[st] = base * mod
        else:
            ramp = max(s["ramp_h"], 1e-3)
            if hour < s["intro_h"] - ramp:
                w[st] = 0.0
            else:
                w[st] = base * _logistic(4.4 * (hour - s["intro_h"]) / ramp)
    return w


def available_count(n, hour, start_frac=0.35):
    """How many of an ordered pool are in circulation at `hour` - grows
    linearly from start_frac to 1 over the day."""
    f = start_frac + (1.0 - start_frac) * min(max(hour / 24.0, 0.0), 1.0)
    return max(1, int(math.ceil(f * n)))


def places_at(hour):
    extra = int(round(len(PLACES_LATER) * min(max((hour - 8.0) / 12.0, 0.0), 1.0)))
    return PLACES_INITIAL + PLACES_LATER[:extra]


class PhrasingSampler:
    def __init__(self, held_templates, held_fixed, rng):
        self.tmpl, self.fixed, self.rng = held_templates, held_fixed, rng
        self.habits = {}                        # (device, type) -> [text]

    def fresh(self, st, hour):
        rng = self.rng
        tm = self.tmpl.get(st, [])
        fx = self.fixed.get(st, [])
        n_fx = 0 if hour < 12.0 or not fx else available_count(len(fx), (hour - 12.0) * 2.0, 0.2)
        n_tm = available_count(len(tm), hour) if tm else 0
        k = int(rng.integers(n_tm + n_fx))
        if k < n_tm:
            pl = places_at(hour)
            return tm[k].format(place=pl[int(rng.integers(len(pl)))])
        return fx[k - n_tm]

    def sample(self, device_id, st, hour):
        key = (device_id, st)
        h = self.habits.get(key)
        if h and self.rng.uniform() < HABIT_P:
            return h[int(self.rng.integers(len(h)))]
        txt = self.fresh(st, hour)
        h = self.habits.setdefault(key, [])
        if len(h) < 2:
            h.append(txt)
        else:
            h[int(self.rng.integers(2))] = txt
        return txt


def _surge_alpha(n_slots, center):
    a = np.zeros(n_slots)
    for s in range(n_slots):
        d = abs(s - center)
        if d <= SURGE_HALF_WIDTH_SLOTS:
            a[s] = 0.5 * (1 + math.cos(math.pi * d / SURGE_HALF_WIDTH_SLOTS))
    return a


def generate(topo, activity, resources, failure_model, pools, *, seed,
             n_requests, n_devices, horizon_s, min_failures=3):
    rng = np.random.default_rng(seed)
    _, held_t, held_f = pools
    base, surge = np.asarray(activity["base"], float), np.asarray(activity["surge"], float)
    n_slots, n_z = base.shape
    slot_s = horizon_s / n_slots
    sz = int(activity["surge_zone"])

    # surge window: where the real surge day exceeds the base day most in
    # the surge zone, restricted to the evening reporting window (C)
    c_lo = int(n_slots * 16 / 24)
    excess = surge[c_lo:, sz] - base[c_lo:, sz]
    center = c_lo + int(np.argmax(excess))
    alpha = _surge_alpha(n_slots, center)

    base_tot, surge_tot = base.sum(1), surge.sum(1)
    rate = base_tot * (1 - alpha) + surge_tot * alpha
    zone_p = []
    for s in range(n_slots):
        pb = base[s] / max(base_tot[s], 1e-12)
        ps = surge[s] / max(surge_tot[s], 1e-12)
        p = (1 - alpha[s]) * pb + alpha[s] * ps
        zone_p.append(p / p.sum())
    zone_p = np.array(zone_p)

    # --- devices ---------------------------------------------------------
    mob = MobilityModel(zone_p * rate[:, None], zone_distance_matrix(topo),
                        slot_s=slot_s)
    home, traj = mob.trajectories(n_devices, rng)
    devices = [{"device_id": f"d{i:05d}", "home_zone": f"z{int(home[i])}",
                "trajectory": compress_trajectory(traj[i])}
               for i in range(n_devices)]

    # --- requests ----------------------------------------------------------
    counts = rng.multinomial(n_requests, rate / rate.sum())
    phr = PhrasingSampler(held_t, held_f, rng)
    reqs, forced = [], 0
    for s in range(n_slots):
        if counts[s] == 0:
            continue
        in_zone = {z: np.where(traj[:, s] == z)[0] for z in range(n_z)}
        times = np.sort(rng.uniform(s * slot_s, (s + 1) * slot_s, size=counts[s]))
        for t in times:
            hour = 24.0 * t / horizon_s
            z = int(rng.choice(n_z, p=zone_p[s]))
            cands = in_zone[z]
            if len(cands) == 0:
                forced += 1
                di = int(rng.integers(n_devices))
            else:
                di = int(cands[int(rng.integers(len(cands)))])
            origin = int(traj[di, s])
            tw = type_weights(hour)
            names = list(tw)
            wv = np.array([tw[k] for k in names])
            st = names[int(rng.choice(len(names), p=wv / wv.sum()))]
            spec = SERVICE_TYPES[st]
            cpu, mem = resources.size(spec, rng)
            dev = devices[di]["device_id"]
            reqs.append({
                "t_s": round(float(t), 2), "device_id": dev,
                "origin_zone": f"z{origin}", "phase": phase_of(hour),
                "intent_text": phr.sample(dev, st, hour),
                "truth": {**ground_truth_profile(st),
                          "latency_ms": spec["latency_ms"],
                          "cpu": cpu, "mem": mem,
                          "lifetime_base_s": round(resources.lifetime(rng), 2),
                          "degrade_floor": spec["degrade_floor"],
                          "accel_pref": spec["accel_pref"]}})
    reqs.sort(key=lambda r: r["t_s"])
    for i, r in enumerate(reqs):
        r["req_id"] = f"r{seed}_{i:05d}"

    # --- events ------------------------------------------------------------
    node_ids = [n["node_id"] for z in topo["zones"] for n in z["nodes"]]
    events = failure_model.failure_schedule(node_ids, horizon_s, rng,
                                            min_failures=min_failures)
    for st, s in SERVICE_TYPES.items():
        if isinstance(s["register_h"], (int, float)):
            events.append({"type": "register_service",
                           "t_s": round(s["register_h"] / 24.0 * horizon_s, 1),
                           "name": st, "spec": s})
    events.sort(key=lambda e: e["t_s"])

    meta = {"seed": seed, "n_requests": len(reqs), "n_devices": n_devices,
            "horizon_s": horizon_s, "slot_s": slot_s, "n_slots": n_slots,
            "surge_zone": f"z{sz}", "surge_center_s": round(center * slot_s, 1),
            "surge_window_s": [round((center - SURGE_HALF_WIDTH_SLOTS) * slot_s, 1),
                               round((center + SURGE_HALF_WIDTH_SLOTS) * slot_s, 1)],
            "phases": PHASES, "mobility_forced_device_picks": forced,
            "activity_source": activity.get("source", ""),
            "base_day": activity.get("base_day"), "surge_day": activity.get("surge_day"),
            "resource_source": resources.source,
            "n_failures": sum(e["type"] == "node_failure" for e in events),
            "lifetime_scale": 1.0}
    return {"meta": meta, "devices": devices, "requests": reqs, "events": events}


def save(wl, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(wl, f)


def load(path, lifetime_scale=None):
    with gzip.open(path, "rt") as f:
        wl = json.load(f)
    if lifetime_scale is not None:
        wl["meta"]["lifetime_scale"] = float(lifetime_scale)
    return wl


def lifetime_of(req, wl):
    return max(1.0, req["truth"]["lifetime_base_s"] * wl["meta"]["lifetime_scale"])


def drift_report(wl):
    """Evidence the workload is non-stationary: per-phase type mix, zone
    mix, distinct phrasings, and first-seen times of new types."""
    from collections import Counter
    out = {}
    for ph, _, _ in PHASES:
        rs = [r for r in wl["requests"] if r["phase"] == ph]
        n = max(len(rs), 1)
        out[ph] = {"n": len(rs),
                   "type_mix": {k: round(v / n, 3) for k, v in sorted(
                       Counter(r["truth"]["service_type"] for r in rs).items())},
                   "zone_mix": {k: round(v / n, 3) for k, v in sorted(
                       Counter(r["origin_zone"] for r in rs).items())},
                   "distinct_texts": len({r["intent_text"] for r in rs})}
    first = {}
    for r in wl["requests"]:
        first.setdefault(r["truth"]["service_type"], r["t_s"])
    out["first_seen_s"] = first
    return out
