"""Auto-generated, auto-scored reasoning cases for the global agent's
one-call structured decision (DECIDE_SYS in src/global_agent.py).

Replaces the old 4 hand-written cases with 100 deterministic ones built in
exactly the JSON shape GlobalAgent._prompt() sends:
   60 zone selection  - several feasible zones + infeasible distractors
   25 pre-emption     - nothing fits; strictly-lower-priority victims offered
   10 degradation     - only reduced-resource placements fit
    5 mixed           - both pre-emption and degradation are possible
Each case records the set of ACCEPTABLE answers (anything the verifier in
GlobalAgent._verify() would accept and the guidance doesn't forbid) and a
PREFERRED answer by a documented heuristic:
   zone      lowest rtt_ms among options.place
   preempt   lowest victim priority, then least remaining_s
   degrade   highest level offered
   mixed     both families acceptable; preferred = the milder degradation
             if its level >= 0.8, else the preferred victim
"""
import json
import random

PRIO = ["low", "normal", "high", "critical"]
TYPES = {"traffic_monitor": ("interactive", "any", "normal", 0.4),
         "video_analytics": ("realtime", "zone_local", "high", 0.6),
         "iot_aggregator": ("batch", "any", "low", 0.4),
         "ar_session": ("realtime", "zone_local", "high", 0.6),
         "federated_ml": ("batch", "any", "normal", 0.4),
         "digital_twin": ("interactive", "any", "normal", 0.6),
         "drone_control": ("realtime", "zone_local", "critical", 0.8)}


def _profile(st):
    lc, loc, pr, _ = TYPES[st]
    return {"service_type": st, "latency_class": lc, "data_locality": loc, "priority": pr}


def _zones(rng, n, origin="z0"):
    zs = {}
    for i in range(n):
        z = f"z{i}"
        zs[z] = {"rtt_ms": 0.0 if z == origin else round(rng.uniform(4.3, 9.4), 2)}
    return zs


def _fill(zs, z, cpu_free, mem_free, rng):
    zs[z].update(cpu_free=round(cpu_free + rng.uniform(0, 3), 2),
                 top_nodes=[[round(cpu_free, 2), round(mem_free, 2)],
                            [round(cpu_free * 0.5, 2), round(mem_free * 0.6, 2)]],
                 util=round(rng.uniform(0.3, 0.95), 2), healthy_frac=1.0)


def _case(name, kind, profile, cpu, mem, origin, zones, options, acceptable, preferred, floor):
    body = {"request": {"profile": profile, "cpu": cpu, "mem": mem, "origin_zone": origin,
                        "degrade_floor": floor},
            "zones": zones, "options": options, "memory_hints": {"failed_zones": []}}
    return {"name": name, "kind": kind, "user": json.dumps(body, sort_keys=True),
            "options": options, "acceptable": acceptable, "preferred": preferred}


def generate(seed=7):
    rng = random.Random(seed)
    cases = []
    any_types = [t for t, v in TYPES.items() if v[1] == "any"]
    # --- 60 zone selection -------------------------------------------------------
    for i in range(60):
        st = rng.choice(any_types)
        cpu, mem = round(rng.uniform(0.5, 4.0), 2), round(rng.uniform(0.3, 3.0), 2)
        zs = _zones(rng, rng.randint(4, 7))
        zones = list(zs)
        feas = rng.sample(zones[1:], rng.randint(1, min(3, len(zones) - 1)))
        for z in zones:
            if z in feas:
                _fill(zs, z, cpu + rng.uniform(0.5, 8), mem + rng.uniform(0.5, 6), rng)
            else:
                _fill(zs, z, max(cpu - rng.uniform(0.3, cpu), 0.0), mem * rng.uniform(0.2, 0.9), rng)
        place = sorted(feas, key=lambda z: zs[z]["rtt_ms"])
        opts = {"place": place, "preempt": [], "degrade": []}
        acc = [{"action": "place", "zone": z} for z in place]
        cases.append(_case(f"zone_{i:02d}", "zone", _profile(st), cpu, mem, "z0", zs, opts, acc,
                           {"action": "place", "zone": place[0]}, TYPES[st][3]))
    # --- 25 pre-emption ------------------------------------------------------------
    for i in range(25):
        st = rng.choice(["video_analytics", "ar_session", "drone_control"])
        prof = _profile(st)
        pr = PRIO.index(prof["priority"])
        cpu, mem = round(rng.uniform(0.8, 3.0), 2), round(rng.uniform(0.3, 2.0), 2)
        zs = _zones(rng, rng.randint(2, 4))
        for z in zs:
            _fill(zs, z, cpu * rng.uniform(0.1, 0.5), mem * 0.5, rng)
        victims = []
        for k in range(rng.randint(2, 4)):
            vp = PRIO[rng.randint(0, pr - 1)]
            victims.append({"victim": f"r{i}_{k}", "zone": "z0", "node": f"n{k}",
                            "priority": vp, "service_type": rng.choice(list(TYPES)),
                            "remaining_s": round(rng.uniform(5, 3000), 1),
                            "cpu": round(cpu + rng.uniform(0, 1), 2), "mem": round(mem + 0.2, 2)})
        victims.sort(key=lambda v: (PRIO.index(v["priority"]), v["remaining_s"]))
        opts = {"place": [], "preempt": victims, "degrade": []}
        acc = [{"action": "preempt", "victim": v["victim"]} for v in victims]
        cases.append(_case(f"preempt_{i:02d}", "preempt", prof, cpu, mem, "z0",
                           {"z0": zs["z0"]}, opts, acc,
                           {"action": "preempt", "victim": victims[0]["victim"]}, TYPES[st][3]))
    # --- 10 degradation -------------------------------------------------------------
    for i in range(10):
        st = rng.choice(list(TYPES))
        prof = _profile(st)
        floor = TYPES[st][3]
        cpu, mem = round(rng.uniform(1.0, 4.0), 2), round(rng.uniform(0.5, 2.0), 2)
        zs = _zones(rng, 3)
        levels = [l for l in (0.8, 0.6, 0.4) if l >= floor]
        top = rng.choice(levels)
        for z in zs:
            _fill(zs, z, cpu * top + 0.05, mem * top + 0.05, rng)
        allowed = ["z0"] if prof["data_locality"] == "zone_local" else list(zs)
        deg = [{"zone": z, "level": l} for z in allowed for l in levels if l <= top]
        opts = {"place": [], "preempt": [], "degrade": deg}
        acc = [{"action": "degrade", "zone": d["zone"], "level": d["level"]} for d in deg]
        best = max(deg, key=lambda d: (d["level"], -zs[d["zone"]]["rtt_ms"]))
        cases.append(_case(f"degrade_{i:02d}", "degrade", prof, cpu, mem, "z0",
                           {z: zs[z] for z in allowed}, opts, acc,
                           {"action": "degrade", "zone": best["zone"], "level": best["level"]}, floor))
    # --- 5 mixed -----------------------------------------------------------------------
    for i in range(5):
        st = "video_analytics"
        prof = _profile(st)
        cpu, mem = 2.0, 1.0
        zs = _zones(rng, 1)
        lvl = rng.choice([0.8, 0.6])
        _fill(zs, "z0", cpu * lvl + 0.05, mem * lvl + 0.05, rng)
        v = {"victim": f"m{i}", "zone": "z0", "node": "n0", "priority": "low",
             "service_type": "iot_aggregator", "remaining_s": round(rng.uniform(10, 900), 1),
             "cpu": 2.5, "mem": 1.2}
        opts = {"place": [], "preempt": [v], "degrade": [{"zone": "z0", "level": lvl}]}
        acc = [{"action": "preempt", "victim": v["victim"]},
               {"action": "degrade", "zone": "z0", "level": lvl}]
        pref = acc[1] if lvl >= 0.8 else acc[0]
        cases.append(_case(f"mixed_{i:02d}", "mixed", prof, cpu, mem, "z0", zs, opts, acc, pref, 0.6))
    return cases


def _match(ans, target):
    if ans.get("action") != target["action"]:
        return False
    if target["action"] == "place":
        return ans.get("zone") == target["zone"]
    if target["action"] == "preempt":
        return ans.get("victim") == target["victim"]
    if target["action"] == "degrade":
        try:
            return ans.get("zone") == target["zone"] and abs(float(ans.get("degrade_level")) - target["level"]) < 1e-6
        except (TypeError, ValueError):
            return False
    return False


def score(case, ans):
    """-> dict(valid_json, acceptable, preferred)"""
    ok_json = isinstance(ans, dict) and "action" in ans and "_parse_error" not in ans
    acc = ok_json and any(_match(ans, a) for a in case["acceptable"])
    pref = ok_json and _match(ans, case["preferred"])
    return {"valid_json": bool(ok_json), "acceptable": bool(acc), "preferred": bool(pref)}


if __name__ == "__main__":
    from collections import Counter
    cs = generate()
    print(len(cs), Counter(c["kind"] for c in cs))
