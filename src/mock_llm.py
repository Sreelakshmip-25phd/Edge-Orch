"""MOCK LLM - a deterministic test double for smoke runs and unit tests.

!!! Nothing produced with this provider is a result. !!!

It exists so the whole pipeline (every orchestrator, ablation and baseline,
the metrics and the evaluator) can be exercised on a machine with no GPU
and no model server. It answers each prompt kind with a simple heuristic
reading of the prompt's own JSON, injects a small deterministic error rate
(so shadow checks, verification/retry and fallbacks are exercised), and
reports PLACEHOLDER latencies/tokens that are NOT measurements. Runs using
it are tagged latency_source="placeholder" and llm="mock" in their
metadata, and live under results/smoke/.
"""
import hashlib
import json
import math
import re

# PLACEHOLDER latencies (ms) - NOT measured. Real numbers come from
# scripts/calibrate_latency.py on the GPU machine.
_PLACEHOLDER_MS = {"slm": 350.0, "llm": 1200.0}
_SLM_KINDS = {"translate", "shadow_check", "zone_choice"}

KEYWORDS = {
    "video_analytics": ["camera", "video", "footage", "surveillance", "stream", "cctv"],
    "drone_control": ["drone", "flight", "fly", "uav", "aerial"],
    "ar_session": ["ar ", "augmented", "overlay", "immersive", "reality"],
    "iot_aggregator": ["sensor", "iot", "telemetry", "air-quality", "air quality",
                       "readings", "environmental"],
    "traffic_monitor": ["traffic", "vehicle", "junction", "intersection", "congestion", "road"],
    "federated_ml": ["federated", "training", "learning", "model", "partition"],
    "digital_twin": ["twin", "simulate", "replica", "grid"],
    "crowd_safety": ["crowd", "density", "crush", "overcrowd", "people", "packed"],
    "ev_charging": ["charging", "charger", "electric", "ev "],
}


def _h(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _json_after(marker, text):
    i = text.find(marker)
    if i < 0:
        return {}
    try:
        return json.loads(text[i + len(marker):].strip().split("\n")[0])
    except Exception:
        return {}


def classify(text):
    t = " " + text.lower() + " "
    scores = {k: sum(t.count(w) for w in ws) for k, ws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "traffic_monitor"


def _latency_class(text, default):
    m = re.search(r"(\d+)\s*ms", text.lower())
    if m:
        ms = int(m.group(1))
        return "realtime" if ms < 100 else ("interactive" if ms <= 500 else "batch")
    return default


def translate(system, text, err_mod=25):
    cat = _json_after("CATALOG:", system)
    st = classify(text)
    h = _h(text)
    if cat and h % err_mod == 0:                         # injected error
        names = sorted(cat)
        st = names[(names.index(st) + 1) % len(names)] if st in names else names[0]
    if st in cat:
        c = cat[st]
        return {"service_type": st, "latency_class": c["latency_class"],
                "data_locality": c["data_locality"], "priority": c["priority"]}
    local = any(w in text.lower() for w in ("local", "not leave", "stays", "in the area",
                                            "without sending", "must not"))
    return {"service_type": {"crowd_safety": "crowd monitoring",
                             "ev_charging": "ev charge scheduling"}.get(st, st),
            "latency_class": _latency_class(text, "interactive"),
            "data_locality": "zone_local" if local else "any",
            "priority": "high" if "alert" in text.lower() else "normal"}


def decide(user):
    u = json.loads(user)
    opts, zones = u.get("options", {}), u.get("zones", {})
    if _h(user) % 12 == 0 and "feedback" not in u:
        return {"action": "place", "zone": "z999", "victim": None,
                "degrade_level": None, "reason": "mock: injected invalid proposal"}
    if opts.get("place"):
        z = min(opts["place"], key=lambda z: zones.get(z, {}).get("rtt_ms", 0))
        return {"action": "place", "zone": z, "victim": None, "degrade_level": None,
                "reason": "mock: nearest feasible zone"}
    if opts.get("preempt"):
        v = opts["preempt"][0]
        return {"action": "preempt", "zone": v["zone"], "victim": v["victim"],
                "degrade_level": None, "reason": "mock: lowest-priority victim"}
    if opts.get("degrade"):
        d = max(opts["degrade"], key=lambda d: d["level"])
        return {"action": "degrade", "zone": d["zone"], "victim": None,
                "degrade_level": d["level"], "reason": "mock: mildest degradation"}
    return {"action": "reject", "zone": None, "victim": None, "degrade_level": None,
            "reason": "mock: nothing feasible"}


def _fits(node, dem):
    return node.get("cpu_free", 0) >= dem.get("cpu", 0) - 1e-6 and \
        node.get("mem_free", 0) >= dem.get("mem", 0) - 1e-6


def tool_policy(u, sample_idx=0):
    """Shared by react_step / lats_expand: read zones nearest-first, place
    on the first node that fits, else pre-empt if allowed, else finish."""
    req = u.get("request", {})
    st = classify(req.get("intent", ""))
    sizes = u.get("catalog_sizes", {})
    dem = sizes.get(st, {"cpu": 1.0, "mem": 1.0})
    zones = [z["zone"] for z in sorted(req.get("zones", []), key=lambda z: z.get("rtt_ms", 0))]
    seen, tried = {}, set()
    for h in u.get("history", []):
        a, obs = h.get("action", {}), h.get("observation", {})
        if a.get("tool") == "read_digest" and isinstance(obs, dict) and "nodes" in obs:
            seen[a["args"].get("zone")] = obs["nodes"]
        if a.get("tool") == "try_place":
            tried.add(a["args"].get("node"))
    options = []
    for z, nodes in seen.items():
        for n in nodes:
            if n["node"] not in tried and _fits(n, dem):
                options.append(("try_place", {"zone": z, "node": n["node"], "service_type": st}))
    if options:
        a = options[sample_idx % len(options)]
        return {"thought": "mock: a node fits", "tool": a[0], "args": a[1]}
    for z in zones:
        if z not in seen:
            return {"thought": "mock: inspect next zone", "tool": "read_digest",
                    "args": {"zone": z}}
    return {"thought": "mock: nothing fits", "tool": "finish", "args": {}}


def mock_respond(kind, system, user, temperature=0.0):
    role = "slm" if kind in _SLM_KINDS else "llm"
    if kind in ("translate", "shadow_check", "translate_llm", "intent"):
        text = user.split("INPUT:", 1)[-1].strip()
        out = translate(system, text)
    elif kind == "decide":
        out = decide(user)
    elif kind == "rule_author":
        u = json.loads(user)
        zc = u.get("successful_zone_counts", {})
        out = {"condition": {"service_type": u.get("service_type")},
               "action": {"prefer_zone": max(zc, key=zc.get) if zc else None},
               "rationale": "mock: majority"}
    elif kind in ("react_step", "lats_expand"):
        u = json.loads(user)
        out = tool_policy(u, sample_idx=_h(system + user + str(temperature)) % 3
                          if kind == "lats_expand" else 0)
    elif kind == "lats_value":
        u = json.loads(user)
        a = u.get("candidate", {})
        out = {"score": 8 if a.get("tool") == "try_place" else (4 if a.get("tool") == "read_digest" else 1)}
    elif kind == "lats_reflect":
        out = {"reflection": "mock: the attempted node was full; try another zone"}
    elif kind == "observe":
        u = json.loads(user)
        zs = u.get("zones", {})
        order = sorted(zs, key=lambda z: -zs[z].get("top_cpu_free", 0))
        out = {"candidate_zones": order[:4], "hotspots": order[-2:], "summary": "mock"}
    elif kind == "plan":
        u = json.loads(user)
        cands = u.get("observability", {}).get("candidate_zones") or list(u.get("zones", {}))
        k = len(u.get("feedback", []))
        if k < len(cands):
            out = {"action": "place", "zone": cands[k], "reason": "mock"}
        elif u.get("preempt_candidates"):
            v = u["preempt_candidates"][0]
            out = {"action": "preempt", "zone": v["zone"], "victim": v["victim"], "reason": "mock"}
        else:
            out = {"action": "reject", "reason": "mock"}
    elif kind == "act":
        u = json.loads(user)
        plan = u.get("plan", {})
        nodes = u.get("zone_nodes", [])
        dem = u.get("demand", {})
        fit = [n for n in nodes if _fits(n, dem)]
        node = (fit or nodes or [{"node": None}])[0]["node"]
        if plan.get("action") == "preempt":
            node = u.get("victim_node", node)
        out = {"zone": plan.get("zone"), "node": node, "victim": plan.get("victim"),
               "degrade_level": 1.0}
    elif kind == "zone_choice":
        u = json.loads(user)
        zs = u.get("zones", {})
        ok = [z for z, d in zs.items() if d.get("fits")]
        out = {"zone": min(ok, key=lambda z: zs[z].get("rtt_ms", 0)) if ok else None}
    else:
        out = {"error": f"mock has no handler for kind={kind}"}
    raw = json.dumps(out)
    tin = (len(system) + len(user)) // 4
    tout = len(raw) // 4
    # deterministic jitter around the placeholder so distributions aren't degenerate
    wall = _PLACEHOLDER_MS[role] * (0.8 + 0.4 * ((_h(user + kind) % 1000) / 1000.0))
    return raw, tin, tout, round(wall, 2), True
