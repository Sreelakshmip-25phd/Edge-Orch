"""Phase 10 - every metric, computed ONLY from a telemetry.Telemetry log.

compute(tel) -> {"scalars": {flat numeric metrics used for CIs / paired
tests}, "series": {time series}, "tables": {breakdowns}}. It refuses to run
on incomplete records (Telemetry.assert_complete).

The nine target metrics:
 (1) SLM/LLM calls over time          series.calls_by_bin, scalars calls_*
     cumulative + windowed, split cache-hit vs SLM vs LLM, fresh vs disk
 (2) services placed                  scalars acceptance*, tables per phase/zone/priority
 (3) successful completion            scalars completion_rate (never = acceptance)
 (4) execution/placement failures     scalars fail_* by cause
 (5) latency                          scalars lat_* incl. the three comparisons
 (6) multi-model comparison           local_llm/model_compare.py output (not a sim metric)
 (7) escalation & cache-hit over time series.*_by_bin
 (8) pre-emption / degradation        tables.preemption, scalars preempt_*, degrade_*
 (9) token cost per request           scalars tokens_* by role and prompt/completion
plus load balance (utilisation variance across zones / nodes), per-zone
placement counts, translation accuracy, locality violations, and the
call-reduction trend (does per-request model usage fall over time?).

Three outcomes are always kept apart:
  acceptance          placed at all, immediately on arrival
  completion          ran to its natural end (not lost to eviction/failure)
  escalation success  of the requests that needed escalation, share placed
"""
from collections import Counter, defaultdict

import numpy as np

from scenario import PRIORITIES

MEMORY_SOURCES = ("memory",)
LLM_SOURCES = ("llm_fresh", "llm_cached_disk")


def _stats(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    a = np.asarray(xs)
    return {"n": len(xs), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95))}


def _rate(num, den):
    return float(num) / den if den else None


def _slope(y, w=None):
    """Weighted least-squares slope of y over bin index (per bin)."""
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return None
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    x, y, w = x[ok], y[ok], w[ok]
    if w.sum() == 0:
        return None
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    den = np.sum(w * (x - xm) ** 2)
    return float(np.sum(w * (x - xm) * (y - ym)) / den) if den else None


def invocations(r):
    """SLM + LLM invocations charged to one request (simulated world:
    fresh and disk-cached both count as invocations; errors don't)."""
    c = r.calls
    return (c["slm_fresh"] + c["slm_cached_disk"], c["llm_fresh"] + c["llm_cached_disk"])


def compute(tel, n_bins=24):
    tel.assert_complete()
    R = sorted(tel.requests.values(), key=lambda r: r.send_time)
    n = len(R)
    meta = tel.meta
    horizon = float(meta.get("horizon_s") or (max(r.send_time for r in R) + 1.0))
    edges = np.linspace(0.0, horizon, n_bins + 1)

    def b(t):
        return int(min(max(np.searchsorted(edges, t, side="right") - 1, 0), n_bins - 1))

    S, series, tables = {}, {}, {}

    # --- (2)(3) outcomes -------------------------------------------------------
    acc = np.array([r.accepted for r in R])
    comp = np.array([r.end_cause == "completed" for r in R])
    esc = np.array([r.escalated for r in R])
    st_ok = np.array([bool(r.service_type_correct) for r in R])
    S["n_requests"] = n
    S["acceptance_rate"] = float(acc.mean()) if n else None
    S["completion_rate"] = float(comp.mean()) if n else None
    S["completed_correct_rate"] = float((comp & st_ok).mean()) if n else None
    S["escalation_rate"] = float(esc.mean()) if n else None
    S["escalation_success"] = float(acc[esc].mean()) if esc.any() else None
    S["rejection_rate"] = float(np.mean([r.end_cause == "rejected" for r in R])) if n else None
    for ph in sorted({r.phase for r in R}):
        m = np.array([r.phase == ph for r in R])
        S[f"acceptance_{ph}"] = float(acc[m].mean())
        S[f"completion_{ph}"] = float(comp[m].mean())
    tables["per_origin_zone"] = {z: {"n": int(m.sum()), "acceptance": float(acc[m].mean()),
                                     "completion": float(comp[m].mean())}
                                 for z in sorted({r.origin_zone for r in R})
                                 for m in [np.array([r.origin_zone == z for r in R])]}
    tables["per_priority"] = {p: {"n": int(m.sum()), "acceptance": float(acc[m].mean()),
                                  "completion": float(comp[m].mean())}
                              for p in PRIORITIES
                              for m in [np.array([r.priority_true == p for r in R])] if m.any()}
    for p, v in tables["per_priority"].items():
        S[f"acceptance_prio_{p}"] = v["acceptance"]
    placed_zone = Counter(r.zone_final for r in R if r.accepted)
    tables["placements_per_zone"] = dict(sorted(placed_zone.items()))
    S["cross_zone_share"] = _rate(sum(r.cross_zone for r in R if r.accepted), int(acc.sum()))
    tables["paths"] = dict(Counter(r.path for r in R))
    tables["actions"] = dict(Counter(r.action for r in R))
    tables["decision_sources"] = dict(Counter(r.decision_source for r in R))
    for p, c in tables["paths"].items():
        S[f"path_{p}_share"] = c / n

    # locality: a zone_local service placed outside its origin zone
    viol = [r for r in R if r.accepted and r.locality_true == "zone_local"
            and r.zone_final != r.origin_zone]
    S["locality_violation_rate"] = _rate(len(viol), int(acc.sum()))

    # --- translation accuracy --------------------------------------------------
    S["translation_service_type_acc"] = float(st_ok.mean()) if n else None
    full = [r.translation_correct for r in R if r.translation_correct is not None]
    S["translation_full_acc"] = float(np.mean(full)) if full else None
    by_src = defaultdict(list)
    for r in R:
        by_src[r.translation_source].append(bool(r.service_type_correct))
    tables["translation_acc_by_source"] = {k: {"n": len(v), "acc": float(np.mean(v))}
                                           for k, v in by_src.items()}
    S["cache_hit_translation_acc"] = tables["translation_acc_by_source"].get("cache", {}).get("acc")
    sh = [r for r in R if r.shadow_checked]
    S["shadow_checks"] = len(sh)
    S["shadow_disagree_rate"] = _rate(sum(not r.shadow_agree for r in sh), len(sh))
    novel_true = [r for r in R if r.service_type_true in ("crowd_safety", "ev_charging")]
    S["new_type_requests"] = len(novel_true)
    S["new_type_acceptance"] = _rate(sum(r.accepted for r in novel_true), len(novel_true))
    S["new_type_type_acc"] = _rate(sum(bool(r.service_type_correct) for r in novel_true), len(novel_true))
    tables["type_resolution"] = dict(Counter(r.type_resolution for r in R))

    # --- (1)(7) calls / cache / escalation over time -------------------------------
    keys = ("requests", "slm_fresh", "slm_cached_disk", "llm_fresh", "llm_cached_disk",
            "nonreq_llm_fresh", "nonreq_llm_cached_disk", "cache_hits", "slm_translations",
            "escalations", "llm_stage", "slm_inv", "llm_inv", "llm_inv_esc", "memory_esc")
    bins = {k: np.zeros(n_bins) for k in keys}
    for r in R:
        i = b(r.send_time)
        bins["requests"][i] += 1
        for k in ("slm_fresh", "slm_cached_disk", "llm_fresh", "llm_cached_disk"):
            bins[k][i] += r.calls[k]
        s_inv, l_inv = invocations(r)
        bins["slm_inv"][i] += s_inv
        bins["llm_inv"][i] += l_inv
        bins["cache_hits"][i] += r.translation_source == "cache"
        bins["slm_translations"][i] += r.translation_source in ("slm_fresh", "slm_cached_disk")
        bins["escalations"][i] += r.escalated
        bins["llm_stage"][i] += r.reached_llm_stage
        if r.escalated:
            bins["llm_inv_esc"][i] += l_inv
            bins["memory_esc"][i] += r.decision_source in MEMORY_SOURCES
    for c in tel.llm_calls:
        key = f"nonreq_{c.role}_{c.source}"
        if c.req_id is None and key in bins:          # e.g. procedural-rule authoring
            bins[key][b(c.t)] += 1
    reqs = bins["requests"]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_req = (bins["slm_inv"] + bins["llm_inv"] + bins["nonreq_llm_fresh"]
                   + bins["nonreq_llm_cached_disk"]) / reqs
        series["calls_by_bin"] = {
            "edges_s": edges.tolist(),
            **{k: v.tolist() for k, v in bins.items()},
            "invocations_per_request": np.where(reqs > 0, per_req, np.nan).tolist(),
            "slm_per_request": np.where(reqs > 0, bins["slm_inv"] / reqs, np.nan).tolist(),
            "llm_per_request": np.where(reqs > 0, (bins["llm_inv"] + bins["nonreq_llm_fresh"]
                                                   + bins["nonreq_llm_cached_disk"]) / reqs,
                                        np.nan).tolist(),
            "cache_hit_rate": np.where(reqs > 0, bins["cache_hits"] / reqs, np.nan).tolist(),
            "escalation_rate": np.where(reqs > 0, bins["escalations"] / reqs, np.nan).tolist(),
            "llm_stage_rate": np.where(reqs > 0, bins["llm_stage"] / reqs, np.nan).tolist(),
            # need-normalised: cost per escalation / per translation, which
            # removes the daily load curve from the call-reduction question
            "llm_per_escalation": np.where(bins["escalations"] > 0,
                                           bins["llm_inv_esc"] / bins["escalations"],
                                           np.nan).tolist(),
            "memory_share_of_escalations": np.where(bins["escalations"] > 0,
                                                    bins["memory_esc"] / bins["escalations"],
                                                    np.nan).tolist(),
            "cache_miss_rate": np.where(reqs > 0, 1 - bins["cache_hits"] / reqs, np.nan).tolist(),
            "acceptance_rate": [float(acc[[b(r.send_time) == i for r in R]].mean())
                                if reqs[i] else None for i in range(n_bins)],
        }
    tot_calls = Counter(f"{c.role}_{c.source}" for c in tel.llm_calls)
    for k in ("slm_fresh", "slm_cached_disk", "slm_error", "llm_fresh", "llm_cached_disk",
              "llm_error"):
        S[f"calls_{k}"] = tot_calls.get(k, 0)
        S[f"calls_{k}_per_req"] = tot_calls.get(k, 0) / n if n else None
    S["invocations_per_req"] = (sum(tot_calls.get(k, 0) for k in
                                    ("slm_fresh", "slm_cached_disk", "llm_fresh",
                                     "llm_cached_disk"))) / n if n else None
    S["llm_invocations_per_req"] = (tot_calls.get("llm_fresh", 0) +
                                    tot_calls.get("llm_cached_disk", 0)) / n if n else None
    S["slm_invocations_per_req"] = (tot_calls.get("slm_fresh", 0) +
                                    tot_calls.get("slm_cached_disk", 0)) / n if n else None
    S["fresh_calls_per_req"] = (tot_calls.get("slm_fresh", 0) + tot_calls.get("llm_fresh", 0)) / n
    S["llm_stage_rate"] = float(np.mean([r.reached_llm_stage for r in R])) if n else None
    S["cache_hit_rate"] = float(np.mean([r.translation_source == "cache" for r in R])) if n else None
    # call-reduction trend: per-request invocation rate, first vs last quarter of
    # the day and the weighted slope over bins (negative = falling)
    q = max(n_bins // 4, 1)
    first, last = reqs[:q].sum(), reqs[-q:].sum()
    inv_bins = bins["slm_inv"] + bins["llm_inv"]
    S["inv_per_req_first_quarter"] = _rate(inv_bins[:q].sum(), first)
    S["inv_per_req_last_quarter"] = _rate(inv_bins[-q:].sum(), last)
    S["llm_per_req_first_quarter"] = _rate(bins["llm_inv"][:q].sum(), first)
    S["llm_per_req_last_quarter"] = _rate(bins["llm_inv"][-q:].sum(), last)
    S["inv_per_req_slope_per_bin"] = _slope(
        np.where(reqs > 0, inv_bins / np.maximum(reqs, 1), np.nan), reqs)
    S["cache_hit_first_quarter"] = _rate(bins["cache_hits"][:q].sum(), first)
    S["cache_hit_last_quarter"] = _rate(bins["cache_hits"][-q:].sum(), last)
    # need-normalised versions (the raw per-request rate above rises and falls
    # with the daily load curve: at night almost nothing escalates)
    esc_b = bins["escalations"]
    S["llm_per_escalation"] = _rate(bins["llm_inv_esc"].sum(), esc_b.sum())
    S["llm_per_escalation_first_quarter"] = _rate(bins["llm_inv_esc"][:q].sum(), esc_b[:q].sum())
    S["llm_per_escalation_last_quarter"] = _rate(bins["llm_inv_esc"][-q:].sum(), esc_b[-q:].sum())
    S["llm_per_escalation_slope_per_bin"] = _slope(
        np.where(esc_b > 0, bins["llm_inv_esc"] / np.maximum(esc_b, 1), np.nan), esc_b)
    S["memory_share_of_escalations"] = _rate(bins["memory_esc"].sum(), esc_b.sum())
    busy = np.where(esc_b >= max(esc_b.max() * 0.25, 1))[0]      # hours with real escalation load
    if len(busy) >= 4:
        h = len(busy) // 2
        S["llm_per_escalation_busy_first_half"] = _rate(bins["llm_inv_esc"][busy[:h]].sum(),
                                                        esc_b[busy[:h]].sum())
        S["llm_per_escalation_busy_second_half"] = _rate(bins["llm_inv_esc"][busy[h:]].sum(),
                                                         esc_b[busy[h:]].sum())
    S["cache_miss_first_quarter"] = _rate(first - bins["cache_hits"][:q].sum(), first)
    S["cache_miss_last_quarter"] = _rate(last - bins["cache_hits"][-q:].sum(), last)
    S["cache_hit_shared_rate"] = float(np.mean([bool(r.cache_hit_shared) for r in R])) if n else None
    # cache metrics only exist for a system that has an intent cache (a
    # lookup sets cache_similarity); elsewhere they would read "0% hits /
    # 100% misses", which is not a measurement
    if not any(r.cache_similarity is not None and r.translation_source != "static_rule"
               for r in R):
        for k in ("cache_hit_rate", "cache_hit_first_quarter", "cache_hit_last_quarter",
                  "cache_miss_first_quarter", "cache_miss_last_quarter", "cache_hit_shared_rate"):
            S[k] = None

    # --- (4) failures by cause -----------------------------------------------------
    causes = Counter(r.end_cause for r in R)
    S["fail_node_loss"] = causes.get("node_loss", 0)
    S["fail_preempt_unmigrated"] = causes.get("preempt_unmigrated", 0)
    S["fail_rate_after_accept"] = _rate(S["fail_node_loss"] + S["fail_preempt_unmigrated"],
                                        int(acc.sum()))
    ev = Counter(e.kind for e in tel.events)
    S["node_failures"] = ev.get("node_failure", 0)
    S["interrupted_services"] = sum(r.n_interruptions > 0 for r in R)
    S["interrupted_but_completed"] = sum(r.n_interruptions > 0 and r.end_cause == "completed"
                                         for r in R)
    mig = [e for e in tel.events if e.kind == "migrate"]
    S["replans_after_failure"] = sum(e.data.get("cause") == "node_failure" for e in mig)
    fail_displaced = sum(e.data.get("n_displaced", 0) for e in tel.events if e.kind == "node_failure")
    S["failure_recovery_rate"] = _rate(S["replans_after_failure"], fail_displaced)
    tables["failure_causes"] = dict(causes)

    # --- (5) latency -----------------------------------------------------------------
    # Only the orchestration path is reported: request arrival -> placement
    # committed (network, translation, escalation, decision, cross-zone hop).
    # Service start-up ("deployment") is an assumed per-device constant, not a
    # measurement, so it stays in the event log but out of every metric.
    A = [r for r in R if r.accepted]

    def setup(r):
        return r.total_latency_ms - r.latency_breakdown.get("deployment", 0.0)

    for c in ("transport_in", "translation", "escalation", "decision", "cross_zone"):
        S[f"lat_{c}_mean"] = float(np.mean([r.latency_breakdown.get(c, 0.0) for r in A])) if A else None
    S.update({f"lat_setup_{k}": v for k, v in _stats([setup(r) for r in A]).items() if k != "n"})
    cmp = {
        "zone_vs_cross_zone": (
            _stats([setup(r) for r in A if not r.cross_zone]),
            _stats([setup(r) for r in A if r.cross_zone])),
        "memory_vs_llm_decision": (
            _stats([r.latency_breakdown.get("decision", 0.0) for r in R
                    if r.decision_source in MEMORY_SOURCES]),
            _stats([r.latency_breakdown.get("decision", 0.0) for r in R
                    if r.decision_source in LLM_SOURCES])),
        "cache_vs_slm_translation": (
            _stats([r.latency_breakdown.get("translation", 0.0) for r in R
                    if r.translation_source == "cache"]),
            _stats([r.latency_breakdown.get("translation", 0.0) for r in R
                    if r.translation_source in ("slm_fresh", "slm_cached_disk")])),
    }
    tables["latency_comparisons"] = {k: {"a": a, "b": bb} for k, (a, bb) in cmp.items()}
    for k, (a, bb) in cmp.items():
        S[f"lat_{k}_a_mean"], S[f"lat_{k}_b_mean"] = a["mean"], bb["mean"]
        S[f"lat_{k}_a_p95"], S[f"lat_{k}_b_p95"] = a["p95"], bb["p95"]
    tables["latency_by_path"] = {p: _stats([r.total_latency_ms for r in R if r.path == p and r.accepted])
                                 for p in sorted({r.path for r in R})}

    # --- (8) pre-emption & degradation ---------------------------------------------------
    pre = [e for e in tel.events if e.kind == "preempt"]
    lost = {e.data["req"] for e in tel.events if e.kind == "victim_lost"}
    migrated = {e.data["req"] for e in mig if e.data.get("cause") == "preempted"}
    tp = defaultdict(lambda: {"preemptions": 0, "victim_migrated": 0, "victim_lost": 0})
    for e in pre:
        v = e.data["victim"]
        for key in (f"by:{e.data.get('by_priority')}", f"victim:{e.data.get('victim_priority')}"):
            tp[key]["preemptions"] += 1
            tp[key]["victim_migrated"] += v in migrated
            tp[key]["victim_lost"] += v in lost
    tables["preemption"] = dict(tp)
    S["preemptions"] = len(pre)
    S["preempt_per_100req"] = 100.0 * len(pre) / n if n else None
    S["preempt_victims_migrated"] = len({e.data["victim"] for e in pre} & migrated)
    S["preempt_victims_lost"] = len({e.data["victim"] for e in pre} & lost)
    S["preempt_victim_migrated_rate"] = _rate(S["preempt_victims_migrated"], len(pre))
    deg = [r for r in R if r.degraded]
    S["degradations"] = len(deg)
    S["degrade_per_100req"] = 100.0 * len(deg) / n if n else None
    S["degrade_mean_level"] = float(np.mean([r.degrade_level for r in deg])) if deg else None
    tables["degradation"] = {
        "by_priority": dict(Counter(r.priority_true for r in deg)),
        "by_level": {str(k): v for k, v in Counter(round(r.degrade_level, 2) for r in deg).items()}}

    # --- (9) tokens ----------------------------------------------------------------------
    for role in ("slm", "llm"):
        for io in ("in", "out"):
            S[f"tokens_{role}_{io}_per_req"] = float(np.mean([r.tokens[role][io] for r in R])) if n else None
    nonreq = [c for c in tel.llm_calls if c.req_id is None]
    S["tokens_nonrequest_total"] = sum(c.tokens_in + c.tokens_out for c in nonreq)
    all_tok = sum(c.tokens_in + c.tokens_out for c in tel.llm_calls)
    S["tokens_total"] = all_tok
    S["tokens_per_req_all"] = all_tok / n if n else None
    S["fresh_tokens_per_req"] = sum(c.tokens_in + c.tokens_out for c in tel.llm_calls
                                    if c.source == "fresh") / n if n else None

    # --- load balance ----------------------------------------------------------------------
    zv, nv = [], []
    for s in tel.snapshots:
        u = s.data.get("util_zone")
        if u:
            zv.append(float(np.var(u)))
        if "util_node_var" in s.data:
            nv.append(float(s.data["util_node_var"]))
    S["util_var_zones_mean"] = float(np.mean(zv)) if zv else None
    S["util_var_nodes_mean"] = float(np.mean(nv)) if nv else None
    pz = np.array(list(placed_zone.values()), float)
    S["placement_share_cv"] = float(pz.std() / pz.mean()) if len(pz) and pz.mean() else None

    # --- energy (approximate: allocated share of node power x time held) --------------
    nodes = meta.get("nodes") or {}
    if nodes:
        ej = 0.0
        for r in R:
            if not r.accepted or r.node_final not in nodes:
                continue
            tr = r.transitions
            t0 = next(t for t, s, note in tr if s == "PLACED")
            t1 = tr[-1][0]
            nd = nodes[r.node_final]
            ej += r.cpu_alloc / nd["cpu"] * nd["power_w"] * max(t1 - t0, 0.0)
        S["energy_kwh_alloc_share"] = ej / 3.6e6

    # --- memory/cache growth (for the call-reduction figure) ---------------------------------
    snaps = tel.snapshots
    if snaps:
        step = max(len(snaps) // 200, 1)
        series["growth"] = {"t": [s.t for s in snaps[::step]],
                            **{k: [s.data.get(k, 0) for s in snaps[::step]]
                               for k in ("cache_entries", "memory_cases", "memory_rules",
                                         "slm_fresh", "llm_fresh", "slm_cached_disk",
                                         "llm_cached_disk")}}
    return {"scalars": S, "series": series, "tables": tables}
