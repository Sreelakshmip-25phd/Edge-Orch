"""Evaluation campaign: every system (proposed, ablations, baselines) x
every seed, on the same workload files, the same simulator and the same
telemetry -> metrics.compute -> aggregated tables with 95% CIs, paired
significance tests against the full system, and all figures.

    python src/evaluator.py --profile smoke            # mock LLM, tiny, for CI
    python src/evaluator.py --profile full             # the real campaign (GPU machine)
    python src/evaluator.py --profile full --systems full,react --seeds 0,1
    python src/evaluator.py --profile full --report-only

Per run it writes results/<profile>/runs/<system>/seed<k>.telemetry.jsonl.gz
(the complete event log) and seed<k>.metrics.json. Runs are skipped when
their metrics file exists (use --force). Each system gets its own LLM disk
cache (cache/<profile>/<system>/), never shared with another system, so a
system can't ride on another's fresh calls; fresh vs cached_disk is logged
per call regardless.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DEVICE_CALIBRATION, DIGEST_S, LATENCY_CALIBRATION, LLM_CACHE_DIR,  # noqa: E402
                    LLM_MODEL_LABEL, POLICY_TICK_S, PROFILES, RESULTS_TAG, SLM_MODEL_LABEL, BASE,
                    ensure_dirs, results_dir)
import ablations  # noqa: E402
import metrics as M  # noqa: E402
import scenario_build as SB  # noqa: E402
from edge_device import DeviceFleet  # noqa: E402
from latency_model import LatencyModel, measure_embedding_samples  # noqa: E402
from llm_client import PROVIDERS, MultiLLM, discover_providers  # noqa: E402
from scenario import ServiceCatalog, load_intent_pools, seed_cache_entries  # noqa: E402
from sim_engine import EdgeSimulation, Manifests  # noqa: E402
from telemetry import Telemetry  # noqa: E402

PROPOSED = ["full"]
ABLATIONS = [a for a in ablations.DISABLES if a != "full"]
# cheapest first, so a partially finished campaign always has the comparisons
# that matter most (LATS, by far the most expensive, runs last)
BASELINES = ["greedy_oracle", "rule_based", "core", "react", "agentedge", "lats"]
ALL_SYSTEMS = PROPOSED + ABLATIONS + BASELINES
# group names accepted wherever a --systems list is (e.g. "proposed,ablations,simple")
SYSTEM_GROUPS = {"proposed": PROPOSED, "ablations": ABLATIONS,
                 "simple": ["greedy_oracle", "rule_based", "core"],
                 "agentic": ["react", "agentedge", "lats"], "all": None}


def expand_systems(spec):
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        names = ALL_SYSTEMS if tok == "all" else SYSTEM_GROUPS.get(tok, [tok])
        out += [n for n in names if n not in out]
    unknown = [n for n in out if n not in ALL_SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; choose from {ALL_SYSTEMS} "
                         f"or groups {sorted(SYSTEM_GROUPS)}")
    return [n for n in ALL_SYSTEMS if n in out]      # canonical (cheapest-first) order


GROUP = {**{s: "proposed" for s in PROPOSED}, **{s: "ablation" for s in ABLATIONS},
         **{s: "baseline" for s in BASELINES}}

# metrics reported in the main table and tested against `full`
KEY_METRICS = ["acceptance_rate", "completion_rate", "escalation_success", "escalation_rate",
               "invocations_per_req", "llm_invocations_per_req", "slm_invocations_per_req",
               "fresh_calls_per_req", "inv_per_req_first_quarter", "inv_per_req_last_quarter",
               "inv_per_req_slope_per_bin", "cache_hit_rate", "llm_stage_rate",
               "llm_per_escalation", "llm_per_escalation_first_quarter",
               "llm_per_escalation_last_quarter", "llm_per_escalation_slope_per_bin",
               "memory_share_of_escalations", "cache_miss_first_quarter",
               "cache_miss_last_quarter",
               "lat_setup_mean", "lat_setup_p95", "tokens_per_req_all",
               "translation_service_type_acc", "locality_violation_rate",
               "fail_rate_after_accept", "preempt_per_100req", "degrade_per_100req",
               "util_var_zones_mean", "cross_zone_share", "new_type_acceptance",
               "acceptance_A", "acceptance_B", "acceptance_C", "cache_hit_shared_rate",
               "completed_correct_rate"]

SEED_CACHE_PER_TYPE = 0          # cold start: the intent cache starts empty
CACHE_CAP = 500
SHADOW_RATE = 0.05


# ---------------------------------------------------------------------------
# shared, per-process context
# ---------------------------------------------------------------------------
class Context:
    """Everything a profile's runs share within one process."""

    def __init__(self, profile, need_embedder=True):
        ensure_dirs(profile)
        self.profile = profile
        self.cfg = PROFILES[profile]
        self.mock = self.cfg["llm"] == "mock"
        self.topo, self.activity = SB.topology_and_activity(profile)
        self.pools = load_intent_pools()
        self.resources = SB.resources_for(profile)
        self.embedder = None
        self.embed_samples = [0.0]
        self.sweep = None
        if need_embedder:
            from zone_agent import Embedder
            texts = [s for v in self.pools[1].values() for s in v]
            self.embedder = Embedder(texts)
            self.sweep = SB.threshold(profile, self.embedder)
            self.embed_samples = measure_embedding_samples(
                self.embedder, ["Run video analytics on the stadium cameras",
                                "Aggregate the market square sensor readings"])

    @property
    def threshold(self):
        return self.sweep["chosen"] if self.sweep else 0.65

    def warm(self, wl):
        if self.embedder is not None:
            self.embedder.warm([r["intent_text"] for r in wl["requests"]])


def _git_rev():
    try:
        return subprocess.check_output(["git", "-C", BASE, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def build_system(name, ctx, catalog, make_llm, seed):
    from baselines.agentedge_baseline import AgentEdgeBaseline
    from baselines.core_baseline import COREBaseline
    from baselines.greedy import GreedyOracle
    from baselines.lats_baseline import LATSBaseline
    from baselines.react_baseline import ReActBaseline
    from baselines.rule_based import RuleBasedHierarchical
    rng = np.random.default_rng(1000 + seed)
    if name in ablations.DISABLES:
        seed_entries = seed_cache_entries(ctx.pools[0], SEED_CACHE_PER_TYPE, rng)
        return ablations.make(name, catalog=catalog, embedder=ctx.embedder, topo=ctx.topo,
                              make_llm=make_llm, threshold=ctx.threshold, rng=rng,
                              seed_entries=seed_entries, cache_cap=CACHE_CAP,
                              shadow_rate=SHADOW_RATE, cache_sync_s=DIGEST_S)
    kw = dict(catalog=catalog, embedder=ctx.embedder, topo=ctx.topo, make_llm=make_llm,
              threshold=ctx.threshold, resources=ctx.resources, train_pool=ctx.pools[0])
    cls = {"greedy_oracle": GreedyOracle, "rule_based": RuleBasedHierarchical,
           "react": ReActBaseline, "lats": LATSBaseline, "agentedge": AgentEdgeBaseline,
           "core": COREBaseline}[name]
    return cls(**kw)


def simulate(ctx, name, wl, seed, cache_root=None):
    """One (system, workload) simulation -> Telemetry."""
    tel = Telemetry()
    lat = LatencyModel.build(ctx.topo, "mock" if ctx.mock else "real", LATENCY_CALIBRATION,
                             DEVICE_CALIBRATION, seed, SLM_MODEL_LABEL, LLM_MODEL_LABEL)
    lat.set_embedding_samples(ctx.embed_samples)
    catalog = ServiceCatalog()

    def make_llm(role, agent):
        label = SLM_MODEL_LABEL if role == "slm" else LLM_MODEL_LABEL
        label = "mock" if ctx.mock else label
        path = None if (ctx.mock or cache_root is None) else \
            os.path.join(cache_root, f"{role}_{label}.jsonl")
        return MultiLLM(path, role, telemetry=tel, agent=agent, model_label=label, latency=lat)

    orch = build_system(name, ctx, catalog, make_llm, seed)
    sim = EdgeSimulation(ctx.topo, wl, orch, tel, lat, catalog,
                         DeviceFleet.from_workload(wl, ctx.topo), Manifests(catalog, ctx.resources),
                         digest_s=DIGEST_S, policy_tick_s=POLICY_TICK_S)
    t0 = time.time()
    sim.run()
    sim.check_invariants()
    tel.meta = {
        "system": name, "group": GROUP.get(name), "seed": seed, "profile": ctx.profile,
        "llm": "mock" if ctx.mock else "real",
        "slm_model": "mock" if ctx.mock else SLM_MODEL_LABEL,
        "llm_model": "mock" if ctx.mock else LLM_MODEL_LABEL,
        **lat.describe(),
        "horizon_s": wl["meta"]["horizon_s"], "workload": wl["meta"],
        "threshold": ctx.threshold, "embedder": getattr(ctx.embedder, "backend", None),
        "seed_cache_per_type": SEED_CACHE_PER_TYPE, "cache_cap": CACHE_CAP,
        "shadow_rate": SHADOW_RATE, "disables": list(getattr(orch, "disables", ())),
        "oracle_calls": sim.oracle_calls, "probes": sim.probe_count,
        "system_stats": orch.stats() if hasattr(orch, "stats") else {},
        "llm_samples": orch.samples() if hasattr(orch, "samples") else {},
        "nodes": {n.node_id: {"cpu": n.cpu_cap, "power_w": n.power_w,
                              "device_class": n.device_class} for n in sim.nodes.values()},
        "wall_s": round(time.time() - t0, 2), "git": _git_rev(),
        "providers": [p["name"] for p in PROVIDERS]}
    return tel


def run_dir(profile, system):
    d = os.path.join(results_dir(profile), "runs", system)
    os.makedirs(d, exist_ok=True)
    return d


def run_one(ctx, name, seed, force=False):
    d = run_dir(ctx.profile, name)
    mp = os.path.join(d, f"seed{seed}.metrics.json")
    if not force and os.path.exists(mp):
        return json.load(open(mp))
    wl = SB.load_workload(ctx.profile, seed)
    ctx.warm(wl)
    # one disk cache per (system, seed): parallel shards never share a file, and a
    # seed's fresh-call count never depends on which seeds happened to run first.
    # A tagged run (model sweep) gets its own tree, like its results: otherwise a
    # sweep of label "medium" would read and append to the main run's cache files
    prof_dir = f"{ctx.profile}__{RESULTS_TAG}" if RESULTS_TAG else ctx.profile
    cache_root = os.path.join(LLM_CACHE_DIR, prof_dir, name, f"seed{seed}")
    tel = simulate(ctx, name, wl, seed, cache_root)
    tp = os.path.join(d, f"seed{seed}.telemetry.jsonl.gz")
    tel.dump(tp + ".tmp")
    os.replace(tp + ".tmp", tp)
    m = M.compute(tel)
    m["meta"] = {k: v for k, v in tel.meta.items() if k not in ("nodes", "workload")}
    with open(mp + ".tmp", "w") as f:        # atomic: a killed run never leaves a
        json.dump(m, f, indent=1, default=_json_default)   # file that looks finished
    os.replace(mp + ".tmp", mp)
    return m


def recompute_metrics(profile, systems, seeds):
    """Rebuild seed*.metrics.json from the saved event logs: what a changed
    metrics.py needs, without re-simulating or calling any model. The run
    metadata (system stats, LLM samples) is carried over from the log."""
    done = 0
    for name in systems:
        d = run_dir(profile, name)
        for k in seeds:
            tp = os.path.join(d, f"seed{k}.telemetry.jsonl.gz")
            if not os.path.exists(tp):
                continue
            tel = Telemetry.load(tp)
            m = M.compute(tel)
            m["meta"] = {kk: v for kk, v in tel.meta.items() if kk not in ("nodes", "workload")}
            mp = os.path.join(d, f"seed{k}.metrics.json")
            with open(mp + ".tmp", "w") as f:
                json.dump(m, f, indent=1, default=_json_default)
            os.replace(mp + ".tmp", mp)
            done += 1
    print(f"recomputed {done} metrics files from their event logs")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_matrix(profile, systems, seeds, force=False, shard=(0, 1)):
    """shard=(i, n): run only every n-th (system, seed) job starting at i, so n
    processes - each pointed at its own model-server pair - can share the
    campaign. Jobs are interleaved, so expensive systems spread evenly."""
    ctx = Context(profile)
    if not ctx.mock and not PROVIDERS:
        discover_providers()
    if not ctx.mock and not PROVIDERS and any(s not in ("greedy_oracle", "rule_based")
                                             for s in systems):
        raise SystemExit("No LLM provider configured (LOCAL_LLM_URL[_SLM/_LLM], GROQ_API_KEY, "
                         "...). See local_llm/README.md, or use --profile smoke.")
    if ctx.mock:
        discover_providers(mock=True)
    jobs = [(name, seed) for name in systems for seed in seeds]
    si, sn = shard
    jobs = [j for idx, j in enumerate(jobs) if idx % sn == si]
    total, k = len(jobs), 0
    tag = f"shard {si + 1}/{sn} " if sn > 1 else ""
    for name, seed in jobs:
        k += 1
        t0 = time.time()
        m = run_one(ctx, name, seed, force)
        S = m["scalars"]
        print(f"{tag}[{k}/{total}] {name:18s} seed{seed}: acc={_f(S['acceptance_rate'])} "
              f"compl={_f(S['completion_rate'])} esc_ok={_f(S['escalation_success'])} "
              f"inv/req={_f(S['invocations_per_req'])} "
              f"LLM/esc={_f(S.get('llm_per_escalation'))} "
              f"cache_miss {_f(S.get('cache_miss_first_quarter'))}->"
              f"{_f(S.get('cache_miss_last_quarter'))} ({time.time() - t0:.0f}s)", flush=True)


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def load_runs(profile, systems, seeds):
    runs = {}
    for s in systems:
        for k in seeds:
            p = os.path.join(results_dir(profile), "runs", s, f"seed{k}.metrics.json")
            if os.path.exists(p):
                runs[(s, k)] = json.load(open(p))
    return runs


def ci95(vals):
    from scipy import stats
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return None, None, 0
    if len(v) == 1:
        return float(v[0]), 0.0, 1
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h), len(v)


def paired_tests(runs, systems, seeds, metrics, ref="full"):
    """Paired over seeds (same workload file per seed): paired t-test and
    Wilcoxon signed-rank (needs >= 6 seeds to be able to reach p < 0.05),
    t-test p Holm-adjusted per metric within each family (ablations,
    baselines)."""
    from scipy import stats
    rows = []
    for m in metrics:
        block = []
        for s in systems:
            if s == ref:
                continue
            pairs = [(runs[(ref, k)]["scalars"].get(m), runs[(s, k)]["scalars"].get(m))
                     for k in seeds if (ref, k) in runs and (s, k) in runs]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None
                     and not (isinstance(a, float) and np.isnan(a))
                     and not (isinstance(b, float) and np.isnan(b))]
            if len(pairs) < 2:
                continue
            a, b = np.array(pairs).T
            d = b - a
            t_p = float(stats.ttest_rel(b, a).pvalue) if np.any(d != 0) else 1.0
            try:
                w_p = float(stats.wilcoxon(b, a).pvalue) if np.any(d != 0) and len(d) >= 5 else None
            except ValueError:
                w_p = None
            block.append({"metric": m, "system": s, "n_pairs": len(pairs),
                          "ref_mean": float(a.mean()), "sys_mean": float(b.mean()),
                          "diff_mean": float(d.mean()), "t_p": t_p, "wilcoxon_p": w_p})
        # Holm adjustment over this metric's comparisons (t-test p), within
        # each family: the ablations (Table 2) and the baselines (Table 1)
        # answer different questions, so each is corrected on its own
        for fam in {GROUP.get(b["system"], "") for b in block}:
            fb = [b for b in block if GROUP.get(b["system"], "") == fam]
            fb.sort(key=lambda b: b["t_p"])
            running = 0.0
            for rank, b in enumerate(fb):
                running = max(running, min(1.0, (len(fb) - rank) * b["t_p"]))
                b["t_p_holm"] = running
                b["holm_family"] = fam
        rows += block
    return rows


def aggregate(profile, systems, seeds):
    import pandas as pd
    runs = load_runs(profile, systems, seeds)
    present = [s for s in systems if any((s, k) in runs for k in seeds)]
    tdir = os.path.join(results_dir(profile), "tables")
    os.makedirs(tdir, exist_ok=True)
    all_keys = sorted({k for r in runs.values() for k in r["scalars"]})
    long_rows, summary = [], []
    for s in present:
        row = {"system": s, "group": GROUP.get(s, "")}
        for key in all_keys:
            vals = [runs[(s, k)]["scalars"].get(key) for k in seeds if (s, k) in runs]
            vals = [v for v in vals if isinstance(v, (int, float))]
            m, h, n = ci95(vals)
            long_rows.append({"system": s, "metric": key, "mean": m, "ci95": h, "n_seeds": n})
            if key in KEY_METRICS:
                row[key] = None if m is None else f"{m:.4g} ± {h:.2g}"
        row["n_seeds"] = sum((s, k) in runs for k in seeds)
        summary.append(row)
    pd.DataFrame(long_rows).to_csv(os.path.join(tdir, "all_metrics_long.csv"), index=False)
    df = pd.DataFrame(summary).set_index("system")
    df.to_csv(os.path.join(tdir, "summary.csv"))
    with open(os.path.join(tdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(_md_table(df.reset_index()))
    tests = paired_tests(runs, present, seeds, KEY_METRICS)
    pd.DataFrame(tests).to_csv(os.path.join(tdir, "paired_tests_vs_full.csv"), index=False)
    _write_breakdowns(runs, present, seeds, tdir)
    _write_group_comparisons(runs, present, seeds, tests, tdir)
    meta = {s: runs[next((s, k) for k in seeds if (s, k) in runs)]["meta"] for s in present}
    json.dump({"profile": profile, "systems": present, "seeds": seeds,
               "latency_source": {s: meta[s].get("latency_source") for s in present},
               "llm": {s: meta[s].get("llm") for s in present}},
              open(os.path.join(tdir, "provenance.json"), "w"), indent=1)
    return runs, present


# Quality vs cost, reported separately for ablations (what each mechanism
# contributes) and baselines (how the proposed system compares with other
# approaches). Every cell is the system's own mean over seeds, followed by its
# difference from full in the same unit (percentage points for rates).
QUALITY = [("acceptance_rate", "accepted"), ("completion_rate", "completed"),
           ("completed_correct_rate", "completed, correct type"),
           ("escalation_success", "escalation success"),
           ("acceptance_C", "accepted, last third"),
           ("locality_violation_rate", "locality violations")]
COST = [("invocations_per_req", "model calls/req", 3, ""), ("tokens_per_req_all", "tokens/req", 0, ""),
        ("lat_setup_mean", "setup latency", 0, " ms")]


def _group_table(runs, systems, seeds, tests):
    def mean(s, k):
        return ci95([runs[(s, i)]["scalars"].get(k) for i in seeds if (s, i) in runs])[0]

    pval = {(t["system"], t["metric"]): t.get("t_p_holm") for t in tests}

    def star(s, k):
        p = pval.get((s, k))
        return " *" if p is not None and p < 0.05 else ""

    import report
    rows = []
    for s in systems:
        r = {"system": s}
        for k, lab in QUALITY:
            a, b = mean("full", k), mean(s, k)
            if (s, k) in report.NOT_COMPARABLE:
                r[lab] = "n/a (not comparable)"
            elif s in report.ORACLE_TYPED and k == "completed_correct_rate":
                r[lab] = "n/a (given type)"
            elif b is None:
                r[lab] = "n/a"
            elif s == "full" or a is None:
                r[lab] = f"{100 * b:.1f}%"
            else:
                r[lab] = f"{100 * b:.1f}% ({100 * (b - a):+.1f} pp){star(s, k)}"
        for k, lab, nd, unit in COST:
            a, b = mean("full", k), mean(s, k)
            if s in report.ORACLE_TYPED and k == "lat_setup_mean":
                r[lab] = "n/a (given type)"
            elif b is None:
                r[lab] = "n/a"
            elif s == "full" or a is None:
                r[lab] = f"{b:,.{nd}f}{unit}"
            else:
                r[lab] = f"{b:,.{nd}f}{unit} ({b - a:+,.{nd}f}){star(s, k)}"
        rows.append(r)
    return rows


def _write_group_comparisons(runs, systems, seeds, tests, tdir):
    import pandas as pd
    if "full" not in systems:
        return
    head = ("Each cell: the system's mean over seeds, then in brackets its difference from "
            "full in the same unit (percentage points for rates; calls, tokens or ms for "
            "costs). Negative = lower than full (worse for quality columns, except locality "
            "violations; cheaper for cost columns). * = paired t-test vs full significant "
            "after Holm correction within this table (p < 0.05). Setup latency = request "
            "arrival to placement; service start-up is not included. n/a (given type): "
            "greedy_oracle is handed the true service type, so it has no translation to "
            "compare; n/a (not comparable): without zone agents every request counts as "
            "escalated.\n\n")
    for group, members in (("ablations", ABLATIONS), ("baselines", BASELINES)):
        sel = [s for s in members if s in systems]
        if not sel:
            continue
        df = pd.DataFrame(_group_table(runs, ["full"] + sel, seeds, tests))
        df.to_csv(os.path.join(tdir, f"compare_{group}.csv"), index=False)
        with open(os.path.join(tdir, f"compare_{group}.md"), "w", encoding="utf-8") as f:
            f.write(f"# full vs {group}\n\n" + head + _md_table(df))


def _write_breakdowns(runs, systems, seeds, tdir):
    import pandas as pd
    rows = []
    for s in systems:
        for k in seeds:
            if (s, k) not in runs:
                continue
            T = runs[(s, k)]["tables"]
            for comp, v in T.get("latency_comparisons", {}).items():
                rows.append({"system": s, "seed": k, "comparison": comp,
                             "a_mean": v["a"]["mean"], "a_p95": v["a"]["p95"], "a_n": v["a"]["n"],
                             "b_mean": v["b"]["mean"], "b_p95": v["b"]["p95"], "b_n": v["b"]["n"]})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "latency_comparisons.csv"), index=False)
    rows = []
    for s in systems:
        for k in seeds:
            if (s, k) not in runs:
                continue
            for key, v in runs[(s, k)]["tables"].get("preemption", {}).items():
                rows.append({"system": s, "seed": k, "split": key, **v})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "preemption_by_priority.csv"), index=False)
    rows = []
    for s in systems:
        for k in seeds:
            if (s, k) not in runs:
                continue
            c = runs[(s, k)]["series"]["calls_by_bin"]
            for i in range(len(c["requests"])):
                rows.append({"system": s, "seed": k, "bin": i, "t_s": c["edges_s"][i],
                             "requests": c["requests"][i],
                             "invocations_per_request": c["invocations_per_request"][i],
                             "slm_per_request": c["slm_per_request"][i],
                             "llm_per_request": c["llm_per_request"][i],
                             "cache_hit_rate": c["cache_hit_rate"][i],
                             "escalation_rate": c["escalation_rate"][i],
                             "llm_stage_rate": c["llm_stage_rate"][i],
                             "acceptance_rate": c["acceptance_rate"][i],
                             "slm_fresh": c["slm_fresh"][i], "slm_cached_disk": c["slm_cached_disk"][i],
                             "llm_fresh": c["llm_fresh"][i], "llm_cached_disk": c["llm_cached_disk"][i]})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "calls_over_time.csv"), index=False)
    rows = []
    for s in systems:
        for k in seeds:
            if (s, k) not in runs:
                continue
            T = runs[(s, k)]["tables"]
            for z, c in T.get("placements_per_zone", {}).items():
                rows.append({"system": s, "seed": k, "zone": z, "placements": c})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "placements_per_zone.csv"), index=False)
    savings_vs_ablations(runs, systems, seeds, tdir)
    _write_mean_tables(systems, tdir)


LAT_CMP = {"cache_vs_slm_translation": ("translation", "cache hit", "SLM call"),
           "memory_vs_llm_decision": ("decision", "memory", "LLM"),
           "zone_vs_cross_zone": ("setup latency", "same zone", "cross-zone")}


def _write_mean_tables(systems, tdir):
    """Readable companions of the per-seed tables: one value per system (mean
    over seeds), as <name>_mean.csv and .md. The per-seed files stay, since
    every mean can be recomputed from them."""
    import pandas as pd

    def out(df, name, title, note=""):
        df.to_csv(os.path.join(tdir, f"{name}.csv"), index=False)
        shown = df.astype(object).where(df.notna(), "n/a")     # no such path for that system
        with open(os.path.join(tdir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n" + (note + "\n\n" if note else "") + _md_table(shown))

    def load(name):
        p = os.path.join(tdir, f"{name}.csv")
        return pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 1 else None

    order = {s: i for i, s in enumerate(systems)}

    # 1. every metric: rows = metrics, columns = systems
    d = load("all_metrics_long")
    if d is not None and len(d):
        d["cell"] = [("n/a" if pd.isna(m) else (f"{m:.4g}" if pd.isna(h) or not h
                                             else f"{m:.4g} ± {h:.2g}"))
                     for m, h in zip(d["mean"], d["ci95"])]
        w = d.pivot(index="metric", columns="system", values="cell")
        w = w[[s for s in systems if s in w.columns]].fillna("n/a").reset_index()
        out(w, "all_metrics_mean", "Every metric, mean ± 95% CI over seeds",
            "One row per metric, one column per system.")

    # 2. model calls per request, hour by hour
    d = load("calls_over_time")
    if d is not None and len(d):
        d["hour"] = (d["t_s"] / 3600.0).round(2)
        num = [c for c in d.columns if c not in ("system", "seed", "bin", "t_s", "hour")]
        m = d.groupby(["system", "bin", "hour"], as_index=False)[num].mean()
        m = m.sort_values(["system", "bin"], key=lambda c: c.map(order) if c.name == "system" else c)
        m.round(4).to_csv(os.path.join(tdir, "calls_over_time_mean.csv"), index=False)
        w = m.pivot(index="hour", columns="system", values="invocations_per_request")
        w = w[[s for s in systems if s in w.columns]].round(3).reset_index()
        out(w, "calls_over_time_mean_invocations",
            "Model calls (SLM + LLM) per request, per hour of the simulated day",
            "Mean over seeds; hour = start of the bin. All columns per system and bin: "
            "calls_over_time_mean.csv.")

    # 3. placements per zone
    d = load("placements_per_zone")
    if d is not None and len(d):
        w = d.groupby(["system", "zone"])["placements"].mean().unstack("zone")
        w = w[sorted(w.columns, key=lambda z: int(str(z).lstrip("z") or 0))]
        w["total"] = w.sum(axis=1)
        w = w.loc[[s for s in systems if s in w.index]].round(0).astype(int).reset_index()
        out(w, "placements_per_zone_mean", "Services placed per zone per day",
            "Mean over seeds; a placement counts in the zone where the service ran.")

    # 4. pre-emption by priority
    d = load("preemption_by_priority")
    if d is not None and len(d):
        num = [c for c in d.columns if c not in ("system", "seed", "split")]
        m = d.groupby(["system", "split"], as_index=False)[num].mean().round(1)
        m = m.sort_values(["system", "split"], key=lambda c: c.map(order) if c.name == "system" else c)
        out(m, "preemption_by_priority_mean", "Pre-emptions per day, by priority",
            "Mean over seeds. by:<p> = pre-emptions made for a request of priority p; "
            "victim:<p> = services of priority p that were evicted, and whether they were "
            "moved elsewhere (migrated) or lost.")

    # 5. the three latency comparisons
    d = load("latency_comparisons")
    if d is not None and len(d):
        num = ["a_mean", "a_p95", "a_n", "b_mean", "b_p95", "b_n"]
        m = d.groupby(["system", "comparison"], as_index=False)[num].mean()
        rows = []
        for _, r in m.iterrows():
            what, la, lb = LAT_CMP.get(r["comparison"], (r["comparison"], "a", "b"))
            rows.append({"system": r["system"], "comparison": what,
                         "A": la, "A mean ms": round(r["a_mean"], 1), "A p95 ms": round(r["a_p95"], 1),
                         "A requests": round(r["a_n"]),
                         "B": lb, "B mean ms": round(r["b_mean"], 1), "B p95 ms": round(r["b_p95"], 1),
                         "B requests": round(r["b_n"])})
        m = pd.DataFrame(rows)
        m = m.sort_values(["system", "comparison"], key=lambda c: c.map(order) if c.name == "system" else c)
        out(m, "latency_comparisons_mean", "Latency with vs without each fast path",
            "Mean over seeds of the per-run means and 95th percentiles; requests = per run. "
            "Setup latency = request arrival to placement (service start-up not included).")

    # 6. savings vs ablations, hour by hour
    d = load("savings_vs_ablations_by_hour")
    if d is not None and len(d):
        m = d.groupby(["reference", "bin"], as_index=False)["savings"].mean()
        w = (100 * m.pivot(index="bin", columns="reference", values="savings")).round(1)
        w.columns = [f"saved vs {c} (%)" for c in w.columns]
        out(w.reset_index().rename(columns={"bin": "hour bin"}), "savings_vs_ablations_by_hour_mean",
            "Share of model calls full saves vs each ablation, per hour",
            "Mean over seeds: 100 × (1 − full's calls / the ablation's calls) in that hour.")


SAVINGS_REFS = ("no_memory", "no_intent_cache", "no_cache_sharing")


def _inv_bins(run):
    c = run["series"]["calls_by_bin"]
    return np.array(c["slm_inv"], float) + np.array(c["llm_inv"], float)


def savings_vs_ablations(runs, systems, seeds, tdir):
    """Causal evidence for the call-reduction claim: per hour, how many model
    invocations the full system saves relative to the same system with a
    mechanism removed, on the same workload (paired by seed). The raw per-
    request rate follows the daily load curve; this ratio does not."""
    import pandas as pd
    rows, summary = [], []
    if "full" not in systems:
        return
    for ref in SAVINGS_REFS:
        if ref not in systems:
            continue
        per_seed = []
        for k in seeds:
            if ("full", k) not in runs or (ref, k) not in runs:
                continue
            f, a = _inv_bins(runs[("full", k)]), _inv_bins(runs[(ref, k)])
            with np.errstate(invalid="ignore", divide="ignore"):
                sv = np.where(a > 0, 1.0 - f / a, np.nan)
            per_seed.append(sv)
            for i, v in enumerate(sv):
                rows.append({"reference": ref, "seed": k, "bin": i,
                             "full_invocations": f[i], "ref_invocations": a[i], "savings": v})
        if not per_seed:
            continue
        P = np.array(per_seed)
        q = max(P.shape[1] // 4, 1)
        first = [np.nanmean(p[:q]) for p in P]
        last = [np.nanmean(p[-q:]) for p in P]
        m1, h1, _ = ci95(first)
        m2, h2, n = ci95(last)
        summary.append({"reference": ref, "n_seeds": n,
                        "savings_first_quarter": m1, "ci95_first": h1,
                        "savings_last_quarter": m2, "ci95_last": h2,
                        "savings_overall": float(1 - sum(_inv_bins(runs[("full", k)]).sum()
                                                         for k in seeds if ("full", k) in runs
                                                         and (ref, k) in runs) /
                                                 max(sum(_inv_bins(runs[(ref, k)]).sum()
                                                         for k in seeds if ("full", k) in runs
                                                         and (ref, k) in runs), 1e-9))})
    for name, data in (("savings_vs_ablations_by_hour.csv", rows),
                       ("savings_vs_ablations.csv", summary)):
        path = os.path.join(tdir, name)
        if data:
            pd.DataFrame(data).to_csv(path, index=False)
        elif os.path.exists(path):      # no ablation in this report (e.g. a model-sweep
            os.remove(path)             # run of full alone): no stale or empty file


def _md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if v is None else str(v) for v in r.values) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
COLORS = {"proposed": "#1b9e77", "ablation": "#7570b3", "baseline": "#d95f02"}


def figures(profile, runs, systems, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = os.path.join(results_dir(profile), "figures")
    os.makedirs(fdir, exist_ok=True)
    mock = any(r["meta"].get("llm") == "mock" for r in runs.values())
    tag = "  [SMOKE: mock LLM, placeholder latency - NOT results]" if mock else ""

    def agg(s, key):
        return ci95([runs[(s, k)]["scalars"].get(key) for k in seeds if (s, k) in runs])

    def save(fig, name):
        fig.savefig(os.path.join(fdir, name), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 1. outcomes: acceptance / completion / escalation success, kept apart
    fig, ax = plt.subplots(figsize=(max(10, len(systems) * 0.9), 4.5))
    x = np.arange(len(systems))
    for j, (key, lab) in enumerate([("acceptance_rate", "accepted"),
                                    ("completion_rate", "completed"),
                                    ("escalation_success", "escalation success")]):
        ms = [agg(s, key) for s in systems]
        ax.bar(x + (j - 1) * 0.27, [m[0] or 0 for m in ms], 0.27, yerr=[m[1] or 0 for m in ms],
               capsize=2, label=lab, alpha=[1.0, 0.7, 0.45][j], color="#444", edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (mean ± 95% CI over seeds)")
    ax.set_title("Three distinct outcomes" + tag, fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig_outcomes.png")

    # 2. call reduction over time (the central claim)
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    for s in systems:
        rows = [runs[(s, k)]["series"]["calls_by_bin"] for k in seeds if (s, k) in runs]
        if not rows:
            continue
        hrs = np.array(rows[0]["edges_s"][:-1]) / 3600.0
        style = dict(color=COLORS[GROUP.get(s, "baseline")],
                     lw=2.4 if s == "full" else 1.0, alpha=1.0 if s == "full" else 0.6,
                     label=s)
        for ax, key in zip(axes, ("invocations_per_request", "cache_hit_rate", "escalation_rate",
                                  "llm_per_escalation")):
            if key not in rows[0]:
                continue
            y = np.nanmean(np.array([[np.nan if v is None else v for v in r[key]] for r in rows],
                                    float), axis=0)
            ax.plot(hrs, y, **style)
    axes[0].set_ylabel("SLM+LLM invocations per request")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("intent-cache hit rate")
    axes[2].set_ylabel("escalation rate")
    axes[3].set_ylabel("LLM invocations per escalation")
    for ax in axes:
        ax.set_xlabel("simulated hour")
    axes[0].set_title("Model invocations per request over time" + tag, fontsize=9)
    axes[3].legend(fontsize=6, ncol=2)
    save(fig, "fig_calls_over_time.png")

    # 2b. savings vs ablations per hour (causal view of the claim)
    sp = os.path.join(results_dir(profile), "tables", "savings_vs_ablations_by_hour.csv")
    if os.path.exists(sp) and os.path.getsize(sp) > 1:
        import pandas as pd
        d = pd.read_csv(sp)
        if len(d):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for ref, g in d.groupby("reference"):
                m = g.groupby("bin")["savings"].mean()
                ax.plot(m.index, m.values, marker="o", ms=3, label=f"vs {ref}")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xlabel("simulated hour")
            ax.set_ylabel("share of model invocations saved by full")
            ax.set_title("Invocations saved by memory / cache, per hour" + tag, fontsize=9)
            ax.legend(fontsize=8)
            save(fig, "fig_savings_vs_ablations.png")

    # 3. latency: the three comparisons + component breakdown
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    comps = ["transport_in", "translation", "escalation", "decision", "cross_zone"]
    bottom = np.zeros(len(systems))
    for c in comps:
        v = np.array([agg(s, f"lat_{c}_mean")[0] or 0 for s in systems])
        axes[0].bar(systems, v, bottom=bottom, label=c)
        bottom += v
    axes[0].set_ylabel("mean setup latency of accepted requests (ms)")
    axes[0].tick_params(axis="x", rotation=40, labelsize=7)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Latency components" + tag, fontsize=9)
    labels = [("lat_zone_vs_cross_zone", "setup: zone vs cross-zone"),
              ("lat_memory_vs_llm_decision", "decision: memory vs LLM"),
              ("lat_cache_vs_slm_translation", "translation: cache vs SLM")]
    ref = "full" if "full" in systems else systems[0]
    a = [agg(ref, f"{k}_a_mean")[0] or 0 for k, _ in labels]
    b = [agg(ref, f"{k}_b_mean")[0] or 0 for k, _ in labels]
    xx = np.arange(3)
    axes[1].bar(xx - 0.2, a, 0.4, label="zone / memory / cache")
    axes[1].bar(xx + 0.2, b, 0.4, label="cross-zone / LLM / SLM")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels([l for _, l in labels], fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("mean ms (log)")
    axes[1].set_title(f"The three latency comparisons ({ref})", fontsize=9)
    axes[1].legend(fontsize=8)
    save(fig, "fig_latency.png")

    # 4. Pareto: success vs latency vs tokens (LATS / AgentEdge style). Points
    # are numbered and keyed beside the axes: many systems sit close together
    # near full, and text labels on the points overlapped.
    fig, (ax, kx) = plt.subplots(1, 2, figsize=(10, 5.5), gridspec_kw={"width_ratios": [3, 1.25]})
    key = []
    for i, s in enumerate(systems, 1):
        c = agg(s, "completion_rate")[0]
        lat = agg(s, "lat_setup_mean")[0]
        tok = agg(s, "tokens_per_req_all")[0] or 0
        if c is None or lat is None:
            continue
        ax.scatter(max(lat, 1.0), 100 * c, s=30 + np.sqrt(tok) * 6,
                   color=COLORS[GROUP.get(s, "baseline")], alpha=0.75, edgecolor="white", lw=0.8)
        ax.annotate(str(i), (max(lat, 1.0), 100 * c), fontsize=8, ha="center", va="center")
        key.append(f"{i:>2}  {s}  ({tok:,.0f} tok/req)")
    ax.set_xscale("log")
    ax.set_xlabel("mean setup latency, excl. service start-up (ms, log)")
    ax.set_ylabel("requests completed (%)")
    ax.set_title("Completion vs setup latency (bubble area = tokens/request)" + tag, fontsize=9,
                 loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    kx.axis("off")
    kx.text(0, 1, "\n".join(key), va="top", ha="left", fontsize=8, family="monospace",
            transform=kx.transAxes)
    save(fig, "fig_pareto.png")

    # 5. failures + pre-emption / degradation
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    w = 0.27
    for j, (key, lab) in enumerate([("fail_node_loss", "lost to node failure"),
                                    ("fail_preempt_unmigrated", "pre-empted, not migrated"),
                                    ("interrupted_but_completed", "interrupted, still completed")]):
        axes[0].bar(x + (j - 1) * w, [agg(s, key)[0] or 0 for s in systems], w, label=lab)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems, rotation=40, ha="right", fontsize=7)
    axes[0].set_ylabel("services per run")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Execution failures by cause" + tag, fontsize=9)
    for j, (key, lab) in enumerate([("preempt_per_100req", "pre-emptions / 100 req"),
                                    ("degrade_per_100req", "degradations / 100 req")]):
        axes[1].bar(x + (j - 0.5) * 0.4, [agg(s, key)[0] or 0 for s in systems], 0.4, label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, rotation=40, ha="right", fontsize=7)
    axes[1].legend(fontsize=7)
    save(fig, "fig_preempt_degrade_failures.png")

    # 6. tokens by role and prompt/completion
    fig, ax = plt.subplots(figsize=(max(10, len(systems) * 0.9), 4.5))
    bottom = np.zeros(len(systems))
    for key, lab in [("tokens_slm_in_per_req", "SLM prompt"), ("tokens_slm_out_per_req", "SLM completion"),
                     ("tokens_llm_in_per_req", "LLM prompt"), ("tokens_llm_out_per_req", "LLM completion")]:
        v = np.array([agg(s, key)[0] or 0 for s in systems])
        ax.bar(systems, v, bottom=bottom, label=lab)
        bottom += v
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("tokens per request")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Token cost per request by role" + tag, fontsize=9)
    save(fig, "fig_tokens.png")

    # 7. load balance
    fig, ax = plt.subplots(figsize=(max(10, len(systems) * 0.9), 4))
    ax.bar(x - 0.2, [agg(s, "util_var_zones_mean")[0] or 0 for s in systems], 0.4,
           label="var. of zone utilisation")
    ax.bar(x + 0.2, [agg(s, "util_var_nodes_mean")[0] or 0 for s in systems], 0.4,
           label="var. of node utilisation")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=40, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Load balance (time-averaged utilisation variance)" + tag, fontsize=9)
    save(fig, "fig_load_balance.png")
    return fdir


def report(profile, systems, seeds):
    runs, present = aggregate(profile, systems, seeds)
    if not runs:
        print("no runs found - nothing to report")
        return
    fdir = figures(profile, runs, present, seeds)
    import report
    tests = paired_tests(runs, present, seeds, KEY_METRICS)
    main_dir = report.write_main(os.path.join(results_dir(profile), "main"), runs, present, seeds,
                                 tests, ABLATIONS, BASELINES, ablations.ISOLATES, GROUP,
                                 profile=profile)
    print(f"main results -> {main_dir}")
    print(f"appendix: tables -> {os.path.join(results_dir(profile), 'tables')}, figures -> {fdir}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--seeds", default=None, help="comma list; default = profile seeds")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--recompute-metrics", action="store_true",
                    help="recompute every seed*.metrics.json from its saved event log "
                         "(after a change to metrics.py; no simulation, no model calls), "
                         "then rebuild the report")
    ap.add_argument("--shard", default="0/1",
                    help="i/N: run every N-th job starting at i (0-based); run the report "
                         "once afterwards with --report-only (scripts/run_parallel.py does this)")
    args = ap.parse_args(argv)
    si, sn = (int(x) for x in args.shard.split("/"))
    if not 0 <= si < sn:
        raise SystemExit("--shard must be i/N with 0 <= i < N")
    systems = expand_systems(args.systems)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else PROFILES[args.profile]["seeds"]
    if args.recompute_metrics:
        recompute_metrics(args.profile, systems, seeds)
    elif not args.report_only:
        SB.build_workloads(args.profile, seeds)
        if SB.calibration(args.profile) is None:
            raise SystemExit("workload not calibrated yet - run src/calibrate_workload.py "
                             f"--profile {args.profile} (main.py does this for you)")
        run_matrix(args.profile, systems, seeds, force=args.force, shard=(si, sn))
        if sn > 1:
            return                       # report once, after all shards finish
    report(args.profile, systems, seeds)


if __name__ == "__main__":
    main()
