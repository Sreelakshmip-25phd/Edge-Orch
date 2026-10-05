"""The main results: two tables and four figures, nothing else.

Written to results/<profile>/main/ by `evaluator.py --report-only` (and by
every full evaluation). Everything else the evaluator writes (tables/,
figures/) is the appendix.

  table_baselines.md   full vs the 6 baselines, absolute values
  table_ablations.md   what each mechanism contributes: change vs full
  fig1_calls_over_time    model calls per request over the day: full vs the
                          ablations that remove a call-saving mechanism
  fig2_quality_vs_cost    acceptance vs model calls per request, full vs baselines
  fig3_acceptance_by_phase  acceptance per third of the day and on the two
                          service types that appear mid-run
  fig4_setup_latency      setup latency split into its components

Conventions: mean ± 95% CI over seeds; "*" = paired t-test vs full
significant after Holm correction (p < 0.05); "n/a" = the metric does not
apply to that system (e.g. greedy_oracle is handed the true service type, so
it has no translation accuracy or translation latency to compare).
Deployment (container start-up) time is NOT part of setup latency: it is an
assumed per-device constant, not a measurement, and is reported only in the
appendix tables.
"""
import os

import numpy as np

# reference palette, light mode, categorical slots in fixed order
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUP_COLOR = {"proposed": SLOT[0], "ablation": SLOT[1], "baseline": SLOT[2]}

LAT_PARTS = [("lat_translation_mean", "translation"), ("lat_decision_mean", "decision"),
             (("lat_transport_in_mean", "lat_escalation_mean", "lat_cross_zone_mean"), "network")]
# systems whose translation is handed to them (no translation to measure)
ORACLE_TYPED = {"greedy_oracle"}


def _ci(runs, s, seeds, key):
    from evaluator import ci95
    keys = key if isinstance(key, tuple) else (key,)
    vals = []
    for k in seeds:
        if (s, k) not in runs:
            continue
        sc = runs[(s, k)]["scalars"]
        v = [sc.get(x) for x in keys]
        vals.append(None if any(x is None for x in v) else float(sum(v)))
    return ci95(vals)


def _p(tests, s, key):
    return next((t.get("t_p_holm") for t in tests if t["system"] == s and t["metric"] == key), None)


def _star(tests, s, key):
    p = _p(tests, s, key)
    return " *" if p is not None and p < 0.05 else ""


def _fmt(m, h, scale=1.0, nd=1):
    if m is None:
        return "n/a"
    return f"{m * scale:.{nd}f} ± {h * scale:.{nd}f}" if h else f"{m * scale:.{nd}f}"


def _md(rows, cols):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return out + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)


# --- tables -------------------------------------------------------------------------
BASELINE_COLS = [  # (label, metric, scale, decimals)
    ("accepted %", "acceptance_rate", 100, 1),
    ("new types accepted %", "new_type_acceptance", 100, 1),
    ("locality violations %", "locality_violation_rate", 100, 1),
    ("service type correct %", "translation_service_type_acc", 100, 1),
    ("model calls / req", "invocations_per_req", 1, 2),
    ("tokens / req", "tokens_per_req_all", 1, 0),
    ("setup latency ms", "lat_setup_mean", 1, 0),
]


def table_baselines(runs, seeds, tests, baselines):
    rows = []
    for s in ["full"] + baselines:
        if not any((s, k) in runs for k in seeds):
            continue
        r = {"system": "**full (proposed)**" if s == "full" else s}
        for lab, key, sc, nd in BASELINE_COLS:
            if s in ORACLE_TYPED and key in ("translation_service_type_acc", "lat_setup_mean"):
                r[lab] = "n/a (given)"
                continue
            m, h, _ = _ci(runs, s, seeds, key)
            r[lab] = _fmt(m, h, sc, nd) + ("" if s == "full" else _star(tests, s, key))
        rows.append(r)
    return rows, ["system"] + [c[0] for c in BASELINE_COLS]


ABLATION_COLS = [  # (label, metric, kind) kind: pp = difference, x = ratio
    ("accepted", "acceptance_rate", "pp"),
    ("escalation success", "escalation_success", "pp"),
    ("model calls / req", "invocations_per_req", "x"),
    ("SLM calls / req", "slm_invocations_per_req", "x"),
    ("LLM calls / req", "llm_invocations_per_req", "x"),
    ("calls / req, last quarter of day", "inv_per_req_last_quarter", "x"),
    ("setup latency", "lat_setup_mean", "x"),
]


def table_ablations(runs, seeds, tests, ablations, isolates):
    full = {key: _ci(runs, "full", seeds, key)[0] for _, key, _ in ABLATION_COLS}
    ref = {"system": "**full (proposed)**", "removes": "-"}
    for lab, key, kind in ABLATION_COLS:
        v = full[key]
        ref[lab] = "n/a" if v is None else (f"{100 * v:.1f} %" if kind == "pp" else
                                            f"{v:.0f} ms" if key == "lat_setup_mean" else f"{v:.2f}")
    rows = [ref]
    for s in ablations:
        if not any((s, k) in runs for k in seeds):
            continue
        r = {"system": s, "removes": isolates.get(s, "")}
        for lab, key, kind in ABLATION_COLS:
            m = _ci(runs, s, seeds, key)[0]
            if m is None or full[key] is None:
                r[lab] = "n/a"
            elif kind == "pp":
                r[lab] = f"{100 * (m - full[key]):+.1f} pp" + _star(tests, s, key)
            else:
                r[lab] = ("n/a" if not full[key] else f"{m / full[key]:.2f}x") + _star(tests, s, key)
        rows.append(r)
    return rows, ["system", "removes"] + [c[0] for c in ABLATION_COLS]


# --- figures ------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_calls_over_time(plt, runs, seeds, systems, path, tag):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, s in enumerate(systems):
        rows = [runs[(s, k)]["series"]["calls_by_bin"] for k in seeds if (s, k) in runs]
        if not rows:
            continue
        hrs = (np.array(rows[0]["edges_s"][:-1]) + np.diff(rows[0]["edges_s"]) / 2) / 3600.0
        Y = np.array([[np.nan if v is None else v for v in r["invocations_per_request"]]
                      for r in rows], float)
        y = np.nanmean(Y, axis=0)
        c = SLOT[i]
        if len(rows) > 1:
            lo, hi = np.nanpercentile(Y, 2.5, axis=0), np.nanpercentile(Y, 97.5, axis=0)
            ax.fill_between(hrs, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(hrs, y, color=c, lw=2.4 if s == "full" else 1.6, label=s,
                marker="o", ms=3.5, markeredgecolor="white", markeredgewidth=0.6)
        last = np.where(~np.isnan(y))[0]
        if len(last):
            j = last[-1]
            ax.annotate(s, (hrs[j], y[j]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
    _style(ax)
    ax.set_xlabel("simulated hour of the day", color=INK2, fontsize=9)
    ax.set_ylabel("model calls per request (SLM + LLM)", color=INK2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title("Model calls per request over the day" + tag, fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_quality_vs_cost(plt, runs, seeds, systems, group, path, tag):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    seen = set()
    for s in systems:
        mx, hx, _ = _ci(runs, s, seeds, "invocations_per_req")
        my, hy, _ = _ci(runs, s, seeds, "acceptance_rate")
        if mx is None or my is None:
            continue
        g = group.get(s, "baseline")
        x = max(mx, 0.01)                      # rule_based / greedy make no calls: shown at 0.01
        ax.errorbar(x, 100 * my, yerr=100 * (hy or 0), fmt="o", ms=8 if s == "full" else 6,
                    color=GROUP_COLOR[g], mec="white", mew=1.0, ecolor=GROUP_COLOR[g], elinewidth=1,
                    capsize=0, label=None if g in seen else
                    {"proposed": "proposed", "ablation": "ablation", "baseline": "baseline"}[g],
                    zorder=3)
        seen.add(g)
        ax.annotate(s + (" (0 calls)" if mx == 0 else ""), (x, 100 * my), xytext=(6, 3),
                    textcoords="offset points", fontsize=7.5, color=INK)
    _style(ax)
    ax.set_xscale("log")
    ax.set_xlabel("model calls per request (log scale; lower = cheaper)", color=INK2, fontsize=9)
    ax.set_ylabel("requests accepted (%)", color=INK2, fontsize=9)
    ax.set_title("Quality vs cost: up and to the left is better" + tag, fontsize=10, color=INK,
                 loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_acceptance_by_phase(plt, runs, seeds, systems, path, tag):
    parts = [("acceptance_A", "first third of day"), ("acceptance_B", "second third"),
             ("acceptance_C", "last third (surge)"), ("new_type_acceptance", "new service types")]
    systems = [s for s in systems if any((s, k) in runs for k in seeds)]
    fig, ax = plt.subplots(figsize=(max(7.2, 1.0 * len(systems) + 1.5), 3.8))
    x = np.arange(len(systems))
    w = 0.8 / len(parts)
    for j, (key, lab) in enumerate(parts):
        ms = [_ci(runs, s, seeds, key) for s in systems]
        ax.bar(x + (j - (len(parts) - 1) / 2) * w, [100 * (m[0] or 0) for m in ms], w * 0.9,
               yerr=[100 * (m[1] or 0) for m in ms], color=SLOT[j], label=lab,
               error_kw=dict(elinewidth=0.8, ecolor=INK2, capsize=0))
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=25, ha="right", fontsize=8, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("requests accepted (%)", color=INK2, fontsize=9)
    ax.set_title("Acceptance as the workload drifts" + tag, fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_setup_latency(plt, runs, seeds, systems, path, tag):
    systems = [s for s in systems if any((s, k) in runs for k in seeds)
               and s not in ORACLE_TYPED]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(systems) + 1.4))
    y = np.arange(len(systems))[::-1]
    left = np.zeros(len(systems))
    for j, (key, lab) in enumerate(LAT_PARTS):
        v = np.array([_ci(runs, s, seeds, key)[0] or 0.0 for s in systems])
        ax.barh(y, v, 0.6, left=left, color=SLOT[j], label=lab, edgecolor="white", linewidth=1)
        left += v
    for yi, tot in zip(y, left):
        ax.annotate(f"{tot:.0f} ms", (tot, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y)
    ax.set_yticklabels(systems, fontsize=8, color=INK)
    ax.set_xlabel("mean setup latency per request (ms; excludes service start-up)", color=INK2,
                  fontsize=9)
    ax.set_title("Where setup time goes" + tag, fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- entry point --------------------------------------------------------------------
def write_main(out, runs, present, seeds, tests, ablations, baselines, isolates, group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    mock = any(r["meta"].get("llm") == "mock" for r in runs.values())
    tag = "  [SMOKE: mock LLM - not results]" if mock else ""
    banner = ("> **SMOKE RUN - mock LLM and placeholder latencies. These numbers only show the "
              "pipeline works; they are not results.**\n\n") if mock else ""
    n = {s: sum((s, k) in runs for k in seeds) for s in present}
    note = ("Mean ± 95% CI over seeds (seeds per system: "
            + ", ".join(f"{s} {n[s]}" for s in present) + "). "
            "* = differs from full, paired t-test, Holm-corrected p < 0.05.\n\n")
    if "full" in present:
        rows, cols = table_baselines(runs, seeds, tests, [b for b in baselines if b in present])
        open(os.path.join(out, "table_baselines.md"), "w").write(
            banner + "# Table 1 - full vs baselines\n\n" + note +
            "greedy_oracle is handed the true service type: its translation columns are n/a. "
            "Setup latency = transport + translation + decision + network hops; service "
            "start-up time is excluded (an assumption, not a measurement).\n\n" + _md(rows, cols))
        rows, cols = table_ablations(runs, seeds, tests, [a for a in ablations if a in present],
                                     isolates)
        open(os.path.join(out, "table_ablations.md"), "w").write(
            banner + "# Table 2 - what each mechanism contributes\n\n" + note +
            "First row: the full system's absolute values. Other rows: quality as the "
            "difference from full in percentage points (negative = worse); cost as a ratio to "
            "full (above 1.00x = the ablation needs more).\n\n" + _md(rows, cols))
    call_refs = [s for s in ("full", "no_intent_cache", "no_memory", "no_cache_sharing")
                 if s in present]
    fig_calls_over_time(plt, runs, seeds, call_refs, os.path.join(out, "fig1_calls_over_time.png"),
                        tag)
    fig_quality_vs_cost(plt, runs, seeds, [s for s in ["full"] + baselines if s in present], group,
                        os.path.join(out, "fig2_quality_vs_cost.png"), tag)
    fig_acceptance_by_phase(plt, runs, seeds, [s for s in ["full"] + baselines if s in present],
                            os.path.join(out, "fig3_acceptance_by_phase.png"), tag)
    fig_setup_latency(plt, runs, seeds, [s for s in ["full"] + baselines if s in present],
                      os.path.join(out, "fig4_setup_latency.png"), tag)
    return out
