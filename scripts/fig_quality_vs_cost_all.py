"""Quality vs cost for the proposed system and every baseline on one profile.

Reads <runs>/<system>/seed*.metrics.json (the quick profile is the only one
on which all seven systems ran) and draws acceptance against model calls per
request. Systems that make no model calls cannot sit on a log axis, so they
get their own narrow panel instead of an invented x position. Each system has
two markers: filled = accepted, hollow = locality-valid acceptance (accepted
and not a data-locality violation).

    python scripts/fig_quality_vs_cost_all.py --runs results/quick/runs \
        --out results/quick/main/fig_quality_vs_cost_all.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluator import ci95                      # noqa: E402
from report import SLOT, INK, INK2, GRID        # noqa: E402

SYSTEMS = [  # (dir name, label, group)
    ("full", "full (proposed)", "proposed"),
    ("core", "CORE-inspired", "baseline"),
    ("greedy_oracle", "greedy_oracle", "baseline"),
    ("rule_based", "rule_based", "baseline"),
    ("react", "ReAct", "agentic"),
    ("agentedge", "AgentEdge-inspired", "agentic"),
    ("lats", "LATS (1 seed)", "agentic"),
]
COLOR = {"proposed": SLOT[0], "baseline": SLOT[2], "agentic": SLOT[1]}
LEGEND = {"proposed": "proposed", "baseline": "baselines",
          "agentic": "LLM-agent orchestrators"}


def load(runs_dir, system):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs_dir, system, "seed*.metrics.json"))):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f)["scalars"])
    return rows


def stats(rows):
    acc = [r["acceptance_rate"] for r in rows]
    valid = [r["acceptance_rate"] * (1 - (r.get("locality_violation_rate") or 0.0)) for r in rows]
    calls = [r["invocations_per_req"] for r in rows]
    return ci95(acc), ci95(valid), ci95(calls)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/quick/runs")
    ap.add_argument("--out", default="results/quick/main/fig_quality_vs_cost_all.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True,
                                   gridspec_kw={"width_ratios": [2.2, 6], "wspace": 0.05})
    zero_x = 0
    seen = set()
    for name, label, group in SYSTEMS:
        rows = load(a.runs, name)
        if not rows:
            print(f"skip {name}: no runs in {a.runs}")
            continue
        (acc, acc_h, n), (val, val_h, _), (calls, _, _) = stats(rows)
        c = COLOR[group]
        ax, x = (ax0, zero_x) if calls == 0 else (ax1, calls)
        if calls == 0:
            zero_x += 1
        ax.plot([x, x], [100 * val, 100 * acc], color=c, lw=0.8, ls=":", zorder=2)
        ax.errorbar(x, 100 * acc, yerr=100 * acc_h, fmt="o", ms=7 if name == "full" else 6,
                    color=c, mec="white", mew=0.8, ecolor=c, elinewidth=1, capsize=0, zorder=4,
                    label=None if group in seen else LEGEND[group])
        seen.add(group)
        # ring larger than the dot and drawn underneath it: where accepted and
        # locality-valid coincide (no violations) both stay visible
        ax.errorbar(x, 100 * val, yerr=100 * val_h, fmt="o", ms=12, mfc="white", mec=c, mew=1.4,
                    ecolor=c, elinewidth=0.8, capsize=0, zorder=3)
        print(f"{name:14s} n={n} calls/req={calls:.2f} accepted={100*acc:.1f}% "
              f"locality-valid={100*val:.1f}%")
        dx, dy, ha = {"react": (-8, 3, "right"), "agentedge": (7, -14, "left")}.get(
            name, (7, 3, "left"))
        ax.annotate(label, (x, 100 * acc), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color=INK, ha=ha)

    for ax in (ax0, ax1):
        style(ax)
    ax0.set_xlim(-0.4, max(zero_x, 1) + 0.9)
    ax0.set_xticks([])
    ax0.set_xlabel("no model\ncalls", color=INK2, fontsize=9)
    ax0.set_ylabel("acceptance (%)", color=INK2, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlim(0.1, 50)
    ax1.set_xlabel("model calls per request (log scale)", color=INK2, fontsize=9)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    h, l = ax1.get_legend_handles_labels()
    h0, l0 = ax0.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    marks = [Line2D([], [], marker="o", ls="", color=INK2, mec="white", ms=6),
             Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.4, ms=12)]
    by = dict(zip(l0 + l, h0 + h))
    fig.legend(list(by.values()) + marks,
               list(by.keys()) + ["accepted", "locality-valid acceptance"],
               fontsize=8, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
