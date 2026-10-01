#!/usr/bin/env python3
"""Paper-ready subset of an evaluation's outputs.

Reads results/<profile>/runs/*/seed*.metrics.json (via evaluator.aggregate)
and writes results/<profile>/paper/:
  claim_call_reduction.md   the central claim, stated as numbers: per-request
                            SLM+LLM invocations in the first vs last quarter
                            of the day and the weighted slope, alongside
                            acceptance/completion, for full vs every
                            ablation/baseline (mean +- 95% CI over seeds)
  table_outcomes.md         acceptance / completion / escalation success
  table_cost.md             invocations, fresh calls, tokens, setup latency
  table_significance.md     paired tests vs full (Holm-adjusted)
  model_comparison.md       if results/model_comparison/model_comparison.csv exists
  fig_*.png                 copies of the evaluator figures
Usage: python scripts/paper_results.py --profile full
"""
import argparse
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import config  # noqa: E402
from config import PROFILES, RESULTS, results_dir  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "src"))
import evaluator as E  # noqa: E402


def cell(runs, s, seeds, key, nd=3):
    m, h, n = E.ci95([runs[(s, k)]["scalars"].get(key) for k in seeds if (s, k) in runs])
    return "n/a" if m is None else f"{m:.{nd}f} ± {h:.{nd}f}"


def table(runs, systems, seeds, cols):
    head = "| system | " + " | ".join(c for c, _ in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = "".join("| " + s + " | " + " | ".join(cell(runs, s, seeds, k) for _, k in cols) + " |\n"
                   for s in systems)
    return head + rows


def _savings_md(profile):
    import pandas as pd
    p = os.path.join(results_dir(profile), "tables", "savings_vs_ablations.csv")
    if not os.path.exists(p):
        return ""
    d = pd.read_csv(p)
    if not len(d):
        return ""
    return ("\n## Invocations saved by the full system vs ablations (paired by seed)\n\n"
            + E._md_table(d.round(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else PROFILES[a.profile]["seeds"]
    runs, systems = E.aggregate(a.profile, E.ALL_SYSTEMS, seeds)
    if not runs:
        raise SystemExit("no runs found")
    out = os.path.join(results_dir(a.profile), "paper")
    os.makedirs(out, exist_ok=True)
    mock = any(r["meta"].get("llm") == "mock" for r in runs.values())
    banner = ("> **SMOKE RUN - mock LLM and placeholder latencies. These numbers only show "
              "the pipeline works; they are not results.**\n\n") if mock else ""
    open(os.path.join(out, "claim_call_reduction.md"), "w").write(
        banner + "# Does model usage fall over time without hurting acceptance?\n\n" +
        table(runs, systems, seeds, [
            ("inv/req first quarter", "inv_per_req_first_quarter"),
            ("inv/req last quarter", "inv_per_req_last_quarter"),
            ("slope per bin", "inv_per_req_slope_per_bin"),
            ("LLM/req first q.", "llm_per_req_first_quarter"),
            ("LLM/req last q.", "llm_per_req_last_quarter"),
            ("cache hit first q.", "cache_hit_first_quarter"),
            ("cache hit last q.", "cache_hit_last_quarter"),
            ("acceptance", "acceptance_rate"), ("completion", "completion_rate")]) +
        "\n## Normalised by need (removes the daily load curve)\n\n" +
        table(runs, systems, seeds, [
            ("LLM/escalation first q.", "llm_per_escalation_first_quarter"),
            ("LLM/escalation last q.", "llm_per_escalation_last_quarter"),
            ("LLM/escalation slope", "llm_per_escalation_slope_per_bin"),
            ("memory share of escalations", "memory_share_of_escalations"),
            ("cache miss first q.", "cache_miss_first_quarter"),
            ("cache miss last q.", "cache_miss_last_quarter"),
            ("escalation rate", "escalation_rate")]) +
        _savings_md(a.profile))
    open(os.path.join(out, "table_outcomes.md"), "w").write(banner + table(runs, systems, seeds, [
        ("accepted", "acceptance_rate"), ("completed", "completion_rate"),
        ("escalation rate", "escalation_rate"), ("escalation success", "escalation_success"),
        ("A", "acceptance_A"), ("B", "acceptance_B"), ("C", "acceptance_C"),
        ("new-type acceptance", "new_type_acceptance"),
        ("locality violations", "locality_violation_rate"),
        ("lost after accept", "fail_rate_after_accept")]))
    open(os.path.join(out, "table_cost.md"), "w").write(banner + table(runs, systems, seeds, [
        ("invocations/req", "invocations_per_req"), ("LLM inv/req", "llm_invocations_per_req"),
        ("SLM inv/req", "slm_invocations_per_req"), ("fresh calls/req", "fresh_calls_per_req"),
        ("tokens/req", "tokens_per_req_all"), ("setup latency ms", "lat_setup_mean"),
        ("total latency p95 ms", "lat_total_p95")]))
    import pandas as pd
    t = pd.read_csv(os.path.join(results_dir(a.profile), "tables", "paired_tests_vs_full.csv"))
    t = t[t.metric.isin(["acceptance_rate", "completion_rate", "invocations_per_req",
                         "llm_invocations_per_req", "tokens_per_req_all", "lat_setup_mean"])]
    open(os.path.join(out, "table_significance.md"), "w").write(banner + E._md_table(
        t[["metric", "system", "n_pairs", "ref_mean", "sys_mean", "diff_mean", "t_p",
           "t_p_holm", "wilcoxon_p"]].round(4)))
    mc = os.path.join(RESULTS, "model_comparison", "model_comparison.csv")
    if os.path.exists(mc):
        df = pd.read_csv(mc)
        cols = [c for c in ("display_name", "params_b", "hosted_reference", "translation_accuracy",
                            "full_schema_exact_match", "reasoning_acceptable", "reasoning_preferred",
                            "translate_ms_mean", "translate_ms_p95", "decide_ms_mean",
                            "decide_ms_p95", "tokens_in_per_call", "tokens_out_per_call")
                if c in df]
        open(os.path.join(out, "model_comparison.md"), "w").write(
            E._md_table(df.sort_values("params_b")[cols]))
    for g in ("ablations", "baselines"):     # kept separate: mechanism value vs other approaches
        src = os.path.join(results_dir(a.profile), "tables", f"compare_{g}.md")
        if os.path.exists(src):
            open(os.path.join(out, f"compare_{g}.md"), "w").write(banner + open(src).read())
    fdir = os.path.join(results_dir(a.profile), "figures")
    for f in os.listdir(fdir) if os.path.isdir(fdir) else []:
        shutil.copy(os.path.join(fdir, f), os.path.join(out, f))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
