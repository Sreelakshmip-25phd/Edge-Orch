#!/usr/bin/env python3
"""Paper-ready results: results/<profile>/main/ (2 tables + 4 figures, see
src/report.py) and, if it exists, the model comparison table.

Everything else (results/<profile>/tables, figures) is the appendix.
Usage: python scripts/paper_results.py --profile full
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from config import PROFILES, RESULTS, results_dir  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "src"))
import evaluator as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else PROFILES[a.profile]["seeds"]
    runs, systems = E.aggregate(a.profile, E.ALL_SYSTEMS, seeds)
    if not runs:
        raise SystemExit("no runs found")
    import pandas as pd
    import report
    import ablations
    tests = E.paired_tests(runs, systems, seeds, E.KEY_METRICS)
    out = report.write_main(os.path.join(results_dir(a.profile), "main"), runs, systems, seeds,
                            tests, E.ABLATIONS, E.BASELINES, ablations.ISOLATES, E.GROUP)
    mc = os.path.join(RESULTS, "model_comparison", "model_comparison.csv")
    if os.path.exists(mc):
        df = pd.read_csv(mc)
        cols = [c for c in ("display_name", "params_b", "hosted_reference", "translation_accuracy",
                            "full_schema_exact_match", "reasoning_valid_json",
                            "reasoning_acceptable", "reasoning_preferred",
                            "translate_ms_mean", "translate_ms_p95", "decide_ms_mean",
                            "decide_ms_p95", "tokens_in_per_call", "tokens_out_per_call")
                if c in df]
        open(os.path.join(out, "table_models.md"), "w").write(
            "# Table 3 - local models compared\n\n" + E._md_table(df.sort_values("params_b")[cols]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
