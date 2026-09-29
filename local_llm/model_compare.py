#!/usr/bin/env python3
"""Phase 9 - multi-model comparison (runs on the GPU machine).

For every --target it measures, with the exact prompts the pipeline uses:
  SLM role   translation accuracy (per field) + full-schema exact match on
             the held-out phrasings (OOD probe + LLM paraphrases + the new
             types' templates), via zone_agent.slm_system_prompt()
  LLM role   the 100 auto-scored structured-decision cases
             (reasoning_cases.py, DECIDE_SYS): valid-JSON rate, acceptable
             rate (overall and per kind), preferred rate
  both       latency mean / p95 (measured wall time of fresh calls) and
             tokens (prompt / completion) per call
It never starts servers - start them first (start_server.py), then:

    python local_llm/model_compare.py \\
        --target llama32_1b@http://127.0.0.1:8080 \\
        --target medium@http://127.0.0.1:8081 \\
        --target hosted_ref_llama70b@hosted          # needs GROQ_API_KEY

Each label@url must be a local_llm/models.py tier key (or a config.py
HOSTED_REFERENCE key with url "hosted"); any other label is allowed but
won't get size/family metadata. Before scoring, the script asks each server
which model file it is serving (/v1/models) and refuses to score two labels
that are served by the same model file - the old comparison table had
identical rows for llama32_1b and medium, consistent with both labels
having been pointed at one server.

Output (merged across invocations, so you can run a few models at a time):
    results/model_comparison/model_comparison.csv
    results/model_comparison/model_comparison_cases.jsonl
    results/model_comparison/model_comparison_figure.png
Hosted rows are labelled "hosted reference, not a deployment candidate".
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)
import config  # noqa: E402,F401
from config import HOSTED_REFERENCE, RESULTS  # noqa: E402

sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, HERE)
from global_agent import DECIDE_SYS  # noqa: E402
from llm_client import MultiLLM  # noqa: E402
from models import MODELS  # noqa: E402
from reasoning_cases import generate as gen_cases, score as score_case  # noqa: E402
from scenario import (CORE_TYPES, PLACES_INITIAL, SERVICE_TYPES, ServiceCatalog,  # noqa: E402
                      ground_truth_profile, load_intent_pools)
from zone_agent import slm_system_prompt  # noqa: E402

OUT = os.path.join(RESULTS, "model_comparison")
FIELDS = ["service_type", "latency_class", "data_locality", "priority"]


def translation_set():
    """Held-out phrasings only (never the training seeds)."""
    _, held, fixed = load_intent_pools()
    items = []
    for st in SERVICE_TYPES:
        for i, tpl in enumerate(held[st]):
            items.append((st, tpl.format(place=PLACES_INITIAL[i % len(PLACES_INITIAL)])))
        items += [(st, s) for s in fixed[st]]
    return items


def served_model(url):
    import requests
    try:
        j = requests.get(url.rstrip("/") + "/v1/models", timeout=10).json()
        return ",".join(sorted(str(m.get("id")) for m in j.get("data", []))) or "unknown"
    except Exception as e:
        return f"unreachable ({e.__class__.__name__})"


def provider_for(label, url):
    if url == "hosted":
        ref = HOSTED_REFERENCE[label]
        key = os.environ.get(ref["provider_env"], "")
        if not key:
            raise SystemExit(f"{label}: set {ref['provider_env']}")
        return {"name": label, "key": key, "url": ref["url"], "models": [ref["model"]],
                "roles": None}
    return {"name": label, "key": "local", "local": True, "roles": None,
            "no_system_role": MODELS.get(label, {}).get("no_system_role", False),
            "url": url.rstrip("/") + "/v1/chat/completions", "models": ["local"]}


def run_target(label, url, cases, items, limit=None):
    prov = provider_for(label, url)
    cache = os.path.join(OUT, "cache", f"{label}.jsonl")
    slm = MultiLLM(cache, "slm", model_label=label, providers=[prov])
    llm = MultiLLM(cache, "llm", model_label=label, providers=[prov])
    lat = {"translate": [], "decide": []}
    tok = {"in": 0, "out": 0, "n": 0}
    per_case = []

    # --- translation (catalog incl. the new types, as after registration) ---
    cat = ServiceCatalog()
    for st in SERVICE_TYPES:
        if st not in CORE_TYPES:
            cat.register(st, SERVICE_TYPES[st], 0.0)
    sys_prompt = slm_system_prompt(cat)
    field_ok = {f: 0 for f in FIELDS}
    exact = n = 0
    for st, text in items[:limit]:
        exp = ground_truth_profile(st)
        try:
            r = slm.ask(sys_prompt, "INPUT: " + text, kind="translate")
            out = r.data
            lat["translate"].append(r.wall_ms)
            tok["in"] += r.tokens_in
            tok["out"] += r.tokens_out
            tok["n"] += 1
        except Exception as e:
            out = {"_error": str(e)}
        n += 1
        ok = [out.get(f) == exp[f] for f in FIELDS]
        for f, o in zip(FIELDS, ok):
            field_ok[f] += o
        exact += all(ok)
        per_case.append({"model": label, "task": "translate", "type": st, "text": text,
                         "answer": out, "exact": all(ok)})

    # --- reasoning ---------------------------------------------------------------
    agg = {"valid_json": 0, "acceptable": 0, "preferred": 0}
    by_kind = {}
    for c in cases[:limit]:
        try:
            r = llm.ask(DECIDE_SYS, c["user"], kind="decide")
            ans = r.data
            lat["decide"].append(r.wall_ms)
            tok["in"] += r.tokens_in
            tok["out"] += r.tokens_out
            tok["n"] += 1
        except Exception as e:
            ans = {"_error": str(e)}
        s = score_case(c, ans)
        for k in agg:
            agg[k] += s[k]
        bk = by_kind.setdefault(c["kind"], [0, 0])
        bk[0] += s["acceptable"]
        bk[1] += 1
        per_case.append({"model": label, "task": "decide", "case": c["name"], "kind": c["kind"],
                         "answer": ans, **s})

    import numpy as np

    def st_(xs):
        return (round(float(np.mean(xs)), 1), round(float(np.percentile(xs, 95)), 1)) if xs else (None, None)
    nc = max(len(cases[:limit]), 1)
    row = {"model": label,
           "display_name": MODELS.get(label, {}).get("display_name")
           or HOSTED_REFERENCE.get(label, {}).get("display_name", label),
           "params_b": MODELS.get(label, {}).get("params_b")
           or HOSTED_REFERENCE.get(label, {}).get("params_b"),
           "role_hint": MODELS.get(label, {}).get("role", "reference"),
           "hosted_reference": url == "hosted",
           "note": "hosted reference, not a deployment candidate" if url == "hosted" else "",
           "served_model": "hosted" if url == "hosted" else served_model(url),
           "n_translation": n,
           "translation_accuracy": round(sum(field_ok.values()) / (n * len(FIELDS)), 4) if n else None,
           "full_schema_exact_match": round(exact / n, 4) if n else None,
           **{f"acc_{f}": round(v / n, 4) if n else None for f, v in field_ok.items()},
           "n_reasoning": nc,
           "reasoning_valid_json": round(agg["valid_json"] / nc, 4),
           "reasoning_acceptable": round(agg["acceptable"] / nc, 4),
           "reasoning_preferred": round(agg["preferred"] / nc, 4),
           **{f"reasoning_acc_{k}": round(a / t, 4) for k, (a, t) in by_kind.items()},
           "translate_ms_mean": st_(lat["translate"])[0], "translate_ms_p95": st_(lat["translate"])[1],
           "decide_ms_mean": st_(lat["decide"])[0], "decide_ms_p95": st_(lat["decide"])[1],
           "tokens_in_per_call": round(tok["in"] / tok["n"], 1) if tok["n"] else None,
           "tokens_out_per_call": round(tok["out"] / tok["n"], 1) if tok["n"] else None,
           "fresh_calls": slm.fresh + llm.fresh, "cached_disk_calls": slm.cached_disk + llm.cached_disk}
    return row, per_case


def make_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    df = df.sort_values("params_b", na_position="last")
    lab = [f"{d}{' *' if h else ''}" for d, h in zip(df.display_name, df.hosted_reference)]
    x = np.arange(len(df))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for j, (col, name) in enumerate([("translation_accuracy", "translation (per field)"),
                                     ("full_schema_exact_match", "translation exact match"),
                                     ("reasoning_acceptable", "decision acceptable"),
                                     ("reasoning_preferred", "decision preferred")]):
        ax[0].bar(x + (j - 1.5) * 0.2, df[col].fillna(0), 0.2, label=name)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=7)
    ax[0].set_title("Quality (* = hosted reference, not a deployment candidate)", fontsize=9)
    ax[1].bar(x - 0.2, df.translate_ms_mean.fillna(0), 0.4, label="translate mean",
              yerr=(df.translate_ms_p95 - df.translate_ms_mean).clip(lower=0).fillna(0))
    ax[1].bar(x + 0.2, df.decide_ms_mean.fillna(0), 0.4, label="decide mean",
              yerr=(df.decide_ms_p95 - df.decide_ms_mean).clip(lower=0).fillna(0))
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("ms (error bar to p95)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=[], help="label@url (repeatable)")
    ap.add_argument("--fresh", action="store_true", help="start the CSV over")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N items per task")
    ap.add_argument("--allow-duplicate-server", action="store_true")
    a = ap.parse_args(argv)
    if not a.target:
        raise SystemExit("pass at least one --target label@url")
    os.makedirs(OUT, exist_ok=True)
    cases, items = gen_cases(), translation_set()
    print(f"{len(items)} held-out translation items, {len(cases)} reasoning cases")
    seen = {}
    for spec in a.target:
        label, url = spec.split("@", 1)
        if url != "hosted":
            sm = served_model(url)
            if sm in seen and not a.allow_duplicate_server:
                raise SystemExit(f"{label} and {seen[sm]} are served by the same model ({sm}) - "
                                 "each label must point at its own server")
            seen[sm] = label
    import pandas as pd
    csv = os.path.join(OUT, "model_comparison.csv")
    rows = [] if a.fresh or not os.path.exists(csv) else pd.read_csv(csv).to_dict("records")
    for spec in a.target:
        label, url = spec.split("@", 1)
        print(f"\n=== {label} ({url}) ===", flush=True)
        row, per = run_target(label, url, cases, items, a.limit)
        rows = [r for r in rows if r["model"] != label] + [row]
        with open(os.path.join(OUT, "model_comparison_cases.jsonl"), "a") as f:
            for p in per:
                f.write(json.dumps(p, default=str) + "\n")
        print({k: row[k] for k in ("translation_accuracy", "full_schema_exact_match",
                                   "reasoning_acceptable", "reasoning_preferred",
                                   "translate_ms_mean", "decide_ms_mean")})
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    make_figure(df, os.path.join(OUT, "model_comparison_figure.png"))
    print(f"\nsaved {csv}")


if __name__ == "__main__":
    main()
