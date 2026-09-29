"""Expand INTENT_TEMPLATES with LLM paraphrases and build an OOD probe set.

Uses the existing MultiLLM client (src/llm_client.py), so it picks up
whichever provider is configured in the environment - on the GPU
workstation that's expected to be LOCAL_LLM_URL (self-hosted, e.g. an
OpenAI-compatible server such as llama.cpp/vLLM/text-generation-webui).

Outputs (both under intent_data/, tracked in git — unlike
data_foundation/validated/, these aren't deterministically reproducible
from raw data, since generation depends on a live LLM call):
  intent_templates_expanded.json  - {service_type: [seed + paraphrase templates]}
  intent_ood_probe.json           - held-out sentences, NOT derived from
                                     the seed templates, for generalization
                                     testing (not used at simulation time).

Run from the repo root:
    export LOCAL_LLM_URL=http://127.0.0.1:8000
    python scripts/generate_intent_paraphrases.py

To regenerate the OOD probe for just one service type (e.g. after
spotting a quality issue like repeated place names), without touching
the rest of the file:
    python scripts/generate_intent_paraphrases.py --ood-only ar_session
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402,F401

from llm_client import MultiLLM, discover_providers  # noqa: E402
from scenario import NEW_TYPE_TEMPLATES, PLACES_INITIAL as PLACES, load_intent_pools  # noqa: E402

# Seeds = the training pool (hand-written templates). For the two mid-run
# types, the hand-written templates in scenario.NEW_TYPE_TEMPLATES are the
# seeds; their paraphrases stay held-out (never cache seeds).
_train, _, _ = load_intent_pools()
SEED_INTENT_TEMPLATES = {st: (t if t else NEW_TYPE_TEMPLATES[st][:3]) for st, t in _train.items()}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "intent_data")
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache",
                     "intent_paraphrase_cache.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE), exist_ok=True)

N_PARAPHRASES_PER_SEED = 3
N_OOD_PER_CLASS = 12

SYSTEM_EXPAND = (
    "You write short imperative sentences describing an edge-computing "
    "service request, for a research dataset. Keep the same meaning as "
    "the seed sentence, but vary wording, structure and length. Keep the "
    "literal placeholder {place} exactly as-is (do not translate or "
    "replace it). Return strict JSON: "
    '{"paraphrases": ["...", "...", "..."]}')

SYSTEM_OOD = (
    "You write short imperative sentences describing an edge-computing "
    "service request, for a research dataset's held-out test set. Do NOT "
    "reuse the wording, structure, or sentence openers of the example "
    "sentences given - write genuinely different phrasings (some terse, "
    "some conversational, some as a ticket/log entry) that still clearly "
    "express the same underlying service intent. Each sentence must use "
    "a DIFFERENT concrete real-world place name (own choosing, not "
    "necessarily from the list given) - do not reuse the same place "
    "name, or trivial variants of it, across more than one sentence in "
    "your response. "
    'Return strict JSON: {"sentences": ["...", "...", ...]}')


class _Ask:
    """Adapter: this script's old-style ask(system, user, cache_key=...) -> dict."""

    def __init__(self, m):
        self.m = m

    def ask(self, system, user, cache_key=None):
        return self.m.ask(system, user, kind="paraphrase").data


def expand_templates(llm):
    expanded = {}
    for stype, seeds in SEED_INTENT_TEMPLATES.items():
        pool = list(seeds)
        for seed in seeds:
            user = (f"Service type: {stype}\nSeed sentence: {seed}\n"
                    f"Generate {N_PARAPHRASES_PER_SEED} paraphrases.")
            try:
                out = llm.ask(SYSTEM_EXPAND, user,
                              cache_key=f"expand:{stype}:{seed}")
                paras = [p for p in out.get("paraphrases", [])
                        if "{place}" in p]
                pool.extend(paras)
            except Exception as e:
                print(f"  [expand] {stype!r} seed failed: {e}")
        expanded[stype] = pool
        print(f"  {stype}: {len(seeds)} seeds -> {len(pool)} templates")
    return expanded


def gen_ood_for_class(llm, stype, seeds, cache_suffix=""):
    examples = "\n".join(f"- {s.format(place=PLACES[0])}" for s in seeds)
    user = (f"Service type: {stype}\nExample sentences (do not "
            f"imitate their phrasing):\n{examples}\n\n"
            f"Generate {N_OOD_PER_CLASS} held-out test sentences, each "
            f"with a different place name.")
    out = llm.ask(SYSTEM_OOD, user, cache_key=f"ood:{stype}{cache_suffix}")
    return [{"service_type": stype, "text": s}
            for s in out.get("sentences", [])]


def build_ood_probe(llm):
    probe = []
    for stype, seeds in SEED_INTENT_TEMPLATES.items():
        try:
            probe.extend(gen_ood_for_class(llm, stype, seeds))
        except Exception as e:
            print(f"  [ood] {stype!r} failed: {e}")
        print(f"  {stype}: {len(probe)} OOD sentences so far")
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-only", metavar="SERVICE_TYPE", default=None,
                    help="Regenerate the OOD probe for just this one "
                         "service type, merging into the existing "
                         "intent_ood_probe.json rather than rebuilding "
                         "everything.")
    args = ap.parse_args()

    discover_providers()
    llm = _Ask(MultiLLM(CACHE.replace(".json", ".jsonl"), role="llm", model_label="paraphrase"))
    ood_path = os.path.join(OUT_DIR, "intent_ood_probe.json")

    if args.ood_only:
        stype = args.ood_only
        if stype not in SEED_INTENT_TEMPLATES:
            sys.exit(f"unknown service type {stype!r}; choices: "
                     f"{list(SEED_INTENT_TEMPLATES)}")
        existing = json.load(open(ood_path)) if os.path.exists(ood_path) \
            else []
        kept = [e for e in existing if e["service_type"] != stype]
        print(f"Regenerating OOD probe for {stype!r} only "
             f"(cache-busted, stronger place-diversity prompt)...")
        new = gen_ood_for_class(llm, stype, SEED_INTENT_TEMPLATES[stype],
                                cache_suffix=":v2_place_diverse")
        probe = kept + new
        json.dump(probe, open(ood_path, "w"), indent=1)
        print(f"wrote {ood_path} ({len(probe)} sentences total, "
             f"{len(new)} regenerated for {stype!r})")
        return

    print("Expanding templates via paraphrasing...")
    expanded = expand_templates(llm)
    exp_path = os.path.join(OUT_DIR, "intent_templates_expanded.json")
    json.dump(expanded, open(exp_path, "w"), indent=1)
    print(f"wrote {exp_path}")

    print("\nBuilding OOD probe set...")
    probe = build_ood_probe(llm)
    json.dump(probe, open(ood_path, "w"), indent=1)
    print(f"wrote {ood_path} ({len(probe)} sentences)")


if __name__ == "__main__":
    main()
