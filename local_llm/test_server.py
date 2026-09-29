#!/usr/bin/env python3
"""Smoke-test a running local LLM server against the same style of
prompt the harness actually sends (see GOA_DECIDE_SYS / SLM_SYSTEM in
run_harness_final.py), and report tokens/sec so you can judge whether
the current hardware is fast enough for a real run.

Usage:
    python3 test_server.py [--url http://127.0.0.1:8080]
"""
import argparse
import json
import re
import time

import requests

SAMPLE_SYSTEM = (
    "You translate a service request into a strict JSON object. "
    "Respond with ONLY a JSON object. Fields: "
    '"service_type", "latency_class", "data_locality", "priority". '
    "Never invent new field names."
)
SAMPLE_USER = (
    "INPUT: run a real-time video analytics job near the user for "
    "low latency"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--max_tokens", type=int, default=150)
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    print(f"pinging {args.url}/v1/models ...")
    r = requests.get(f"{args.url}/v1/models", timeout=10)
    r.raise_for_status()
    print("  server reachable:", r.json())

    print("sending sample decision prompt (no response_format — see "
          "README 'known issue' on json_object grammar mode)...")
    t0 = time.time()
    r = requests.post(f"{args.url}/v1/chat/completions",
                       timeout=args.timeout,
                       headers={"Content-Type": "application/json"},
                       json={"model": "local",
                             "messages": [
                                 {"role": "system", "content": SAMPLE_SYSTEM},
                                 {"role": "user", "content": SAMPLE_USER}],
                             "temperature": 0,
                             "max_tokens": args.max_tokens})
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    raw = j["choices"][0]["message"]["content"]
    usage = j.get("usage", {})
    completion_tok = usage.get("completion_tokens", 0)
    tps = completion_tok / dt if dt > 0 else 0

    print(f"\nresponse ({dt:.1f}s, {completion_tok} tokens, "
          f"{tps:.1f} tok/s):")
    print(" ", raw.strip())

    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            print("\nparsed JSON OK:", parsed)
        except json.JSONDecodeError as e:
            print("\nJSON PARSE FAILED:", e)
    else:
        print("\nNO JSON OBJECT FOUND IN RESPONSE — prompt/model mismatch")

    print(f"\n{'OK' if tps >= 5 else 'SLOW'}: {tps:.1f} tok/s — "
          f"{'usable for a real harness run' if tps >= 5 else 'likely too slow; see README hardware guidance'}")


if __name__ == "__main__":
    main()
