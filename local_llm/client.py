"""Minimal client for the local LLM server, matching the request/response
shape run_harness_final.py's MultiLLM already expects from any provider
(OpenAI-compatible /v1/chat/completions). Useful for quick manual checks;
the harness itself talks to the server directly via the PROVIDERS list
(see integrate_with_harness.py in this folder).
"""
import json
import re

import requests


class LocalLLMClient:
    def __init__(self, url="http://127.0.0.1:8080", model="local"):
        self.url = url.rstrip("/")
        self.model = model

    def ask(self, system, user, max_tokens=200, timeout=120):
        r = requests.post(f"{self.url}/v1/chat/completions", timeout=timeout,
            headers={"Content-Type": "application/json"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0, "max_tokens": max_tokens})
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError(f"no JSON object in response: {raw!r}")
        return json.loads(m.group(0))


if __name__ == "__main__":
    c = LocalLLMClient()
    out = c.ask(
        system='Respond with ONLY a JSON object: {"answer": "<word>"}',
        user="What is the capital of France?")
    print(out)
