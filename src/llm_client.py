"""Multi-provider SLM/LLM caller with an on-disk response cache.

Kept from the old repo: provider discovery (role-specific local servers
first, then a shared local server, then hosted Groq/Cerebras/OpenRouter),
no blocking sleep on 429, the balanced-brace JSON extraction, and the
no-system-role workaround (Gemma / Mistral chat templates silently drop a
system message, so it is folded into the user turn).

Fixed:
  * every ask() writes one LLMCallRecord to telemetry, tagged
    fresh | cached_disk | error, with prompt and completion tokens kept
    separate and the role (slm | llm) explicit. There is no other path to
    a model, so no call site can forget to log.
  * the disk-cache key is a hash of the *full* prompt + model label +
    sampling params. (The old keys were coarse, e.g. goa|type|cpu|digest
    summary, so a "cache hit" could replay an answer given for a
    different prompt.)
  * the cache is append-only JSONL (the old code rewrote the whole JSON
    file after every call - quadratic in run length).
  * roles are "slm" and "llm" (env: LOCAL_LLM_URL_SLM / LOCAL_LLM_URL_LLM;
    LOCAL_LLM_URL_GOA is still accepted as an alias for the llm role).

The disk cache is an *experiment-cost* device (re-running a seed must not
re-bill every call), not a system mechanism: in the simulated world a
cached_disk answer is still an LLM invocation and is charged the same
simulated latency - but it is always counted separately from a fresh call.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import requests as http_requests

PROVIDERS = []
MAX_TOKENS = 400


class LLMUnavailable(RuntimeError):
    pass


def extract_json_object(raw):
    """First balanced {...} object in `raw` (quoted braces skipped)."""
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no '{{' found in response: {raw[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    raise ValueError(f"no balanced JSON object found in response: {raw[:200]!r}")


def _truthy(v):
    return v not in ("", "0", "false", "False", None)


def discover_providers(mock=False, verbose=True):
    """Populate PROVIDERS from the environment (or the mock, for smoke
    runs/tests only)."""
    PROVIDERS.clear()
    if mock:
        PROVIDERS.append({"name": "MOCK", "mock": True, "roles": None,
                          "models": ["mock"]})
        if verbose:
            print("  provider: MOCK LLM (smoke/test only - outputs are NOT results)")
        return PROVIDERS
    for env_name, role in [("LOCAL_LLM_URL_SLM", "slm"), ("LOCAL_LLM_URL_LLM", "llm"),
                           ("LOCAL_LLM_URL_GOA", "llm")]:
        url = os.environ.get(env_name, "")
        if url:
            PROVIDERS.append({
                "name": env_name, "key": "local", "roles": [role], "local": True,
                "no_system_role": _truthy(os.environ.get(f"{env_name}_NO_SYSTEM_ROLE")),
                "url": f"{url.rstrip('/')}/v1/chat/completions", "models": ["local"]})
            if verbose:
                print(f"  provider: {env_name} ({url}) [role={role}]")
    url = os.environ.get("LOCAL_LLM_URL", "")
    if url:
        PROVIDERS.append({
            "name": "LOCAL_LLM_URL", "key": "local", "roles": None, "local": True,
            "no_system_role": _truthy(os.environ.get("LOCAL_LLM_URL_NO_SYSTEM_ROLE")),
            "url": f"{url.rstrip('/')}/v1/chat/completions", "models": ["local"]})
        if verbose:
            print(f"  provider: LOCAL_LLM_URL ({url}) [shared]")
    for name, purl, models in [
            ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions",
             ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
            ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1/chat/completions",
             ["llama-3.3-70b"]),
            ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions",
             ["openai/gpt-oss-20b:free"])]:
        k = os.environ.get(name, "")
        if k:
            PROVIDERS.append({"name": name, "key": k, "url": purl, "models": models,
                              "roles": None})
            if verbose:
                print(f"  provider: {name} ({len(models)} models)")
    return PROVIDERS


@dataclass
class LLMResult:
    data: dict
    source: str                 # fresh | cached_disk
    tokens_in: int
    tokens_out: int
    wall_ms: float              # measured when the response was first produced
    sim_ms: float = 0.0         # simulated latency charged (latency_model)
    raw: str = ""
    parse_ok: bool = True
    extra: dict = field(default_factory=dict)


class DiskCache:
    """Append-only JSONL response cache, shareable between MultiLLM
    instances (e.g. all zone agents of one system share one store, so
    they never hold divergent in-memory copies of the same file)."""
    _open = {}

    def __init__(self, path):
        self.path = path
        self.d = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        self.d[o["k"]] = o["v"]
                    except Exception:
                        continue

    @classmethod
    def shared(cls, path):
        if path not in cls._open:
            cls._open[path] = cls(path)
        return cls._open[path]

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")

    def __len__(self):
        return len(self.d)


class MultiLLM:
    def __init__(self, cache_path, role, telemetry=None, agent="",
                 model_label="", latency=None, providers=None):
        if role not in ("slm", "llm"):
            raise ValueError(f"role must be slm|llm, got {role!r}")
        self.role, self.agent, self.model_label = role, agent, model_label
        self.tel, self.latency = telemetry, latency
        self.providers = providers           # None -> module PROVIDERS
        self.cache = DiskCache.shared(cache_path) if cache_path else DiskCache(None)
        self.fresh = self.cached_disk = self.errors = 0
        self.tokens_in = self.tokens_out = 0
        self.wall_ms = []

    def bind(self, telemetry=None, latency=None, agent=None):
        if telemetry is not None:
            self.tel = telemetry
        if latency is not None:
            self.latency = latency
        if agent is not None:
            self.agent = agent
        return self

    def _key(self, system, user, temperature, sample_idx, max_tokens):
        h = hashlib.sha256()
        for part in (self.model_label, self.role, f"{temperature:.3f}",
                     str(sample_idx), str(max_tokens), system, "\x00", user):
            h.update(part.encode())
        return h.hexdigest()

    def _log(self, req_id, kind, source, tin, tout, wall, sim):
        if self.tel is not None:
            self.tel.log_llm_call(req_id=req_id, agent=self.agent, role=self.role,
                                  kind=kind, source=source, tokens_in=tin,
                                  tokens_out=tout, wall_ms=wall, sim_latency_ms=sim,
                                  model_label=self.model_label)

    def ask(self, system, user, *, kind, req_id=None, temperature=0.0,
            sample_idx=0, max_tokens=MAX_TOKENS):
        k = self._key(system, user, temperature, sample_idx, max_tokens)
        hit = self.cache.get(k)
        if hit is not None:
            self.cached_disk += 1
            sim = self.latency.llm_call(self.role, kind, hit["wall_ms"]) if self.latency else 0.0
            self._log(req_id, kind, "cached_disk", hit["tin"], hit["tout"], 0.0, sim)
            return LLMResult(data=hit["out"], source="cached_disk", tokens_in=hit["tin"],
                             tokens_out=hit["tout"], wall_ms=hit["wall_ms"], sim_ms=sim,
                             raw=hit.get("raw", ""), parse_ok=hit.get("ok", True))
        provs = self.providers if self.providers is not None else PROVIDERS
        provs = sorted(provs, key=lambda p: 0 if (p.get("roles") is None
                                                  or self.role in p["roles"]) else 1)
        last_err = None
        for prov in provs:
            for model in prov["models"]:
                try:
                    raw, tin, tout, wall, est = self._call(prov, model, system, user,
                                                           temperature, max_tokens, kind)
                except Exception as e:           # network / HTTP / rate limit
                    last_err = e
                    continue
                try:
                    out, ok = json.loads(extract_json_object(raw)), True
                    if not isinstance(out, dict):
                        out, ok = {"_value": out}, False
                except Exception:
                    out, ok = {"_parse_error": raw[:300]}, False
                self.fresh += 1
                self.tokens_in += tin
                self.tokens_out += tout
                self.wall_ms.append(wall)
                v = {"out": out, "tin": tin, "tout": tout, "wall_ms": wall,
                     "raw": raw[:2000], "ok": ok, "provider": prov["name"],
                     "tokens_estimated": est}
                self.cache.put(k, v)
                sim = self.latency.llm_call(self.role, kind, wall) if self.latency else 0.0
                self._log(req_id, kind, "fresh", tin, tout, wall, sim)
                return LLMResult(data=out, source="fresh", tokens_in=tin, tokens_out=tout,
                                 wall_ms=wall, sim_ms=sim, raw=raw, parse_ok=ok)
        self.errors += 1
        self._log(req_id, kind, "error", 0, 0, 0.0, 0.0)
        raise LLMUnavailable(f"all providers failed for role={self.role} kind={kind}: {last_err}")

    def _call(self, prov, model, system, user, temperature, max_tokens, kind):
        if prov.get("mock"):
            from mock_llm import mock_respond
            return mock_respond(kind, system, user, temperature)
        if prov.get("no_system_role"):
            msgs = [{"role": "user", "content": system + "\n\n" + user}]
        else:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        payload = {"model": model, "messages": msgs, "temperature": temperature,
                   "max_tokens": max_tokens}
        if not prov.get("local"):
            # llama.cpp's json_object grammar mode hangs - hosted only
            payload["response_format"] = {"type": "json_object"}
        t0 = time.time()
        r = http_requests.post(prov["url"], timeout=120,
                               headers={"Authorization": f"Bearer {prov['key']}",
                                        "Content-Type": "application/json"},
                               json=payload)
        if r.status_code == 429:
            raise RuntimeError(f"rate limited on {prov['name']}/{model}")
        r.raise_for_status()
        wall = (time.time() - t0) * 1000.0
        j = r.json()
        raw = j["choices"][0]["message"]["content"] or ""
        u = j.get("usage") or {}
        tin, tout = u.get("prompt_tokens"), u.get("completion_tokens")
        est = tin is None or tout is None
        if est:
            tin, tout = (len(system) + len(user)) // 4, len(raw) // 4
        return raw, int(tin), int(tout), wall, est

    def stats(self):
        return {"fresh": self.fresh, "cached_disk": self.cached_disk,
                "errors": self.errors, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out}
