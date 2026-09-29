"""Tier 2 - zone agents. One real ZoneAgent instance per zone (old-repo
flaw #1: a single shared orchestrator object pretended to be every zone).

Each ZoneAgent owns:
  * its own IntentLibrary (semantic intent cache): starts empty, or with a
    tiny seed drawn from the *training* phrasing pool only; LRU-capped;
  * its own SLM handle (MultiLLM(role="slm")) with its own counters;
  * its own placement policy weights (the global agent may adjust them).

translate(): cache lookup (MiniLM cosine >= threshold) -> hit; otherwise an
SLM call whose free-text output is resolved against the *live* catalog
(exact label / nearest registered type by embedding / novel). A random
~5% of cache hits are shadow-checked: the SLM is run anyway, and if it
disagrees the cached entry is replaced (the system's own self-correction,
no ground truth involved). Ground-truth correctness is scored by the
simulator afterwards, never visible to the agent.

try_local(): the deterministic solver on this zone's live node table.
"""
import json
import os

import numpy as np

from latency_model import measure
from llm_client import LLMUnavailable
from scenario import LATENCY_CLASSES, LOCALITIES, PRIORITIES, PROFILE_FIELDS

GENERIC_NOVEL = {"latency_class": "interactive", "data_locality": "any",
                 "priority": "normal"}


class Embedder:
    """all-MiniLM-L6-v2 sentence embeddings (unit-normalised); TF-IDF
    character n-grams if sentence-transformers / the model are missing.
    encode() memoises per text (a pure speed-up for the simulation; the
    *latency* charged for a lookup is the measured single-sentence encode
    time, see latency_model.measure_embedding_samples)."""

    def __init__(self, fit_texts, prefer="minilm"):
        self._memo = {}
        self.backend = None
        if prefer == "minilm":
            try:
                import torch
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "minilm"
            except Exception:
                self.backend = None
        if self.backend is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(fit_texts)
            self.backend = "tfidf_char"

    def encode_uncached(self, texts):
        if self.backend == "minilm":
            return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                                show_progress_bar=False))
        m = self.vec.transform(list(texts))
        return np.asarray((m / (np.sqrt(m.multiply(m).sum(1)) + 1e-12)).todense())

    def warm(self, texts, batch_size=256):
        """Batch-encode texts into the memo (simulation speed-up only)."""
        todo = sorted({t for t in texts if t not in self._memo})
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            for t, v in zip(chunk, self.encode_uncached(chunk)):
                self._memo[t] = v
        return len(todo)

    def encode(self, texts):
        missing = [t for t in texts if t not in self._memo]
        if missing:
            for t, v in zip(missing, self.encode_uncached(missing)):
                self._memo[t] = v
        return np.stack([self._memo[t] for t in texts])


class IntentLibrary:
    """Semantic cache with an LRU cap. lookup() returns (profile, sim, idx)
    on a hit (sim >= threshold), else (None, best_sim, None)."""

    def __init__(self, embedder, threshold, cap=500, seed_entries=()):
        self.emb, self.threshold, self.cap = embedder, float(threshold), int(cap)
        self.texts, self.profiles, self.last_used = [], [], []
        self.M = None
        self._tick = 0
        self.evictions = 0
        for t, p in seed_entries:
            self.add(t, p)

    def __len__(self):
        return len(self.texts)

    def lookup(self, text):
        if not self.texts:
            return None, 0.0, None
        q = self.emb.encode([text])[0]
        sims = self.M @ q
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            self._tick += 1
            self.last_used[i] = self._tick
            return dict(self.profiles[i]), float(sims[i]), i
        return None, float(sims[i]), None

    def add(self, text, profile):
        v = self.emb.encode([text])[0].reshape(1, -1)
        self._tick += 1
        if len(self.texts) >= self.cap:
            j = int(np.argmin(self.last_used))
            self.texts[j], self.profiles[j], self.last_used[j] = text, dict(profile), self._tick
            self.M[j] = v[0]
            self.evictions += 1
            return
        self.texts.append(text)
        self.profiles.append(dict(profile))
        self.last_used.append(self._tick)
        self.M = v if self.M is None else np.vstack([self.M, v])

    def replace(self, idx, profile):
        self.profiles[idx] = dict(profile)


def slm_system_prompt(catalog):
    return ("You translate an edge-computing service request into a strict JSON object.\n"
            "Respond with ONLY a JSON object with exactly these fields:\n"
            '  "service_type": the catalog name below that fits the request; if none '
            "fits, a short new snake_case name describing the service\n"
            '  "latency_class": "realtime" (<100 ms) | "interactive" (100-500 ms) | '
            '"batch" (>500 ms)\n'
            '  "data_locality": "zone_local" if the data/control must stay in the area, '
            'else "any"\n'
            '  "priority": "low" | "normal" | "high" | "critical"\n'
            "CATALOG: " + json.dumps(catalog.prompt_block(), sort_keys=True))


def resolve_profile(out, catalog, embedder, resolve_threshold=0.55):
    """Turn raw SLM JSON into a valid profile against the live catalog.
    Returns (profile, resolution_kind)."""
    st, _, kind = catalog.resolve(out.get("service_type"), embedder, resolve_threshold)
    if st is not None:
        base = catalog.profile(st)
    else:
        raw = str(out.get("service_type") or "unknown").strip().lower().replace(" ", "_")
        base = {"service_type": f"novel:{raw[:40]}", **GENERIC_NOVEL}
    prof = {"service_type": base["service_type"]}
    for f, allowed in (("latency_class", LATENCY_CLASSES),
                       ("data_locality", LOCALITIES), ("priority", PRIORITIES)):
        v = out.get(f)
        prof[f] = v if v in allowed else base[f]
    return prof, kind


def profiles_equal(a, b):
    return all(a.get(f) == b.get(f) for f in PROFILE_FIELDS)


class ZoneAgent:
    def __init__(self, zone_id, catalog, embedder, slm, *, threshold, cache_cap=500,
                 shadow_rate=0.05, rng=None, seed_entries=(), use_cache=True,
                 latency=None, resolve_threshold=0.55):
        self.zone_id, self.catalog, self.emb, self.slm = zone_id, catalog, embedder, slm
        self.lib = IntentLibrary(embedder, threshold, cache_cap, seed_entries)
        self.use_cache, self.shadow_rate = use_cache, shadow_rate
        self.rng = rng or np.random.default_rng(0)
        self.latency = latency
        self.resolve_threshold = resolve_threshold
        self.policy = {}
        self.counters = {"cache_hits": 0, "cache_misses": 0, "slm_calls": 0,
                         "shadow_checks": 0, "shadow_disagree": 0, "slm_errors": 0}
        self._sys = None
        catalog.on_register(lambda name: self._invalidate())

    def _invalidate(self):
        self._sys = None

    @property
    def system_prompt(self):
        if self._sys is None:
            self._sys = slm_system_prompt(self.catalog)
        return self._sys

    def _slm(self, text, kind, req_id):
        res = self.slm.ask(self.system_prompt, "INPUT: " + text, kind=kind, req_id=req_id)
        (prof, rkind), ms = measure(resolve_profile, res.data, self.catalog, self.emb,
                                    self.resolve_threshold)
        return prof, rkind, res, ms

    def translate(self, t, req_id, text, rec):
        """Fills rec.translation_* and the translation latency; returns a
        profile dict or None (SLM unavailable and no cache hit)."""
        lat = 0.0
        if self.use_cache:
            (prof, sim, idx), ms = measure(self.lib.lookup, text)
            lat += (self.latency.embed_ms() if self.latency else 0.0) + ms
            rec.cache_similarity = round(sim, 4)
            if prof is not None:
                self.counters["cache_hits"] += 1
                rec.translation_source = "cache"
                rec.type_resolution = "cache"
                rec.add_latency("translation", lat)
                if self.rng.uniform() < self.shadow_rate:
                    self._shadow(text, prof, idx, req_id, rec)
                return prof
            self.counters["cache_misses"] += 1
        try:
            prof, rkind, res, ms = self._slm(text, "translate", req_id)
        except LLMUnavailable:
            self.counters["slm_errors"] += 1
            rec.translation_source = "none"
            rec.add_latency("translation", lat)
            return None
        self.counters["slm_calls"] += 1
        rec.translation_source = "slm_fresh" if res.source == "fresh" else "slm_cached_disk"
        rec.type_resolution = rkind
        rec.add_latency("translation", lat + res.sim_ms + ms)
        if self.use_cache:
            self.lib.add(text, prof)
        return prof

    def _shadow(self, text, cached_prof, idx, req_id, rec):
        """Background audit of a cache hit - not on the request's critical
        path, so no latency is charged, but its tokens are (via req_id)."""
        try:
            prof, _, _, _ = self._slm(text, "shadow_check", req_id)
        except LLMUnavailable:
            return
        self.counters["shadow_checks"] += 1
        agree = profiles_equal(prof, cached_prof)
        rec.shadow_checked, rec.shadow_agree = True, agree
        if not agree:
            self.counters["shadow_disagree"] += 1
            self.lib.replace(idx, prof)

    def try_local(self, view, demand, profile, rec):
        """Deterministic solver on this zone's live table. Measured compute
        time is charged to the decision component."""
        nid, ms = measure(view.solve, self.zone_id, demand, profile, self.policy)
        rec.add_latency("decision", ms)
        return nid

    def stats(self):
        return {"cache_entries": len(self.lib), "cache_evictions": self.lib.evictions,
                **self.counters}


# --- threshold sweep utility ------------------------------------------------------
def threshold_sweep(embedder, train_pool, held_templates, held_fixed, places,
                    thresholds=None, target_precision=0.98):
    """Offline calibration of the cache threshold, ground truth used only
    here (design-time), never at run time. Library = training pool (seed
    templates x places); queries = the held-out pool, *including* phrasings
    of service types that are not in the library (the mid-run / never-
    registered types) - a hit on one of those is a false hit, because an
    open-vocabulary cache must miss on a genuinely new service. For each
    threshold: hit rate on known types, false-hit rate on unseen types, and
    hit precision (nearest entry has the right service type). Picks the
    lowest threshold whose precision >= target (max hits at acceptable
    accuracy)."""
    thresholds = thresholds if thresholds is not None else [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
    lib_t, lib_y = [], []
    for st, tm in train_pool.items():
        for tpl in tm:
            for p in places:
                lib_t.append(tpl.format(place=p))
                lib_y.append(st)
    q_t, q_y = [], []
    for st, tm in held_templates.items():
        for tpl in tm:
            q_t.append(tpl.format(place=places[len(q_t) % len(places)]))
            q_y.append(st)
        for s in held_fixed.get(st, []):
            q_t.append(s)
            q_y.append(st)
    if not lib_t or not q_t:
        return {"thresholds": [], "chosen": 0.8, "note": "empty pools"}
    L, Q = embedder.encode(lib_t), embedder.encode(q_t)
    S = Q @ L.T
    best = S.argmax(1)
    best_sim = S.max(1)
    correct = np.array([lib_y[b] == y for b, y in zip(best, q_y)])
    unseen = np.array([not train_pool.get(y) for y in q_y])
    rows = []
    for thr in thresholds:
        hit = best_sim >= thr
        n_hit = int(hit.sum())
        rows.append({"threshold": thr,
                     "hit_rate_known": round(float(hit[~unseen].mean()), 4),
                     "false_hit_rate_unseen": round(float(hit[unseen].mean()), 4)
                     if unseen.any() else 0.0,
                     "precision": round(float(correct[hit].mean()) if n_hit else 1.0, 4),
                     "n_hits": n_hit})
    ok = [r for r in rows if r["precision"] >= target_precision and r["n_hits"] > 0]
    chosen = min(ok, key=lambda r: r["threshold"])["threshold"] if ok else max(thresholds)
    return {"backend": embedder.backend, "n_library": len(lib_t), "n_queries": len(q_t),
            "target_precision": target_precision, "chosen": chosen, "rows": rows}
