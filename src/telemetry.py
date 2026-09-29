"""Phase 0 - the one shared event log.

Every metric in this repo is computed from what is recorded here and
nothing else (metrics.py only reads a Telemetry dump; no component keeps
private counters that don't also land here). Four record streams:

  RequestRecord  - one per offered request, carried through its whole life
                   (arrival -> translation -> placement path -> state
                   transitions -> end cause), incl. the latency breakdown
                   and per-role token usage.
  LLMCallRecord  - one per SLM/LLM invocation *attempt*, at every call site,
                   tagged fresh network call vs. answered from the on-disk
                   response cache vs. error. This is where "reached the LLM
                   stage" and "made a fresh LLM call" are kept apart once,
                   for everyone (fix for old-repo flaw #8).
  EventRecord    - cluster events: node failures/recoveries, pre-emptions,
                   migrations, displacements, service registrations.
  Snapshot       - periodic (digest-tick cadence) cumulative counters:
                   cache size, memory size, calls by type, utilisation,
                   for the call-reduction-over-time curve.

Records are only "complete" once finalize() has run; metrics refuse to
compute over an incomplete record (see assert_complete).
"""
import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# --- vocabularies (validated on write) ---------------------------------
TRANSLATION_SOURCES = ("cache", "slm_fresh", "slm_cached_disk",
                       "llm_fresh", "llm_cached_disk",  # baselines that translate with the LLM
                       "static_rule", "oracle", "none")
PATHS = ("local", "episodic", "procedural", "digest", "llm", "preempt",
         "preempt_local", "llm_degraded", "unresolved")
DECISION_SOURCES = ("local", "memory", "digest", "llm_fresh",
                    "llm_cached_disk", "slm_fresh", "slm_cached_disk",
                    "rule", "none")
STATES = ("REQUESTED", "PLACED", "COMPLETED", "DISPLACED", "REJECTED")
TERMINAL_STATES = ("COMPLETED", "DISPLACED", "REJECTED")
ACTIONS = ("place", "preempt", "degrade", "reject")
CALL_SOURCES = ("fresh", "cached_disk", "error")
ROLES = ("slm", "llm")
INTERRUPTION_NOTES = ("preempted", "node_failure", "migrated", "replanned")
END_CAUSES = ("completed", "rejected", "node_loss", "preempt_unmigrated")

# Latency components (ms). transport_in + translation + escalation +
# decision + cross_zone + deployment = total. See latency_model.py.
LATENCY_COMPONENTS = ("transport_in", "translation", "escalation",
                      "decision", "cross_zone", "deployment")


class IncompleteRecordError(ValueError):
    pass


def _tok():
    return {"slm": {"in": 0, "out": 0}, "llm": {"in": 0, "out": 0}}


def _calls():
    return {"slm_fresh": 0, "slm_cached_disk": 0, "slm_error": 0,
            "llm_fresh": 0, "llm_cached_disk": 0, "llm_error": 0}


@dataclass
class RequestRecord:
    req_id: str
    device_id: str
    origin_zone: str
    send_time: float
    service_type_true: str
    intent_text: str
    phase: str = ""
    priority_true: str = ""
    locality_true: str = ""
    # --- translation -----------------------------------------------------
    translation_source: Optional[str] = None
    translation_correct: Optional[bool] = None     # full 4-field profile vs ground truth
    service_type_resolved: Optional[str] = None
    service_type_correct: Optional[bool] = None
    type_resolution: Optional[str] = None          # exact | nearest | novel | static
    cache_similarity: Optional[float] = None
    shadow_checked: bool = False
    shadow_agree: Optional[bool] = None
    # --- decision --------------------------------------------------------
    path: Optional[str] = None
    decision_source: Optional[str] = None
    action: Optional[str] = None
    escalated: bool = False
    reached_llm_stage: bool = False                # LLM was consulted for this request's decision
    llm_verified: Optional[bool] = None            # its proposal passed the capacity check
    llm_retries: int = 0
    fallback_used: bool = False
    zone_final: Optional[str] = None
    node_final: Optional[str] = None
    cross_zone: bool = False
    degraded: bool = False
    degrade_level: float = 1.0
    victims: List[str] = field(default_factory=list)
    cpu_alloc: float = 0.0
    mem_alloc: float = 0.0
    # --- cost ------------------------------------------------------------
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    tokens: Dict[str, Dict[str, int]] = field(default_factory=_tok)
    calls: Dict[str, int] = field(default_factory=_calls)
    # --- lifecycle -------------------------------------------------------
    state: str = "REQUESTED"
    transitions: List[list] = field(default_factory=list)   # [t, state, note]
    n_interruptions: int = 0
    n_migrations: int = 0
    end_cause: Optional[str] = None
    finalized: bool = False

    # ------------------------------------------------------------------
    def add_latency(self, component, ms):
        if component not in LATENCY_COMPONENTS:
            raise ValueError(f"unknown latency component {component!r}")
        self.latency_breakdown[component] = \
            self.latency_breakdown.get(component, 0.0) + float(ms)

    @property
    def total_latency_ms(self):
        return float(sum(self.latency_breakdown.values()))

    @property
    def accepted(self):
        """Placed at all, immediately on arrival."""
        return any(s == "PLACED" and n == "arrival"
                   for _, s, n in self.transitions)

    def validate(self):
        if self.translation_source is not None and \
                self.translation_source not in TRANSLATION_SOURCES:
            raise ValueError(f"bad translation_source {self.translation_source!r}")
        if self.path is not None and self.path not in PATHS:
            raise ValueError(f"bad path {self.path!r}")
        if self.decision_source is not None and \
                self.decision_source not in DECISION_SOURCES:
            raise ValueError(f"bad decision_source {self.decision_source!r}")
        if self.action is not None and self.action not in ACTIONS:
            raise ValueError(f"bad action {self.action!r}")
        if self.state not in STATES:
            raise ValueError(f"bad state {self.state!r}")

    def missing_fields(self):
        miss = []
        if not self.finalized:
            miss.append("finalized")
        for f in ("translation_source", "path", "decision_source", "end_cause"):
            if getattr(self, f) is None:
                miss.append(f)
        if self.state not in TERMINAL_STATES:
            miss.append("terminal_state")
        if not self.transitions or self.transitions[0][1] != "REQUESTED":
            miss.append("transitions")
        if "transport_in" not in self.latency_breakdown:
            miss.append("latency_breakdown")
        return miss


@dataclass
class LLMCallRecord:
    t: float
    req_id: Optional[str]
    agent: str            # e.g. "zone:z3", "global", "react", "lats"
    role: str             # slm | llm
    kind: str             # translate | shadow_check | decide | rule_author | react_step | ...
    source: str           # fresh | cached_disk | error
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: float = 0.0  # measured wall time of the real network call (fresh only)
    sim_latency_ms: float = 0.0   # what the simulator charged for it
    model_label: str = ""


@dataclass
class EventRecord:
    t: float
    kind: str
    data: Dict = field(default_factory=dict)


@dataclass
class Snapshot:
    t: float
    data: Dict = field(default_factory=dict)


class Telemetry:
    """Collects every stream for one (system, seed) run. `now` is kept in
    sync with simulated time by the simulator, so LLM call sites can
    timestamp themselves without threading `t` through every signature."""

    def __init__(self, run_meta=None):
        self.meta = dict(run_meta or {})
        self.requests: Dict[str, RequestRecord] = {}
        self.llm_calls: List[LLMCallRecord] = []
        self.events: List[EventRecord] = []
        self.snapshots: List[Snapshot] = []
        self.now = 0.0
        self._cum = Counter()

    # --- requests ------------------------------------------------------
    def new_request(self, **kw):
        rec = RequestRecord(**kw)
        if rec.req_id in self.requests:
            raise ValueError(f"duplicate req_id {rec.req_id}")
        rec.transitions.append([round(rec.send_time, 3), "REQUESTED", "arrival"])
        self.requests[rec.req_id] = rec
        return rec

    def transition(self, req_id, t, state, note=""):
        if state not in STATES:
            raise ValueError(f"bad state {state!r}")
        rec = self.requests[req_id]
        if rec.finalized:
            raise ValueError(f"{req_id} already finalized")
        rec.state = state
        rec.transitions.append([round(t, 3), state, note])
        # every non-arrival (re)placement or loss is an interruption of a
        # running service; a successful re-placement is a migration
        if note in INTERRUPTION_NOTES:
            rec.n_interruptions += 1
        if note in ("migrated", "replanned"):
            rec.n_migrations += 1
        self._cum[f"state_{state}"] += 1

    def finalize(self, req_id, end_cause):
        if end_cause not in END_CAUSES:
            raise ValueError(f"bad end_cause {end_cause!r}")
        rec = self.requests[req_id]
        rec.end_cause = end_cause
        rec.validate()
        rec.finalized = True

    # --- LLM / SLM calls -----------------------------------------------
    def log_llm_call(self, *, req_id, agent, role, kind, source,
                     tokens_in=0, tokens_out=0, wall_ms=0.0,
                     sim_latency_ms=0.0, model_label=""):
        if role not in ROLES:
            raise ValueError(f"bad role {role!r}")
        if source not in CALL_SOURCES:
            raise ValueError(f"bad call source {source!r}")
        rec = LLMCallRecord(t=round(self.now, 3), req_id=req_id, agent=agent,
                            role=role, kind=kind, source=source,
                            tokens_in=int(tokens_in), tokens_out=int(tokens_out),
                            wall_ms=float(wall_ms),
                            sim_latency_ms=float(sim_latency_ms),
                            model_label=model_label)
        self.llm_calls.append(rec)
        self._cum[f"{role}_{source}"] += 1
        self._cum[f"{role}_tokens_in"] += int(tokens_in)
        self._cum[f"{role}_tokens_out"] += int(tokens_out)
        if req_id is not None and req_id in self.requests:
            r = self.requests[req_id]
            r.calls[f"{role}_{source}"] += 1
            r.tokens[role]["in"] += int(tokens_in)
            r.tokens[role]["out"] += int(tokens_out)
        return rec

    # --- events / snapshots ----------------------------------------------
    def log_event(self, t, kind, **data):
        self.events.append(EventRecord(t=round(t, 3), kind=kind, data=data))
        self._cum[f"event_{kind}"] += 1

    def count(self, key, n=1):
        """Cumulative counter that lands in every later snapshot. Used for
        per-request path/translation counters so the call-reduction curve
        can be drawn from snapshots alone."""
        self._cum[key] += n

    def snapshot(self, t, **extra):
        data = dict(self._cum)
        data.update(extra)
        self.snapshots.append(Snapshot(t=round(t, 3), data=data))

    # --- integrity --------------------------------------------------------
    def assert_complete(self):
        bad = {rid: r.missing_fields() for rid, r in self.requests.items()
               if r.missing_fields()}
        if bad:
            first = list(bad.items())[:3]
            raise IncompleteRecordError(
                f"{len(bad)} incomplete request record(s), e.g. {first}")

    # --- persistence -------------------------------------------------------
    def dump(self, path):
        """gzip JSONL: first line meta, then one tagged record per line."""
        with gzip.open(path, "wt") as f:
            f.write(json.dumps({"_type": "meta", **self.meta}) + "\n")
            for r in self.requests.values():
                f.write(json.dumps({"_type": "request", **asdict(r)}) + "\n")
            for c in self.llm_calls:
                f.write(json.dumps({"_type": "llm_call", **asdict(c)}) + "\n")
            for e in self.events:
                f.write(json.dumps({"_type": "event", **asdict(e)}) + "\n")
            for s in self.snapshots:
                f.write(json.dumps({"_type": "snapshot", **asdict(s)}) + "\n")

    @classmethod
    def load(cls, path):
        tel = cls()
        with gzip.open(path, "rt") as f:
            for line in f:
                o = json.loads(line)
                typ = o.pop("_type")
                if typ == "meta":
                    tel.meta = o
                elif typ == "request":
                    rec = RequestRecord(**o)
                    tel.requests[rec.req_id] = rec
                elif typ == "llm_call":
                    tel.llm_calls.append(LLMCallRecord(**o))
                elif typ == "event":
                    tel.events.append(EventRecord(**o))
                elif typ == "snapshot":
                    tel.snapshots.append(Snapshot(**o))
        return tel
