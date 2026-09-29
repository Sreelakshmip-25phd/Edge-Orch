"""Phase 7 - ablations of the proposed system.

Each ablation is the *same* HierarchicalOrchestrator with exactly the
flags listed in DISABLES turned off - nothing else differs (tests check
this). What each one isolates:

  no_memory           episodic + procedural memory off, digest ON
                      -> value of memory specifically
  no_digest           digest exchange off (no digest scan; the LLM sees no
                      capacity; memory re-checks by probing), memory ON
                      -> value of the capacity digest specifically
  memory_ablated      memory AND digest off (the old repo's MemoryAblated,
                      now documented as the combination, not "no memory")
  no_intent_cache     zone caches off: the SLM translates every request
                      -> value of the intent cache
  no_preempt_degrade  LLM action space and rule chain limited to place /
                      reject -> value of pre-emption + degradation
  no_zone_tier        no zone agents: every request goes straight to the
                      global tier (central translator + central placement)
                      -> value of Tier 2 itself
  no_cross_zone       placement restricted to the origin zone (no cross-
                      zone escalation target) -> value of zone cooperation
"""
from orchestrator import HierarchicalOrchestrator

DISABLES = {
    "full": (),
    "no_memory": ("use_memory",),
    "no_digest": ("use_digest",),
    "memory_ablated": ("use_memory", "use_digest"),
    "no_intent_cache": ("use_cache",),
    "no_preempt_degrade": ("use_preempt_degrade",),
    "no_zone_tier": ("use_zone_tier",),
    "no_cross_zone": ("use_cross_zone",),
}

ISOLATES = {
    "full": "complete proposed system",
    "no_memory": "episodic + procedural memory (digest kept)",
    "no_digest": "cross-zone capacity digest (memory kept)",
    "memory_ablated": "memory and digest together (old MemoryAblated)",
    "no_intent_cache": "per-zone semantic intent cache",
    "no_preempt_degrade": "pre-emption and the degradation menu",
    "no_zone_tier": "Tier-2 zone agents (local translation cache + local placement)",
    "no_cross_zone": "cross-zone escalation targets",
}


def make(name, **kw):
    """Build ablation `name`; kw are the shared HierarchicalOrchestrator
    constructor arguments (catalog, embedder, topo, make_llm, ...)."""
    if name not in DISABLES:
        raise KeyError(f"unknown ablation {name!r}; choose from {sorted(DISABLES)}")
    flags = {f: False for f in DISABLES[name]}
    orch = HierarchicalOrchestrator(name=name, **flags, **kw)
    orch.disables = DISABLES[name]
    return orch
