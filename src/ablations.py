"""Phase 7 - ablations of the proposed system.

Each ablation is the *same* HierarchicalOrchestrator with exactly the
flags listed in DISABLES turned off - nothing else differs (tests check
this). What each one isolates:

  no_memory           episodic + procedural memory AND the capacity digest
                      off -> value of everything the global tier remembers
                      and is told about other zones. (The quick pilot ran
                      memory-only and digest-only variants too: digest-only
                      had no measurable effect and memory-only was not
                      significant, so they were merged into this one.)
  no_intent_cache     zone caches off: the SLM translates every request
                      -> value of the intent cache
  no_cache_sharing    every zone keeps its own cache (the pilot design)
                      -> value of sharing translations across zones
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
    "no_memory": ("use_memory", "use_digest"),
    "no_intent_cache": ("use_cache",),
    "no_cache_sharing": ("use_cache_sharing",),
    "no_preempt_degrade": ("use_preempt_degrade",),
    "no_zone_tier": ("use_zone_tier",),
    "no_cross_zone": ("use_cross_zone",),
}

ISOLATES = {
    "full": "complete proposed system",
    "no_memory": "similar-case memory, learned rules and the capacity digest",
    "no_intent_cache": "the semantic intent cache",
    "no_cache_sharing": "sharing the intent cache across zones",
    "no_preempt_degrade": "pre-emption and the degradation menu",
    "no_zone_tier": "Tier-2 zone agents (local translation + local placement)",
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
