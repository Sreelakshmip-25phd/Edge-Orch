"""Model tiers available to download_model.py / start_server.py.

Pick a tier based on available hardware — see README.md for the VRAM/RAM
guidance. All are single-file, llama.cpp-compatible GGUF quantizations
from bartowski (repo IDs/filenames verified against the actual HF repos
before being added here — no invented paths).

Each entry also carries:
  - "role": which MultiLLM role (see ../src/llm_client.py) it's suited
    to — "slm" (fast intent compilation), "goa" (reasoning-heavy zone
    selection/preemption), or "both". Only a suggestion; any tier can
    still be pointed at either role via LOCAL_LLM_URL/_SLM/_GOA.
  - "chat_format": the llama_cpp.server --chat_format value for this
    model family (start_server.py used to hardcode "qwen" for every
    tier, which only happened to work because all 3 original tiers were
    Qwen2.5).
  - "display_name" / "params_b": the human-readable model name and
    parameter count (billions, float) - model_compare.py uses these so
    its table/figure show e.g. "Qwen2.5-7B-Instruct (7B)" sorted by
    size, instead of the internal tier key ("medium").
  - "no_system_role": True for a model whose chat template has no
    system-role slot at all - confirmed for Gemma via
    llama-cpp-python's format_gemma(), which hardcodes
    system_message="" and silently drops it (only a debug log, no
    error), so a system-role prompt sent to it never actually reaches
    the model. src/llm_client.py's MultiLLM.ask() checks this flag
    (set on the provider dict by discover_providers()'s
    *_NO_SYSTEM_ROLE env vars, or by model_compare.py's MODELS lookup)
    and folds system+user into a single user turn instead - the
    standard workaround, verified empirically (0/6 -> 6/6 valid JSON
    on Gemma-2-9B-it with identical instruction content, just moved
    into the user turn).
"""

MODELS = {
    "small": {
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "role": "slm",
        "chat_format": "qwen",
        "display_name": "Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "note": "~1GB. Runs on almost anything, incl. constrained CPU. "
                "Lowest quality — use only if nothing else fits.",
    },
    "medium": {
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "role": "both",
        "chat_format": "qwen",
        "display_name": "Qwen2.5-7B-Instruct",
        "params_b": 7.0,
        "note": "~4.7GB. Sweet spot for this project's JSON-decision "
                "tasks. Needs an 8GB+ GPU for good throughput, or "
                "~16 dedicated (non-throttled) CPU cores.",
    },
    "large": {
        "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "role": "goa",
        "chat_format": "qwen",
        "display_name": "Qwen2.5-14B-Instruct",
        "params_b": 14.0,
        "note": "~9GB. Needs a 12-16GB GPU. Better reasoning quality "
                "than 'medium' if the extra headroom is available.",
    },
    "llama32_1b": {
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "role": "slm",
        "chat_format": "llama-3",
        "display_name": "Llama-3.2-1B-Instruct",
        "params_b": 1.0,
        "note": "~0.8GB. Smallest/fastest option for SLM-tier intent "
                "compilation — a different model family than Qwen2.5 for "
                "response-quality/speed comparison (see model_compare.py).",
    },
    "llama32_3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "role": "slm",
        "chat_format": "llama-3",
        "display_name": "Llama-3.2-3B-Instruct",
        "params_b": 3.0,
        "note": "~2GB. Mid-size Llama family SLM candidate, still fast "
                "enough for per-request intent compilation.",
    },
    "phi35_mini": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "role": "both",
        # llama-cpp-python (as pinned) has no built-in "phi3" chat format
        # registered — leave chat_format unset so the server falls back
        # to the model's own embedded GGUF chat template.
        "chat_format": None,
        "display_name": "Phi-3.5-mini-instruct",
        "params_b": 3.8,
        "note": "~2.4GB, 3.8B params. Strong reasoning-per-parameter — "
                "a good SLM if you want more headroom than Llama-3.2-3B, "
                "or a fast GOA option on constrained hardware.",
    },
    "mistral7b": {
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "role": "goa",
        "chat_format": "mistral-instruct",
        "no_system_role": True,
        "display_name": "Mistral-7B-Instruct-v0.3",
        "params_b": 7.0,
        "note": "~4.4GB. Non-Qwen GOA reasoning candidate for the "
                "zone-selection/preemption comparison. IMPORTANT: "
                "llama-cpp-python's \"mistral-instruct\" chat_format "
                "only handles role in (user, assistant) - a system "
                "message silently falls through every branch and is "
                "dropped with not even a debug log (worse than Gemma's "
                "format_gemma(), which at least logs it). MultiLLM.ask() "
                "works around this the same way, via no_system_role.",
    },
    "gemma2_9b": {
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "role": "goa",
        "chat_format": "gemma",
        "no_system_role": True,
        "display_name": "Gemma-2-9B-it",
        "params_b": 9.0,
        "note": "~5.8GB. Larger non-Qwen GOA reasoning candidate; needs "
                "an 8GB+ GPU for good throughput. IMPORTANT: Gemma's "
                "chat template has no system role - llama-cpp-python's "
                "\"gemma\" chat_format silently drops it. MultiLLM.ask() "
                "works around this automatically via no_system_role "
                "(folds system+user into one user turn) as long as "
                "this tier is referenced by its models.py key (e.g. "
                "model_compare.py's --target gemma2_9b@url) or the "
                "matching LOCAL_LLM_URL*_NO_SYSTEM_ROLE env var is set.",
    },
}

DEFAULT_TIER = "medium"
