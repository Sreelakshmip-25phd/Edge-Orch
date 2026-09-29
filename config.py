"""Shared paths, constants, seeds, run profiles and the model registry.

BASE is always the repo root, so `python main.py`, `python src/evaluator.py`
and `pytest` all resolve the same directories regardless of cwd.
"""
import os
import sys

# The simulator does many tiny mat-vec products (cache lookups, digests).
# On many-core hosts, BLAS/OpenMP thread pools make each one *slower*
# (oversubscription). Cap them unless the user already chose. Must run
# before numpy is first imported - config is imported first everywhere.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")
for _p in (BASE, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- data ------------------------------------------------------------------
DATA_FOUNDATION = os.path.join(BASE, "data_foundation")
RAW = os.path.join(DATA_FOUNDATION, "raw")
VALIDATED = os.path.join(DATA_FOUNDATION, "validated")
REPORTS = os.path.join(DATA_FOUNDATION, "reports")
INTENT_DATA = os.path.join(BASE, "intent_data")

# --- generated outputs (all gitignored) ----------------------------------
RESULTS = os.path.join(BASE, "results")
LLM_CACHE_DIR = os.path.join(BASE, "cache")

# Measured on the GPU machine by scripts/calibrate_latency.py. Absent until
# that script has been run there; see latency_model.py for what happens
# when it is missing (live per-call measurement, never a made-up number).
LATENCY_CALIBRATION = os.path.join(SRC, "latency_distributions.json")
# Optional: container start-up times measured by
# scripts/generate_device_calibration.py --measure-container-start
DEVICE_CALIBRATION = os.path.join(VALIDATED, "device_calibration.json")


# Optional tag to keep separate result trees side by side, e.g. one per
# model in scripts/run_model_sweep.py (RESULTS_TAG=model_medium).
RESULTS_TAG = os.environ.get("RESULTS_TAG", "")


def results_dir(profile):
    """All outputs of one profile live under results/<profile>[__<tag>]/ so
    a smoke run can never overwrite (or be mistaken for) a real run."""
    return os.path.join(RESULTS, f"{profile}__{RESULTS_TAG}" if RESULTS_TAG else profile)


def ensure_dirs(profile=None):
    for d in (DATA_FOUNDATION, RAW, VALIDATED, REPORTS, RESULTS, LLM_CACHE_DIR):
        os.makedirs(d, exist_ok=True)
    if profile:
        for sub in ("scenario", "runs", "tables", "figures"):
            os.makedirs(os.path.join(results_dir(profile), sub), exist_ok=True)


# --- seeds -------------------------------------------------------------------
SEEDS = list(range(10))          # >= 10 seeds for the full campaign
TOPOLOGY_SEED = 42               # KMeans zoning + node allocation, fixed

# --- simulation cadence (kept from the old repo) ----------------------------
DIGEST_S = 5.0                   # zone digest broadcast period
STALE_S = 15.0                   # digest older than this is ignored
POLICY_TICK_S = 120.0            # procedural-rule authoring period
HORIZON_S = 86400.0              # one simulated day

# --- run profiles --------------------------------------------------------------
# smoke : synthetic topology/activity, mock LLM, tiny. For CI / this
#         repo's own tests. Its numbers are NOT results.
# small : real datasets + real LLM, reduced size - a quick sanity run on
#         the GPU machine before committing to the full campaign.
# full  : the evaluation campaign (10k+ requests/day, >= 10 seeds).
PROFILES = {
    "smoke": {"data": "synthetic", "n_requests": 600, "n_zones": 4,
              "seeds": [0, 1], "llm": "mock", "n_devices": 120,
              "horizon_s": 6 * 3600.0},
    "small": {"data": "real", "n_requests": 2000, "n_zones": 10,
              "seeds": [0, 1, 2], "llm": "real", "n_devices": 400,
              "horizon_s": HORIZON_S},
    "full": {"data": "real", "n_requests": 12000, "n_zones": 10,
             "seeds": SEEDS, "llm": "real", "n_devices": 2400,
             "horizon_s": HORIZON_S},
}
DEFAULT_PROFILE = "full"

# Workload load calibration target (greedy acceptance band), as in the
# old repo's calibrate_workload.py.
CALIBRATION_TARGET = (0.80, 0.90)

# --- model registry ------------------------------------------------------------
# The local tiers live in local_llm/models.py (single source of truth for
# download/serve/compare). HOSTED_REFERENCE is an upper-bound quality
# reference only - never a deployment candidate (it doesn't compete on the
# local-hosting/cost axis the comparison is about).
sys.path.insert(0, os.path.join(BASE, "local_llm"))
from models import MODELS as LOCAL_MODELS  # noqa: E402

HOSTED_REFERENCE = {
    "hosted_ref_llama70b": {
        "provider_env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "display_name": "Llama-3.3-70B (hosted reference, not a deployment candidate)",
        "params_b": 70.0,
    },
    "hosted_ref_cerebras70b": {
        "provider_env": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama-3.3-70b",
        "display_name": "Llama-3.3-70B via Cerebras (hosted reference, not a deployment candidate)",
        "params_b": 70.0,
    },
}

# Which model label each role uses in the main evaluation. Used to (a) pick
# the matching entry in latency_distributions.json and (b) tag disk caches,
# so switching models never replays another model's cached answers.
SLM_MODEL_LABEL = os.environ.get("SLM_MODEL_LABEL", "llama32_3b")
LLM_MODEL_LABEL = os.environ.get("LLM_MODEL_LABEL", "medium")
