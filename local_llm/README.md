# local_llm

Self-hosted, OpenAI-compatible model servers (llama.cpp) for the pipeline,
plus the model-comparison harness. Everything in this folder is meant to run
on the **GPU machine**. Model weights are never committed; `download_model.py`
fetches them.

## Quick start

```bash
pip install -r requirements.txt            # CPU build; see "GPU setup" for CUDA
python start_server.py medium              # downloads + serves Qwen2.5-7B on :8080
python test_server.py                      # smoke test + tokens/sec
export LOCAL_LLM_URL=http://127.0.0.1:8080
cd .. && python main.py --profile small
```

## Separate SLM and LLM models

The zone agents use `MultiLLM(role="slm")` and the global agent uses
`MultiLLM(role="llm")`. To give each role its own model:

```bash
python start_server.py llama32_3b --port 8080 --also medium --also-port 8081
export LOCAL_LLM_URL_SLM=http://127.0.0.1:8080
export LOCAL_LLM_URL_LLM=http://127.0.0.1:8081      # LOCAL_LLM_URL_GOA still accepted
export SLM_MODEL_LABEL=llama32_3b LLM_MODEL_LABEL=medium
```

`SLM_MODEL_LABEL` / `LLM_MODEL_LABEL` must name the models actually being
served. They select the matching entry in `src/latency_distributions.json`
and tag the disk caches, so one model's cached answers are never replayed for
another. For Gemma or Mistral, set `LOCAL_LLM_URL[_SLM|_LLM]_NO_SYSTEM_ROLE=1`:
their chat templates silently drop system messages, so `llm_client.py` folds
the system prompt into the user turn.

## Model tiers (`models.py`)

| tier | model | params | role hint |
|---|---|---|---|
| `small` | Qwen2.5-1.5B-Instruct | 1.5B | slm |
| `medium` | Qwen2.5-7B-Instruct | 7B | both |
| `large` | Qwen2.5-14B-Instruct | 14B | llm |
| `llama32_1b` | Llama-3.2-1B-Instruct | 1B | slm |
| `llama32_3b` | Llama-3.2-3B-Instruct | 3B | slm |
| `phi35_mini` | Phi-3.5-mini-instruct | 3.8B | both |
| `mistral7b` | Mistral-7B-Instruct-v0.3 | 7B | llm (no system role) |
| `gemma2_9b` | Gemma-2-9B-it | 9B | llm (no system role) |

The hosted reference (`config.HOSTED_REFERENCE`, e.g. Llama-3.3-70B via Groq)
is an upper-bound quality reference only. It is labelled "hosted reference,
not a deployment candidate" in every table and figure. If you have another
locally-hostable model downloaded, add it to `MODELS` as an extra row rather
than replacing one.

## Measuring (all on the GPU machine)

| step | command | output |
|---|---|---|
| latency distributions | `python ../scripts/calibrate_latency.py --target <tier>@<url> ...` | `src/latency_distributions.json` (commit it) |
| model comparison | `python model_compare.py --target <tier>@<url> ... [--target hosted_ref_llama70b@hosted]` | `results/model_comparison/` |
| everything, per tier | `python ../scripts/run_model_sweep.py --tiers small,medium,...` | the above plus one full evaluation per tier in `results/<profile>__model_<tier>/` |

`model_compare.py` scores:

- translation accuracy (per field) and full-schema exact match on the
  **held-out** phrasings;
- the 100 auto-scored decision cases in `reasoning_cases.py` (60 zone
  selection, 25 pre-emption, 10 degradation, 5 mixed): valid-JSON,
  acceptable and preferred rates;
- latency mean/p95 and tokens per call.

It queries `/v1/models` first and refuses two labels served by the same model
file. The old repo's comparison had identical rows for `llama32_1b` and
`medium`, which is what that mistake looks like.

## GPU setup

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir "llama-cpp-python[server]"
```

`start_server.py` detects `nvidia-smi` and offloads all layers. On CPU-only
hosts it sizes threads from the cgroup quota. CPU inference is too slow for
the full campaign.

## Known issue: `response_format: json_object`

llama.cpp's grammar JSON mode hangs on the pinned version, so `llm_client.py`
omits it for local servers. It relies on the prompt plus balanced-brace JSON
extraction instead.
