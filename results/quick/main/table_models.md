# Table 3 - local models compared

Probe: 158 held-out translations and 100 decision cases per model. e2e_*: the full system on the quick workloads with that one model as both SLM and LLM (mean over e2e_seeds seeds).

| display_name | params_b | hosted_reference | translation_accuracy | full_schema_exact_match | reasoning_valid_json | reasoning_acceptable | reasoning_preferred | translate_ms_mean | translate_ms_p95 | decide_ms_mean | decide_ms_p95 | tokens_in_per_call | tokens_out_per_call | e2e_seeds | e2e_accepted | e2e_completed_correct | e2e_new_types_accepted | e2e_calls_per_req | e2e_setup_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1.0 | False | 0.7674 | 0.519 | 0.17 | 0.08 | 0.05 | 149.9 | 175.5 | 319.1 | 811.1 | 649.1 | 53.6 | 3.0 | 0.867 | 0.601 | 0.678 | 0.497 | 211.233 |
| Qwen2.5-1.5B-Instruct | 1.5 | False | 0.8877 | 0.7089 | 1.0 | 0.92 | 0.49 | 192.9 | 207.3 | 267.4 | 343.5 | 663.1 | 39.4 | 3.0 | 0.919 | 0.83 | 0.933 | 0.254 | 78.545 |
| Llama-3.2-3B-Instruct | 3.0 | False | 0.8877 | 0.7089 | 1.0 | 1.0 | 0.59 | 288.9 | 311.6 | 277.3 | 335.9 | 649.1 | 34.8 | 3.0 | 0.915 | 0.786 | 0.964 | 0.244 | 84.405 |
| Phi-3.5-mini-instruct | 3.8 | False | 0.9494 | 0.8861 | 1.0 | 1.0 | 0.74 | 484.9 | 500.8 | 579.7 | 694.4 | 786.2 | 54.8 | 3.0 | 0.917 | 0.836 | 0.853 | 0.259 | 138.24 |
| Qwen2.5-7B-Instruct | 7.0 | False | 0.9114 | 0.7532 | 1.0 | 1.0 | 0.75 | 476.4 | 495.7 | 539.5 | 643.2 | 663.1 | 35.9 | 3.0 | 0.891 | 0.767 | 0.764 | 0.281 | 152.374 |
| Mistral-7B-Instruct-v0.3 | 7.0 | False | 0.9304 | 0.7722 | 1.0 | 0.98 | 0.73 | 566.5 | 615.2 | 849.6 | 1136.6 | 748.6 | 53.8 | 3.0 | 0.912 | 0.832 | 0.876 | 0.261 | 175.07 |
| Gemma-2-9B-it | 9.0 | False | 0.9604 | 0.9051 | 1.0 | 1.0 | 0.8 | 857.1 | 887.2 | 783.5 | 919.7 | 679.8 | 43.1 | 3.0 | 0.915 | 0.844 | 0.808 | 0.249 | 186.917 |
| Qwen2.5-14B-Instruct | 14.0 | False | 0.981 | 0.9367 | 1.0 | 1.0 | 0.81 | 822.0 | 907.3 | 1017.7 | 1212.1 | 663.1 | 34.5 | 3.0 | 0.916 | 0.856 | 0.806 | 0.243 | 206.176 |
