import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "local_llm"))

import model_compare as MC  # noqa: E402
import reasoning_cases as RC  # noqa: E402


def test_reasoning_case_set_is_large_and_self_consistent():
    cs = RC.generate()
    assert 50 <= len(cs) <= 100
    for c in cs:
        pid = [o["id"] for o in c["choices"]
               if RC._match({k: v for k, v in o.items() if k != "id"}, c["preferred"])]
        assert len(pid) == 1
        assert RC.score(c, {"option": pid[0]})["preferred"]
        assert not RC.score(c, {"option": "o99"})["acceptable"]
        assert not RC.score(c, {"action": "place", "zone": "z_nowhere"})["valid_json"]


def test_model_compare_scores_a_target_with_the_mock(monkeypatch, tmp_path):
    monkeypatch.setattr(MC, "OUT", str(tmp_path))
    monkeypatch.setattr(MC, "provider_for", lambda label, url: {
        "name": "MOCK", "mock": True, "roles": None, "models": ["mock"]})
    monkeypatch.setattr(MC, "served_model", lambda url: "mock.gguf")
    row, per = MC.run_target("medium", "http://x", RC.generate(), MC.translation_set(), limit=40)
    assert row["n_reasoning"] == 40 and row["n_translation"] == 40
    assert 0 < row["reasoning_acceptable"] <= 1 and row["translation_accuracy"] > 0
    assert row["display_name"].startswith("Qwen2.5-7B")
