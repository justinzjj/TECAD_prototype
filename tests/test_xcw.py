"""Native baseline cutoff handling without redistributing third-party facts."""

from types import SimpleNamespace
from experiments.baselines import xcw


def test_unsupported_protocol_is_not_a_negative_prediction():
    target = {
        "target_id": "x",
        "tx_hash": "abc",
        "timestamp": "1970-01-01T00:01:40Z",
        "protocol": "other",
    }
    result = xcw.run_one(target, {}, {}, {}, {})
    assert result["pred"] == "NA" and not result["covered"]


def test_causal_inputs_exclude_future_and_include_own_event(tmp_path, monkeypatch):
    monkeypatch.setattr(xcw, "RUN", tmp_path / "run")
    monkeypatch.setattr(xcw, "OFFICIAL", tmp_path)
    monkeypatch.setattr(xcw, "SOUFFLE", "souffle")
    source = {
        "sc_token_deposited": [b"old\tevent", b"own\tevent", b"future\tevent"],
        "token_mapping": [b"a\tb"],
    }
    indexed = {
        "sc_token_deposited": [
            (99, 6648936, 0, "old", source["sc_token_deposited"][0]),
            (100, 6648936, 0, "own", source["sc_token_deposited"][1]),
            (101, 6648936, 0, "future", source["sc_token_deposited"][2]),
        ]
    }
    target = {
        "target_id": "x",
        "chain": "ethereum",
        "tx_hash": "own",
        "timestamp": "1970-01-01T00:01:40Z",
        "protocol": "nomad",
    }

    def run(*args, **kwargs):
        content = (xcw.RUN / "runtime/facts/sc_token_deposited.facts").read_text()
        assert "old" in content and "own" in content and "future" not in content
        assert "a\tb" in (xcw.RUN / "runtime/facts/token_mapping.facts").read_text()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(xcw.subprocess, "run", run)
    result = xcw.run_one(target, source, {"own": (100, 6648936, 0)}, indexed, {})
    assert result["pred"] == 0 and result["covered"]
