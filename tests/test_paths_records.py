"""Path independence and normalized-observation contracts."""

import hashlib
import json
import sqlite3
import numpy as np
import pytest
from tecad.config import ROOT, load_config
from tecad.records import RecordStore
from tecad.history import rank_distinct_evidence


def test_paths_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config["paths"]["runs"] == ROOT / "runs"
    assert config["paths"]["dataset"] == (ROOT / "data/dataset-V5").resolve()
    override = tmp_path / "custom.yaml"
    override.write_text("paths:\n  runs: runs/custom\n")
    assert load_config(override)["paths"]["runs"] == ROOT / "runs/custom"


def test_record_cache_only_stores_observations(tmp_path):
    path = tmp_path / "records.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE records(kind TEXT,id TEXT,payload TEXT,PRIMARY KEY(kind,id))"
    )
    target = [
        {
            "node_type": "transaction",
            "is_root": True,
            "transaction_success": 1,
            "transaction_success_available": True,
            "transaction_value_log1p": 2.0,
            "transaction_value_available": True,
            "gas_used_log1p": 3.0,
            "gas_used_available": True,
            "event_count_log1p": 0.0,
            "transfer_count_log1p": 0.0,
        }
    ]
    db.execute("INSERT INTO records VALUES(?,?,?)", ("target", "t", json.dumps(target)))
    db.commit()
    db.close()
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    store = RecordStore(tmp_path)
    first = store.target("t")
    assert first.shape == (14,)
    first[:] = 999
    second = store.target("t")
    assert second[0] == 1 and second[1] == 2
    assert isinstance(store.cache[("target", "t")], list)
    with pytest.raises(KeyError):
        store.target("missing")
    store.close()
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        RecordStore(tmp_path)


def test_history_excludes_future_and_deduplicates():
    focal = {
        "focal_time": "2026-01-02T00:00:00Z",
        "focal_tx_hash": "f",
        "focal_mapping_unknown": True,
    }
    event = {
        "physical_evidence_id": "e",
        "context_block_time": "2026-01-01T23:00:00Z",
        "context_tx_hash": "x",
        "from_address": "p",
        "to_address": "q",
        "asset_type": "native",
        "amount_normalized": "1",
    }
    future = {
        **event,
        "physical_evidence_id": "future",
        "context_block_time": "2026-01-02T00:00:01Z",
    }
    rows, count = rank_distinct_evidence(focal, {"p"}, [event, event, future])
    assert count == 1 and [r["physical_evidence_id"] for r in rows] == ["e"]
