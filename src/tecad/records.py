"""Recompute descriptors from normalized records (paper timing mode A-N).

Only parsed records are memoized. Graphs, features and history ranking are
recomputed on every call. Raw decoding and normalization are offline.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from . import graphs
from .record_features import extract_target_d, extract_candidate_modules
from .history import rank_distinct_evidence
from .history_boolean import rank_context


class RecordStore:
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        if manifest.get("status") != "PASS":
            raise ValueError("Normalized-record manifest must have status PASS")
        path = directory / "records.sqlite"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["records_sha256"]:
            raise ValueError("Normalized-record checksum mismatch")
        self.sha256 = digest.hexdigest()
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.cache = {}

    def close(self):
        self.db.close()

    def clear(self):
        self.cache.clear()

    def get(self, kind, key):
        identity = (kind, str(key))
        if identity not in self.cache:
            row = self.db.execute(
                "SELECT payload FROM records WHERE kind=? AND id=?", identity
            ).fetchone()
            if row is None:
                raise KeyError(f"Missing normalized record: {identity}")
            self.cache[identity] = json.loads(row[0])
        return self.cache[identity]

    def target(self, tid):
        values, mask = extract_target_d(pd.DataFrame(self.get("target", tid)))
        return np.where(mask, values, np.nan)

    def execution(self, cid):
        record = self.get("execution", cid)
        certificate = record["completeness"]
        if not certificate["logs_file_present"] or certificate[
            "execution_state"
        ] not in {
            "complete",
            "observed_nonempty",
            "observed_nonempty_v2_backfill",
            "observed_nonempty_e3e4_gapfill",
        }:
            raise ValueError(f"Incomplete execution evidence: {cid}")
        graph = graphs.execution(record["root"], record["logs"])
        values, mask = extract_candidate_modules(
            graphs.summarize(graph, "exec_"), {"execution_complete": True}, (graph, {})
        )["S_exec"]
        return values[:8], mask[:8]

    def history(self, cid):
        record = self.get("history", cid)
        scope = record["scope"]
        if scope["rank_adapter"] == "b3_core_boolean":
            focal = dict(
                candidate_time=scope["focal_time"],
                tx_hash=scope["focal_tx_hash"],
                candidate_asset_local_id=scope.get("focal_asset_local_id"),
                candidate_canonical_asset_group=scope.get(
                    "focal_canonical_asset_group"
                ),
            )
            chosen, audit = rank_context(focal, record["events"], top_k=20)
            count = audit["eligible_rows"]
        else:
            focal = dict(
                focal_time=scope["focal_time"],
                focal_tx_hash=scope["focal_tx_hash"],
                focal_asset_local_id=scope.get("focal_asset_local_id"),
                focal_canonical_asset_group=None,
                focal_mapping_unknown=True,
            )
            chosen, count = rank_distinct_evidence(
                focal, {scope["participant_id"]}, record["events"], top_k=20
            )
        if count != record["eligible_count"]:
            raise ValueError(f"Eligible history count mismatch: {cid}")
        candidate = SimpleNamespace(
            source_from=scope["participant_id"],
            timestamp=int(pd.Timestamp(scope["focal_time"]).timestamp()),
            candidate_tx_hash=scope["focal_tx_hash"],
            candidate_asset_local_id=scope.get("focal_asset_local_id"),
        )
        graph = graphs.context(candidate, chosen)
        return extract_candidate_modules(
            graphs.summarize(graph, "ctx_"),
            {"execution_complete": False, "context_complete": True},
            ({}, graph),
        )["S_ctx"]
