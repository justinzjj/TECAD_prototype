"""Load V5 evidence and construct candidate-set representations without run caches."""

from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from .config import ROOT
from .features import (
    evidence_arrays,
    pair_features,
    select_members,
    mean_valid,
    aggregate as mean_std,
)

BLOCKS = ("D", "P", "Ctx")


class Dataset:
    def __init__(self, path, paper_features=None):
        root = Path(path)
        data = root / "data"
        feature_dir = Path(paper_features) if paper_features else data / "features"
        self.root = root
        lock = json.loads((root / "DATASET_LOCK.json").read_text())
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        self.hashes = {}
        for rel, key in [
            ("SPLIT_MANIFEST.csv", "split_manifest_sha256"),
            ("data/dataset.json", "source_dataset_json_sha256"),
        ]:
            digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
            if digest != lock[key] or digest != manifest[key]:
                raise ValueError(f"Dataset lock mismatch: {rel}")
            self.hashes[rel] = digest
        for name in ("candidate_execution.parquet", "CTX_K20.npz"):
            digest = hashlib.sha256((feature_dir / name).read_bytes()).hexdigest()
            if digest != manifest["paper_features"]["files"][name]:
                raise ValueError(f"Paper feature hash mismatch: {name}")
            self.hashes[str(feature_dir / name)] = digest
        for rel, expected in manifest["dataset_inputs"].items():
            digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Dataset feature hash mismatch: {rel}")
            self.hashes[rel] = digest
        self.table = pd.read_csv(root / "SPLIT_MANIFEST.csv", keep_default_na=False)
        self.ids = self.table.target_id.to_numpy(str)
        if not self.table.target_id.is_unique:
            raise ValueError("duplicate target identity")
        self.rows = {
            r: np.flatnonzero(self.table.role.eq(r))
            for r in ("reference", "scale", "cal", "test")
        }
        if any(not len(v) for v in self.rows.values()):
            raise ValueError("missing protocol role")
        if self.table.loc[self.table.role.ne("test"), "dataset_label"].ne(0).any():
            raise ValueError("non-normal fitting/calibration sample")
        with np.load(data / "features/D14.npz", allow_pickle=False) as z:
            ix = self._align(z["target_ids"], self.ids)
            self.dv, self.dm = z["values"][ix], z["mask"][ix].astype(bool)
        self.D = np.where(self.dm, self.dv, np.nan)
        self.pool = pd.read_parquet(
            data / "candidates/pool.parquet",
            columns=["target_id", "candidate_id", "candidate_key", "gap_seconds"],
        )
        ex = pd.read_parquet(feature_dir / "candidate_execution.parquet")
        self.candidate_ids = ex.candidate_id.to_numpy(str)
        self.cv, self.cm = evidence_arrays(
            ex.S_exec_values, ex.S_exec_mask, 21, ex.execution_state
        )
        with np.load(feature_dir / "CTX_K20.npz", allow_pickle=False) as z:
            ix = pd.Index(z["candidate_ids"]).get_indexer(self.candidate_ids)
            self.context_available = ix >= 0
            self.hv = np.zeros((len(ix), 32))
            self.hm = np.zeros((len(ix), 32), bool)
            self.hv[ix >= 0] = z["values"][ix[ix >= 0]]
            self.hm[ix >= 0] = z["mask"][ix[ix >= 0]].astype(bool)
        self.candidate_index = pd.Index(self.candidate_ids)
        self.target_index = pd.Index(self.ids)

    @staticmethod
    def _align(source, target):
        index = pd.Index(source)
        if not index.is_unique:
            raise ValueError("duplicate evidence identity")
        ix = index.get_indexer(target)
        if (ix < 0).any():
            raise ValueError("missing evidence identity")
        return ix

    def build(self, minutes=90, k=10, representation="mean"):
        membership = select_members(self.pool, minutes * 60, k)
        ti = self._align(self.ids, membership.target_id)
        ci = self._align(self.candidate_ids, membership.candidate_id)
        if not self.context_available[ci].all():
            raise ValueError(
                "Selected candidates lack context descriptions; final scope-grid revision is required"
            )
        pv, pm = pair_features(
            self.dv[ti, :8], self.cv[ci, :8], self.dm[ti, :8], self.cm[ci, :8]
        )

        def aggregate(v, m):
            if representation == "state":
                counts = np.bincount(ti, minlength=len(self.ids))
                valid = np.zeros((len(self.ids), m.shape[1]), dtype=int)
                np.add.at(valid, ti, m)
                fraction = np.divide(
                    valid,
                    counts[:, None],
                    out=np.zeros_like(valid, dtype=float),
                    where=counts[:, None] > 0,
                )
                return np.column_stack(
                    [
                        np.log1p(counts),
                        (counts == 0).astype(float),
                        fraction,
                        (valid > 0).any(axis=1).astype(float),
                    ]
                )
            mean = mean_valid(v, m, ti, len(self.ids))
            if representation == "mean":
                return mean
            if representation != "mean_std":
                raise ValueError("unknown representation")
            return mean_std(v, m, ti, len(self.ids))

        return {
            "D": self.D,
            "P": aggregate(pv, pm),
            "Ctx": aggregate(self.hv[ci], self.hm[ci]),
        }, membership
