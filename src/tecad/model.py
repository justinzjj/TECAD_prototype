"""Normal-only fitting, fusion, calibration and auditable model persistence."""

from pathlib import Path
import json
import numpy as np
from .data import BLOCKS
from .distance import fit_standardizer, transform, kth_distance, rms
from .scoring import learn_rates, combine


class Detector:
    def __init__(self, neighbors=10):
        self.neighbors = neighbors

    def fit(self, raw, rows):
        self.models = {}
        distances = []
        supported = []
        normalized = []
        scales = []
        query = np.r_[rows["scale"], rows["cal"], rows["test"]]
        nscale = len(rows["scale"])
        for b in BLOCKS:
            x = raw[b]
            model = fit_standardizer(x[rows["reference"]])
            model["reference"] = transform(model, x[rows["reference"]])
            support = np.isfinite(x[query]).any(axis=1)
            distance = kth_distance(
                transform(model, x[query]), model["reference"], self.neighbors
            )
            distance[~support] = 0
            scale = rms(distance[:nscale])
            if scale <= 0:
                raise ValueError(f"{b}: no positive normal scale")
            self.models[b] = model
            distances.append(distance)
            supported.append(support)
            normalized.append(distance / scale)
            scales.append(scale)
        self.scales = np.array(scales)
        self.distances = np.column_stack(distances)
        self.support = np.column_stack(supported)
        self.u = np.column_stack(normalized)
        self.rates, self.positive_counts = learn_rates(
            self.u[:nscale], self.support[:nscale]
        )
        self.query = query
        return self

    def scores(self, active=(0, 1, 2), equal=False, normalized=True):
        return combine(
            self.u if normalized else self.distances,
            self.rates if normalized else self.rates / self.scales,
            active,
            equal,
        )

    def infer(self, raw, active=(0, 1, 2)):
        ds = []
        for j, b in enumerate(BLOCKS):
            if j not in active:
                ds.append(np.zeros(len(raw["D"])))
                continue
            x = raw[b]
            d = kth_distance(
                transform(self.models[b], x),
                self.models[b]["reference"],
                self.neighbors,
            )
            d[~np.isfinite(x).any(axis=1)] = 0
            ds.append(d / self.scales[j])
        return combine(np.column_stack(ds), self.rates, active)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for b, model in self.models.items():
            np.savez_compressed(path / f"{b}.npz", **model)
        np.savez_compressed(
            path / "parameters.npz",
            scales=self.scales,
            rates=self.rates,
            positive_counts=self.positive_counts,
            neighbors=self.neighbors,
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path / "parameters.npz", allow_pickle=False) as z:
            model = cls(int(z["neighbors"]))
            model.scales = z["scales"].copy()
            model.rates = z["rates"].copy()
            model.positive_counts = z["positive_counts"].copy()
        model.models = {}
        for b in BLOCKS:
            with np.load(path / f"{b}.npz", allow_pickle=False) as z:
                model.models[b] = {key: z[key].copy() for key in z.files}
        return model
