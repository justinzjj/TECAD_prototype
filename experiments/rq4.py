"""RQ4: serial chronological replay; B descriptions or A-N normalized records."""

import json, platform, time
from threadpoolctl import threadpool_limits, threadpool_info
import numpy as np
import pandas as pd
from tecad.model import Detector
from tecad.features import pair_features, mean_valid
from tecad.scoring import calibrate, combine
from tecad.distance import kth_distance, transform
from .common import arguments, setup


class Replay:
    def __init__(
        self,
        data,
        model,
        minutes=90,
        k=10,
        mode="B",
        records=None,
        active=(0, 1, 2),
        threshold=None,
    ):
        self.data = data
        self.model = model
        self.minutes = minutes
        self.k = k
        self.mode = mode
        self.active = active
        self.threshold = threshold
        pool = data.pool.loc[
            (data.pool.gap_seconds > 0) & (data.pool.gap_seconds <= 10800)
        ].sort_values(["target_id", "gap_seconds", "candidate_key"], kind="mergesort")
        self.index = {
            tid: (g.gap_seconds.to_numpy(float), g.candidate_id.to_numpy(str))
            for tid, g in pool.groupby("target_id", sort=False)
        }
        self.store = {}
        self.cache = {}
        self.record_store = None
        if mode == "A-N":
            from tecad.records import RecordStore

            self.record_store = RecordStore(records)
        else:
            for i, cid in enumerate(data.candidate_ids):
                payload = [data.cv[i, :8].tolist(), data.cm[i, :8].tolist()]
                if 2 in active:
                    payload += [data.hv[i].tolist(), data.hm[i].tolist()]
                self.store[str(cid)] = json.dumps(payload, allow_nan=False)

    def clear(self):
        self.cache.clear()
        if self.record_store:
            self.record_store.clear()

    def description(self, cid):
        if self.mode == "A-N":
            cv, cm = self.record_store.execution(cid)
            hv, hm = (
                self.record_store.history(cid)
                if 2 in self.active
                else (np.zeros(32), np.zeros(32, bool))
            )
            return cv, cm, hv, hm
        if cid not in self.cache:
            self.cache[cid] = tuple(
                np.asarray(x, dtype=bool if j % 2 else float)
                for j, x in enumerate(json.loads(self.store[cid]))
            )
        value = self.cache[cid]
        return value if len(value) == 4 else (*value, np.zeros(32), np.zeros(32, bool))

    def candidates(self, tid):
        gaps, cids = self.index.get(tid, (np.array([]), np.array([], str)))
        return cids[
            : min(self.k, int(np.searchsorted(gaps, self.minutes * 60, side="right")))
        ]

    def prime(self, targets):
        for tid in targets:
            if self.record_store:
                self.record_store.get("target", tid)
                for cid in self.candidates(tid) if self.k else []:
                    self.record_store.get("execution", str(cid))
                    if 2 in self.active:
                        self.record_store.get("history", str(cid))
            else:
                for cid in self.candidates(tid) if self.k else []:
                    self.description(str(cid))

    def infer(self, tid):
        start = time.perf_counter()
        i = self.data.target_index.get_loc(tid)
        d = (
            self.record_store.target(tid)
            if self.record_store
            else self.data.D[i].copy()
        )
        chosen = self.candidates(tid) if self.k else []
        desc = [self.description(str(cid)) for cid in chosen]
        loaded = time.perf_counter()
        raw = {
            "D": d[None, :],
            "P": np.full((1, 16), np.nan),
            "Ctx": np.full((1, 32), np.nan),
        }
        if desc:
            cv = np.stack([x[0] for x in desc])
            cm = np.stack([x[1] for x in desc])
            target = np.broadcast_to(d[:8], cv.shape)
            if 1 in self.active:
                pv, pm = pair_features(target, cv, np.isfinite(target), cm)
                raw["P"] = mean_valid(pv, pm, np.zeros(len(desc), int), 1)
            if 2 in self.active:
                raw["Ctx"] = mean_valid(
                    np.stack([x[2] for x in desc]),
                    np.stack([x[3] for x in desc]),
                    np.zeros(len(desc), int),
                    1,
                )
        built = time.perf_counter()
        distance = np.zeros(3)
        for j, b in enumerate(("D", "P", "Ctx")):
            if j in self.active and np.isfinite(raw[b]).any():
                distance[j] = (
                    kth_distance(
                        transform(self.model.models[b], raw[b]),
                        self.model.models[b]["reference"],
                        self.model.neighbors,
                    )[0]
                    / self.model.scales[j]
                )
        # Two-row GEMV preserves the batch scorer's accumulation at tied thresholds.
        score = float(
            combine(np.stack([distance, np.zeros(3)]), self.model.rates, self.active)[0]
        )
        prediction = int(score > self.threshold) if self.threshold is not None else None
        end = time.perf_counter()
        return dict(
            target_id=tid,
            score=score,
            prediction=prediction,
            candidate_count=len(chosen),
            description_s=loaded - start,
            aggregation_s=built - loaded,
            query_fusion_s=end - built,
            total_s=end - start,
        )


def main():
    p = arguments(__doc__)
    p.add_argument("--mode", choices=["B", "A-N"], default="B")
    p.add_argument("--minutes", type=int, default=90)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    c, out, data = setup(a, "rq4")
    raw, _ = data.build(a.minutes, a.k)
    model = Detector().fit(raw, data.rows)
    ns = len(data.rows["scale"])
    nc = len(data.rows["cal"])
    scores = model.scores()
    cut, _, _ = calibrate(scores[ns : ns + nc], 0.005)
    test = data.table.iloc[data.rows["test"]].copy()
    test["_time"] = pd.to_datetime(test.timestamp, format="mixed", utc=True)
    targets = test.sort_values(
        ["_time", "chain", "tx_hash"], kind="mergesort"
    ).target_id.tolist()
    targets = targets if a.limit is None else targets[: a.limit]
    summaries = []
    top_raw, _ = data.build(a.minutes, 1)
    top = Detector().fit(top_raw, data.rows)
    for method, fitted, k, active in [
        ("Local", model, 0, (0,)),
        ("Top1", top, 1, (0, 1, 2)),
        ("DP", model, a.k, (0, 1)),
        ("TECAD", model, a.k, (0, 1, 2)),
    ]:
        scores = fitted.scores(active)
        cut, _, _ = calibrate(scores[ns : ns + nc], 0.005)
        expected = dict(zip(data.ids[fitted.query], scores))
        replay = Replay(
            data, fitted, a.minutes, k, a.mode, c["paths"].get("records"), active, cut
        )
        for state in ("cold", "warm"):
            for repeat in range(1, a.repeats + 1):
                replay.clear()
                if state == "warm":
                    replay.prime(targets)
                start = time.perf_counter()
                rows = [replay.infer(tid) for tid in targets]
                wall = time.perf_counter() - start
                frame = pd.DataFrame(rows)
                errors = np.abs(
                    frame.score.to_numpy() - np.array([expected[t] for t in targets])
                )
                pred = np.array([expected[t] > cut for t in targets])
                mismatch = int((pred != frame.prediction.to_numpy(bool)).sum())
                frame.to_csv(
                    out / f"{a.mode}-{method}-{state}-{repeat}.csv", index=False
                )
                summary = dict(
                    mode=a.mode,
                    method=method,
                    cache=state,
                    repeat=repeat,
                    n=len(frame),
                    wall_s=wall,
                    throughput_per_s=len(frame) / wall,
                    median_s=frame.total_s.median(),
                    p95_s=frame.total_s.quantile(0.95),
                    p99_s=frame.total_s.quantile(0.99),
                    max_score_error=float(errors.max()),
                    prediction_mismatches=mismatch,
                )
                summaries.append(summary)
                pd.DataFrame(summaries).to_csv(out / "TIMING.csv", index=False)
                print(summary, flush=True)
                if mismatch or not np.allclose(
                    frame.score, [expected[t] for t in targets], rtol=1e-9, atol=1e-9
                ):
                    raise RuntimeError("replay differs from freshly fitted batch score")
        if replay.record_store:
            replay.record_store.close()
    (out / "BOUNDARY.json").write_text(
        json.dumps(
            dict(
                mode=a.mode,
                platform=platform.platform(),
                threads=2,
                threadpools=threadpool_info(),
                batch_size=1,
                concurrency=1,
                includes_fit=False,
                includes_raw_rpc=False,
                includes_record_decode=a.mode == "A-N",
                cold="application cache only; OS cache uncontrolled",
                targets=len(targets),
                candidate_minutes=a.minutes,
                candidate_k=a.k,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
