#!/usr/bin/env python3
"""Run unchanged XCW rules on published Nomad facts visible at each target time.

This script deliberately never reads a label or TECAD prediction. It creates
cutoff-specific inputs from official facts and executes Souffle for each target.
"""

import argparse
import bisect
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tecad.config import load_config

OFFICIAL = FACTS = SOUFFLE = RUN = None
STATIC = {
    "wrapped_native_token",
    "token_mapping",
    "bridge_controlled_address",
    "cctx_finality",
}
ALT = {"alternative_chains", "additional_alternative_chains"}
RULES = {
    "SC_DepositsWithoutMatch",
    "TC_DepositsWithoutMatch",
    "SC_WithdrawalsWithoutMatchWithAdditionalData",
    "TC_WithdrawalsWithoutMatchWithAdditionalData",
    "CCTX_Deposit_FinalityBreak",
    "CCTX_Withdrawal_FinalityBreak",
    "SingleTokenDeposits",
    "SingleEventDeposit",
    "SingleTokenWithdrawal",
    "SC_RandomTransferEthToBridge",
    "SC_WithdrewTokensToBridge",
    "SC_ValidERC20TokenDeposit_Anomalies",
    "SC_ValidERC20TokenWithdrawal_Anomalies",
    "SC_ValidNativeTokenDeposit_Anomalies",
    "TC_ValidERC20TokenDeposit_Anomalies",
    "TC_ValidERC20TokenWithdrawal_Anomalies",
    "TC_ValidNativeTokenWithdrawal_Anomalies",
}


def utc_seconds(value):
    return int(
        datetime.fromisoformat(
            value.replace(" UTC", "+00:00").replace("Z", "+00:00")
        ).timestamp()
    )


def read_targets(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {"target_id", "chain", "tx_hash", "timestamp", "protocol"} <= set(rows[0])
    assert "dataset_label" not in rows[0]
    assert len(rows) == len({row["target_id"] for row in rows})
    return rows


def load_facts():
    source = {}
    for path in FACTS.glob("*.facts"):
        source[path.stem] = path.read_bytes().splitlines()
    txmeta = {}
    for name in ("transaction", "additional_transaction"):
        for raw in source[name]:
            row = raw.decode().split("\t")
            key = row[2].lower()
            meta = (int(row[0]), int(row[1]), int(row[3]))
            if key in txmeta:
                assert txmeta[key] == meta, ("ambiguous_hash", key)
            txmeta[key] = meta
    indexed, unknown = {}, Counter()
    for name, lines in source.items():
        if name in STATIC:
            continue
        current = []
        for raw in lines:
            row = raw.decode().split("\t")
            hash_index = (
                2
                if name in ("transaction", "additional_transaction")
                else 1 if name in ALT else 0
            )
            key = row[hash_index].lower()
            if key not in txmeta:
                unknown[name] += 1
                continue
            ts, chain_id, txindex = txmeta[key]
            current.append((ts, chain_id, txindex, key, raw))
        current.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        indexed[name] = current
    return source, txmeta, indexed, unknown


def declarations():
    result = {}
    for path in list(OFFICIAL.glob("*.dl")) + list((OFFICIAL / "lib").glob("*.dl")):
        for name, args in re.findall(r"\.decl\s+(\w+)\(([^\n]+)\)", path.read_text()):
            result[name] = [x.strip().split(":")[0] for x in args.split(",")]
    return result


def run_one(target, source, txmeta, indexed, schema, sample=False):
    key = target["tx_hash"].lower()
    observed = txmeta.get(key)
    target_ts = utc_seconds(target["timestamp"])
    if target["protocol"] != "nomad":
        return dict(
            target_id=target["target_id"],
            status="unsupported_protocol",
            reason="native_rules_require_bridge_specific_semantics",
            pred="NA",
            covered=False,
        )
    if not observed:
        alt = any(
            key == raw.decode().split("\t")[1].lower()
            for name in ALT
            for raw in source.get(name, [])
        )
        return dict(
            target_id=target["target_id"],
            status="outside_configured_domain" if alt else "target_transaction_missing",
            reason=(
                "alternative_nomad_domain" if alt else "no_official_transaction_fact"
            ),
            pred="NA",
            covered=False,
        )
    ts, chain_id, txindex = observed
    if ts != target_ts:
        return dict(
            target_id=target["target_id"],
            status="timestamp_mismatch",
            reason=f"manifest={target_ts};fact={ts}",
            pred="NA",
            covered=False,
        )
    domain = {"ethereum": 6648936, "moonbeam": 1650811245}.get(target["chain"])
    if domain != chain_id:
        return dict(
            target_id=target["target_id"],
            status="chain_domain_mismatch",
            reason=f'manifest={target["chain"]};fact_domain={chain_id}',
            pred="NA",
            covered=False,
        )
    inputs = RUN / "runtime/facts"
    outputs = RUN / "runtime/outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    count = {}
    target_facts = []
    for name, lines in source.items():
        path = inputs / (name + ".facts")
        if name in STATIC:
            selected = lines
        else:
            rr = indexed[name]
            end = bisect.bisect_left(rr, (target_ts,))
            selected = [x[4] for x in rr[:end]]
            stop = bisect.bisect_right(rr, (target_ts, 1 << 63))
            own = [x[4] for x in rr[end:stop] if x[3] == key]
            selected.extend(own)
            if own:
                target_facts.append(name)
        path.write_bytes(b"\n".join(selected) + (b"\n" if selected else b""))
        count[name] = len(selected)
    event_present = bool(
        {
            "sc_token_deposited",
            "sc_token_withdrew",
            "tc_token_deposited",
            "tc_token_withdrew",
        }
        & set(target_facts)
    )
    if not event_present:
        return dict(
            target_id=target["target_id"],
            status="bridge_event_missing",
            reason="no_target_bridge_event_fact",
            pred="NA",
            covered=False,
        )
    for path in outputs.glob("*.csv"):
        path.unlink()
    start = time.perf_counter()
    process = subprocess.run(
        [
            str(SOUFFLE),
            "-j",
            "2",
            "-F",
            str(inputs),
            "-D",
            str(outputs),
            str(OFFICIAL / "acceptance-rules.dl"),
        ],
        cwd=OFFICIAL,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"Souffle failed for {key}: {process.stderr[:350]}")
    alerts, matches = [], []
    for path in outputs.glob("*.csv"):
        name = path.stem
        columns = schema[name]
        indices = [i for i, col in enumerate(columns) if "transaction_hash" in col]
        if not indices:
            continue
        for raw in path.read_text().splitlines():
            row = raw.split("\t")
            if any(i < len(row) and row[i].lower() == key for i in indices):
                if name in RULES:
                    alerts.append(name)
                if name.startswith("CCTX_") or (
                    name.endswith("MatchWithAdditionalData")
                    and "WithoutMatch" not in name
                ):
                    matches.append(
                        {
                            "relation": name,
                            "hashes": [row[i] for i in indices if i < len(row)],
                            "tuple": row,
                        }
                    )
    result = dict(
        target_id=target["target_id"],
        chain=target["chain"],
        protocol=target["protocol"],
        tx_hash=key,
        status="ran_published_facts_causal",
        reason="official_fact_range;raw_completeness_audited_separately",
        covered=True,
        pred=int(bool(alerts)),
        rules=";".join(sorted(set(alerts))),
        matches_json=json.dumps(matches, separators=(",", ":")),
        cutoff_utc=target["timestamp"],
        target_fact_relations=";".join(sorted(target_facts)),
        runtime_s=round(elapsed, 5),
        fact_rows=sum(count.values()),
        source_kind="official_facts_time_filtered",
        method="XCW-native",
    )
    if sample:
        result["input_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs.glob("*.facts")
        }
        result["output_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in outputs.glob("*.csv")
        }
        result["input_counts"] = count
        result["output_target_rows"] = matches
    return result


def main():
    global OFFICIAL, FACTS, SOUFFLE, RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    upstream = config["paths"]["xcw"]
    pinned_commit = "4cc1ff173c18f80d275d7af6cbd318cdd3906e77"
    if not (upstream / ".git").exists():
        parser.error("paths.xcw must be a Git checkout of the pinned upstream revision")
    actual = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != pinned_commit:
        parser.error(f"XCW revision mismatch: expected {pinned_commit}, got {actual}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "status",
            "--porcelain",
            "--",
            "cross-chain-rules-validator/datalog",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        parser.error(
            "XCW datalog rules/facts have local changes; use the clean pinned checkout"
        )
    OFFICIAL = upstream / "cross-chain-rules-validator/datalog"
    FACTS = OFFICIAL / "nomad-bridge/facts"
    SOUFFLE = shutil.which(config.get("tools", {}).get("souffle", "souffle"))
    if not SOUFFLE:
        parser.error("Souffle executable missing; install it or set tools.souffle")
    if not FACTS.is_dir():
        parser.error(f"XCW facts missing: {FACTS}. See data/README.md")
    RUN = config["paths"]["runs"] / "xcw"
    output = RUN / "predictions.jsonl"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}; use a fresh paths.runs")
    with (config["paths"]["dataset"] / "SPLIT_MANIFEST.csv").open() as stream:
        # Remove labels before inference; keep a separate evaluation join below.
        panel = [row for row in csv.DictReader(stream) if row["role"] == "test"]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        panel = panel[: args.limit]
    targets = [
        {k: row[k] for k in ("target_id", "chain", "tx_hash", "timestamp", "protocol")}
        for row in panel
    ]
    RUN.mkdir(parents=True, exist_ok=True)
    source, txmeta, indexed, unknown = load_facts()
    schema = declarations()
    source_hash = {
        name: hashlib.sha256((FACTS / (name + ".facts")).read_bytes()).hexdigest()
        for name in source
    }
    rule_hash = {
        str(p.relative_to(OFFICIAL)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OFFICIAL.rglob("*.dl")
    }
    (RUN / "inputs.json").write_text(
        json.dumps(
            {
                "upstream_commit": actual,
                "facts_sha256": source_hash,
                "rules_sha256": rule_hash,
                "unindexed_dynamic_rows": dict(unknown),
                "targets": len(targets),
            },
            indent=2,
        )
        + "\n"
    )
    results = []
    with output.open("x") as stream:
        for i, target in enumerate(targets, 1):
            result = run_one(target, source, txmeta, indexed, schema, args.sample)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            results.append(result)
            if i % 100 == 0:
                print(f"{i}/{len(targets)}", flush=True)
    # Native uncovered cases remain NA. The paper full-panel comparison uses
    # the explicit system policy "no output -> no alert", not native coverage.
    tp = fp = tn = fn = na = 0
    for target, result in zip(panel, results):
        prediction = result["pred"]
        na += prediction == "NA"
        positive = str(prediction) == "1"
        anomaly = int(target["dataset_label"]) == 1
        tp += anomaly and positive
        fp += not anomaly and positive
        tn += not anomaly and not positive
        fn += anomaly and not positive
    metrics = dict(
        N=len(panel),
        TP=tp,
        FP=fp,
        TN=tn,
        FN=fn,
        native_NA=na,
        policy="no_output_as_no_alert",
        partial=args.limit is not None,
    )
    (RUN / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
