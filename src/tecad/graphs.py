"""Execution and context graph primitives for normalized-record replay."""

import pandas as pd
import numpy as np
import math
from collections import Counter

ZERO = "0x" + "0" * 40

TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

NUM = [
    "success",
    "native_value_log",
    "gas_log",
    "log_count_log",
    "transfer_count_log",
    "in_tx_log",
    "out_tx_log",
    "counterparty_log",
    "fan_in_log",
    "fan_out_log",
    "reciprocal_log",
    "context_count_log",
    "asset_diversity_log",
    "recency_min_log",
    "recency_p25_log",
    "recency_p50_log",
    "recency_p75_log",
    "recency_max_log",
    "interarrival_mean_log",
    "interarrival_std_log",
    "hhi",
    "relative_time_log",
    "mint",
    "burn",
    "self",
    "same_asset",
]

CAT = ["node_type", "asset_kind", "movement_direction"]


def val(x):
    try:
        if isinstance(x, str) and x.startswith("0x"):
            return float(int(x, 16))
        f = float(x)
        return f if math.isfinite(f) else None
    except:
        return None


def log(x):
    x = val(x)
    return math.log1p(x) if x is not None and x >= 0 else None


def present(x):
    return x is not None and str(x) not in ["", "nan", "None", "<NA>"]


def movement(src, dst):
    return (
        None
        if not present(src) or not present(dst)
        else (
            "mint"
            if src == ZERO
            else "burn" if dst == ZERO else "self" if src == dst else "transfer"
        )
    )


def newgraph():
    return {"nodes": [], "edges": [], "lookup": {}}


def node(g, identity, typ, **kw):
    kw = {k: v for k, v in kw.items() if k not in CAT or v is not None}
    if identity not in g["lookup"]:
        g["lookup"][identity] = len(g["nodes"])
        g["nodes"].append(dict(node_type=typ, **kw))
    return g["lookup"][identity]


def edge(g, a, b, t):
    g["edges"].append((a, b, t))


def decodelog(topics, srcasset):
    if len(topics) >= 3 and topics[0] == TOPIC:
        return {
            "from": "0x" + topics[1][-40:],
            "to": "0x" + topics[2][-40:],
            "asset": srcasset,
            "kind": "non_fungible_token" if len(topics) >= 4 else "fungible_token",
        }
    return None


def execution(root, logs):
    g = newgraph()
    transfers = []
    for x in logs:
        tr = decodelog(x.get("topics", []), x.get("address", "unknown"))
        if tr:
            transfers.append(tr)
    r = node(
        g,
        "root",
        "transaction",
        success=float(bool(root["success"])),
        native_value_log=(
            log(val(root.get("value")) / 1e18)
            if val(root.get("value")) is not None
            else None
        ),
        gas_log=log(root.get("gas")),
        log_count_log=log(len(logs)),
        transfer_count_log=log(len(transfers)),
    )
    for field, typ, et in [
        ("from", "address_account", "sender"),
        ("to", "contract", "receiver"),
    ]:
        if present(root.get(field)):
            edge(g, r, node(g, root[field], typ), et)
    for i, x in enumerate(logs):
        edge(g, r, node(g, "log" + str(i), "event_log"), "emits")
    for i, t in enumerate(transfers):
        direction = movement(t["from"], t["to"])
        v = node(
            g,
            "tr" + str(i),
            "event_transfer",
            asset_kind=t["kind"],
            movement_direction=direction,
            mint=float(direction == "mint"),
            burn=float(direction == "burn"),
            self=float(direction == "self"),
        )
        edge(g, r, v, "moves_asset")
        edge(
            g,
            v,
            node(g, "asset:" + t["asset"], "asset", asset_kind=t["kind"]),
            "uses_asset",
        )
        for fld in ["from", "to"]:
            if t[fld] != ZERO:
                edge(g, v, node(g, t[fld], "address_account"), fld)
    return g


def asset_of(x):
    if "native" in str(x.get("asset_type", "")).lower():
        return "native"
    return str(x["asset_contract"]) if present(x.get("asset_contract")) else None


def context(candidate, rows):
    g = newgraph()
    p = candidate.source_from
    gaps = []
    ct = Counter()
    ins = set()
    outs = set()
    intx = set()
    outtx = set()
    assets = set()
    cp_unknown = False
    asset_unknown = False
    for x in rows:
        a = str(x["from_address"]) if present(x.get("from_address")) else ""
        b = str(x["to_address"]) if present(x.get("to_address")) else ""
        h = x["context_tx_hash"]
        gap = candidate.timestamp - int(
            pd.Timestamp(x["context_block_time"]).timestamp()
        )
        assert 0 < gap <= 86400 and h != candidate.candidate_tx_hash
        gaps.append(gap)
        cp = b if a == p else a
        if not present(cp):
            cp_unknown = True
        else:
            ct[cp] += 1
        if a == p:
            outs.add(cp)
            outtx.add(h)
        if b == p:
            ins.add(cp)
            intx.add(h)
        asset = asset_of(x)
        if asset is None:
            asset_unknown = True
        else:
            assets.add(asset)
    attrs = dict(
        in_tx_log=log(len(intx)),
        out_tx_log=log(len(outtx)),
        counterparty_log=log(len(ct)),
        fan_in_log=log(len(ins)),
        fan_out_log=log(len(outs)),
        reciprocal_log=log(len(ins & outs)),
        context_count_log=log(len(rows)),
        asset_diversity_log=log(len(assets)),
        hhi=sum(((n / len(rows)) ** 2 for n in ct.values())) if rows else None,
    )
    if cp_unknown:
        for field in [
            "counterparty_log",
            "fan_in_log",
            "fan_out_log",
            "reciprocal_log",
            "hhi",
        ]:
            attrs[field] = None
    if asset_unknown:
        attrs["asset_diversity_log"] = None
    if gaps:
        for name, q in [
            ("min", 0),
            ("p25", 0.25),
            ("p50", 0.5),
            ("p75", 0.75),
            ("max", 1),
        ]:
            attrs["recency_" + name + "_log"] = log(np.quantile(gaps, q))
    if len(gaps) > 1:
        intervals = np.diff(sorted(gaps))
        attrs["interarrival_mean_log"] = log(intervals.mean()) if len(intervals) else 0
        attrs["interarrival_std_log"] = log(intervals.std()) if len(intervals) else 0
    r = node(g, "root", "focal_transaction", **attrs)
    if not rows:
        edge(g, r, node(g, "empty", "context_state"), "observed_as")
    for i, x in enumerate(rows):
        gap = gaps[i]
        a = str(x["from_address"]) if present(x.get("from_address")) else ""
        b = str(x["to_address"]) if present(x.get("to_address")) else ""
        at = str(x.get("asset_type", "")).lower()
        kind = (
            "native"
            if "native" in at
            else (
                "multi_token"
                if "1155" in at
                else (
                    "non_fungible_token"
                    if any((s in at for s in ["721", "nft"]))
                    else "fungible_token" if "20" in at else None
                )
            )
        )
        di = movement(a, b)
        asset = asset_of(x)
        same = (
            asset == str(candidate.candidate_asset_local_id)
            if asset is not None and present(candidate.candidate_asset_local_id)
            else None
        )
        v = node(
            g,
            "evt" + str(i),
            "physical_event",
            relative_time_log=log(gap),
            asset_kind=kind,
            movement_direction=di,
            mint=float(di == "mint") if di else None,
            burn=float(di == "burn") if di else None,
            self=float(di == "self") if di else None,
            same_asset=float(same) if same is not None else None,
        )
        tx = node(
            g,
            "tx:" + x["context_tx_hash"],
            "context_transaction",
            relative_time_log=log(gap),
        )
        edge(g, r, v, "has_context_membership")
        edge(g, v, tx, "evidence_of_transaction")
        edge(g, tx, r, "precedes_focal")
        edge(
            g,
            v,
            node(
                g,
                (
                    "asset:" + str(asset)
                    if asset is not None
                    else "unknown_asset:" + str(i)
                ),
                "asset",
                asset_kind=kind,
            ),
            "uses_asset",
        )
        for addr, et in [(a, "from"), (b, "to")]:
            if addr and addr != ZERO:
                edge(g, v, node(g, addr, "address_account"), et)
    return g


def summarize(g, prefix):
    d = {}
    root = g["nodes"][0]
    for k in NUM:
        vv = [n[k] for n in g["nodes"] if k in n and n[k] is not None]
        if k in root:
            d[prefix + k] = root[k]
        elif k in ["mint", "burn", "self", "same_asset"]:
            d[prefix + k + "_ratio"] = np.mean(vv) if vv else None
    for field, values in [
        (
            "asset_kind",
            ["native", "fungible_token", "non_fungible_token", "multi_token"],
        ),
        ("movement_direction", ["mint", "burn", "self", "transfer"]),
    ]:
        obs = [n[field] for n in g["nodes"] if field in n]
        for v in values:
            d[prefix + field + "_" + v + "_ratio"] = (
                obs.count(v) / len(obs) if obs else None
            )
    deg = np.zeros(len(g["nodes"]))
    for a, b, t in g["edges"]:
        deg[a] += 1
        deg[b] += 1
    d.update(
        {
            prefix + "struct_degree_mean": float(deg.mean()),
            prefix + "struct_degree_std": float(deg.std()),
            prefix + "struct_degree_max": float(deg.max()),
            prefix + "size_nodes": len(deg),
            prefix + "size_edges": len(g["edges"]),
        }
    )
    return d
