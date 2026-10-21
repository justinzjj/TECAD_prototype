"""Original boolean-priority adapter for the frozen eligible history."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

UTC = timezone.utc

FUNGIBLE_TYPES = {"erc20", "native", "native_substrate", "token", "fungible"}


def parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_z(value: Any) -> str:
    return parse_time(value).isoformat().replace("+00:00", "Z")


def _physical_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(name) or "").lower()
        for name in (
            "chain_namespace",
            "chain_id",
            "context_tx_hash",
            "context_block_time",
            "event_index",
            "trace_address",
            "source_unique_key",
            "asset_contract",
            "from_address",
            "to_address",
            "token_id",
            "physical_evidence_id",
        )
    )


def rank_context(
    candidate: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    focal_time = parse_time(candidate["candidate_time"])
    focal_hash = str(candidate["tx_hash"]).lower()
    focal_asset = str(candidate.get("candidate_asset_local_id") or "").lower()
    focal_canonical = str(candidate.get("candidate_canonical_asset_group") or "")
    source = [dict(row) for row in evidence_rows]
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in source:
        unique.setdefault(_physical_key(row), row)
    audit = {
        "source_rows": len(source),
        "duplicate_rows": len(source) - len(unique),
        "future_or_equal_rows": 0,
        "focal_rows": 0,
        "window_violation_rows": 0,
        "eligible_rows": 0,
        "selected_rows": 0,
    }
    eligible: list[dict[str, Any]] = []
    for incoming in unique.values():
        row = dict(incoming)
        tx_hash = str(row.get("context_tx_hash") or "").lower()
        if tx_hash == focal_hash:
            audit["focal_rows"] += 1
            continue
        event_time = parse_time(row["context_block_time"])
        gap = int((focal_time - event_time).total_seconds())
        if gap <= 0:
            audit["future_or_equal_rows"] += 1
            continue
        if gap > 86400:
            audit["window_violation_rows"] += 1
            continue
        asset_contract = str(row.get("asset_contract") or "").lower()
        asset_type = str(row.get("asset_type") or "").lower()
        row["same_local_asset"] = bool(focal_asset and asset_contract == focal_asset)
        row_canonical = str(row.get("canonical_asset_group") or "")
        row["same_canonical_asset"] = bool(
            row["same_local_asset"]
            or (focal_canonical and row_canonical and focal_canonical == row_canonical)
        )
        row["same_asset_family"] = asset_type in FUNGIBLE_TYPES
        row["amount_semantics_comparable"] = row.get("amount_normalized") not in (
            None,
            "",
        )
        row["time_gap_seconds"] = gap
        row["context_block_time"] = iso_z(event_time)
        eligible.append(row)
    eligible.sort(
        key=lambda row: (
            not bool(row["same_canonical_asset"]),
            not bool(row["same_asset_family"]),
            not bool(row["amount_semantics_comparable"]),
            -parse_time(row["context_block_time"]).timestamp(),
            str(row.get("context_tx_hash") or "").lower(),
            str(row.get("event_index") or ""),
            str(row.get("source_unique_key") or ""),
            str(row.get("physical_evidence_id") or ""),
        )
    )
    selected = eligible[:top_k]
    for rank, row in enumerate(selected, start=1):
        row["context_rank"] = rank
    audit["eligible_rows"] = len(eligible)
    audit["selected_rows"] = len(selected)
    return selected, audit
