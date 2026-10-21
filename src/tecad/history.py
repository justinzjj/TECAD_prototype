"""Deterministic physical-event selection from the complete eligible history."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def _timestamp_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace(" UTC", "+00:00")
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _asset_kind(asset_type: Any) -> tuple[str, bool | None]:
    value = str(asset_type or "").lower()
    if value in {"native", "native_substrate"}:
        return "native", True
    if value in {"erc20", "token", "fungible"}:
        return "fungible_token", True
    if "721" in value or value == "nft":
        return "non_fungible_token", False
    if "1155" in value:
        return "multi_token", False
    return "unknown_other", None


def _boolean_priority(value: bool | None) -> int:
    return 0 if value is True else 1 if value is None else 2


def _rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    event_index = _optional_text(row.get("event_index"))
    try:
        event_key: tuple[int, Any] = (
            (0, int(event_index)) if event_index is not None else (1, "")
        )
    except ValueError:
        event_key = (1, event_index or "")
    return (
        _boolean_priority(row.get("same_canonical_asset")),
        _boolean_priority(row.get("same_asset_family")),
        _boolean_priority(row.get("amount_semantics_comparable")),
        float(row["time_gap_seconds"]),
        str(row["context_tx_hash"]),
        event_key,
        str(row["physical_evidence_id"]),
    )


def rank_distinct_evidence(
    focal: Mapping[str, Any],
    participants: set[str],
    evidence: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate and rank physical events once per focal, before incidence."""
    focal_time = _timestamp_dt(focal["focal_time"])
    focal_hash = str(focal["focal_tx_hash"]).lower()
    participant_set = {value.lower() for value in participants}
    unique: dict[str, dict[str, Any]] = {}
    for incoming in evidence:
        row = dict(incoming)
        evidence_id = str(row["physical_evidence_id"])
        if evidence_id in unique:
            continue
        gap = (focal_time - _timestamp_dt(row["context_block_time"])).total_seconds()
        if not 0 < gap <= 86400 or str(row["context_tx_hash"]).lower() == focal_hash:
            continue
        if (
            str(row["from_address"]).lower() not in participant_set
            and str(row["to_address"]).lower() not in participant_set
        ):
            continue
        focal_asset = _optional_text(focal.get("focal_asset_local_id"))
        local_asset = _optional_text(row.get("asset_contract"))
        same_local = (
            local_asset is not None
            and focal_asset is not None
            and local_asset.lower() == focal_asset.lower()
        )
        focal_group = _optional_text(focal.get("focal_canonical_asset_group"))
        context_group = _optional_text(row.get("canonical_asset_group"))
        if same_local:
            same_canonical: bool | None = True
        elif (
            focal.get("focal_mapping_unknown")
            or focal_group is None
            or context_group is None
        ):
            same_canonical = None
        else:
            same_canonical = focal_group == context_group
        _, fungible = _asset_kind(row.get("asset_type"))
        same_family = fungible
        comparable = (
            row.get("amount_normalized") not in (None, "")
            if fungible
            else (
                row.get("asset_quantity") not in (None, "")
                if fungible is False
                else None
            )
        )
        row.update(
            {
                "time_gap_seconds": gap,
                "same_local_asset": same_local,
                "same_canonical_asset": same_canonical,
                "same_asset_family": same_family,
                "amount_semantics_comparable": comparable,
                "asset_match_unknown": same_canonical is None,
            }
        )
        unique[evidence_id] = row
    ordered = sorted(unique.values(), key=_rank_key)
    for rank, row in enumerate(ordered[:top_k], 1):
        row["event_rank"] = rank
        row["rank"] = rank
    return ordered[:top_k], len(ordered)
