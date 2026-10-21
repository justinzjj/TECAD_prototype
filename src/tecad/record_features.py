"""Frozen execution/context descriptor extraction used in record-level replay."""

from __future__ import annotations
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Invalid normalized observation or feature schema."""


D_ASSET_VOCAB = ("fungible_token", "native", "__UNK__")

D_MOVEMENT_VOCAB = ("mint", "transfer", "__UNK__")

S_EXEC_NODE_VOCAB = (
    "address_account",
    "asset",
    "contract",
    "event_log",
    "event_transfer",
    "transaction",
    "__UNK__",
)

S_EXEC_ASSET_VOCAB = ("fungible_token", "__UNK__")

S_EXEC_MOVEMENT_VOCAB = ("burn", "mint", "transfer", "__UNK__")

S_CTX_NODE_VOCAB = (
    "address_account",
    "asset",
    "context_transaction",
    "physical_event",
    "__UNK__",
)

S_CTX_ASSET_VOCAB = (
    "fungible_token",
    "multi_token",
    "native",
    "non_fungible_token",
    "__UNK__",
)

S_CTX_MOVEMENT_VOCAB = ("burn", "mint", "self", "transfer", "__UNK__")

S_CTX_ALLOWED_NODE_TYPES = frozenset(S_CTX_NODE_VOCAB[:-1])

S_CTX_EXCLUDED_NODE_TYPES = frozenset(
    ("focal_transaction", "context_state", "state", "anchor")
)

S_EXEC_NUMERIC_FIELDS = (
    "exec_success",
    "exec_native_value_log",
    "exec_gas_log",
    "exec_log_count_log",
    "exec_transfer_count_log",
    "exec_mint_ratio",
    "exec_burn_ratio",
    "exec_self_ratio",
)

S_CTX_NUMERIC_FIELDS = (
    "ctx_in_tx_log",
    "ctx_out_tx_log",
    "ctx_counterparty_log",
    "ctx_fan_in_log",
    "ctx_fan_out_log",
    "ctx_reciprocal_log",
    "ctx_context_count_log",
    "ctx_asset_diversity_log",
    "ctx_recency_min_log",
    "ctx_recency_p25_log",
    "ctx_recency_p50_log",
    "ctx_recency_p75_log",
    "ctx_recency_max_log",
    "ctx_interarrival_mean_log",
    "ctx_interarrival_std_log",
    "ctx_hhi",
)


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _true(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(value)


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(
        str(value)
        for value in values
        if value is not None and not pd.isna(value) and str(value) != ""
    )
    return dict(sorted(counts.items()))


def category_proportions(
    counts: Mapping[str, int],
    vocabulary: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    vocabulary = tuple(vocabulary)
    if not vocabulary or vocabulary[-1] != "__UNK__":
        raise ContractError("vocabulary_requires_terminal_unknown")
    known = set(vocabulary[:-1])
    normalized = {
        str(key): int(value) for key, value in counts.items() if int(value) > 0
    }
    total = sum(normalized.values())
    if total == 0:
        return np.zeros(len(vocabulary), dtype=np.float64), np.zeros(
            len(vocabulary), dtype=bool
        )
    unknown = sum(value for key, value in normalized.items() if key not in known)
    output = [normalized.get(category, 0) / total for category in vocabulary[:-1]]
    output.append(unknown / total)
    return np.asarray(output, dtype=np.float64), np.ones(len(vocabulary), dtype=bool)


def extract_target_d(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not {"node_type", "is_root"}.issubset(nodes.columns):
        raise ContractError("target_execution_schema_missing")
    roots = nodes[nodes["is_root"].fillna(False).astype(bool)]
    if len(roots) != 1:
        raise ContractError(f"target_execution_root_count:{len(roots)}")
    root = roots.iloc[0]
    bindings = (
        ("transaction_success", "transaction_success_available"),
        ("transaction_value_log1p", "transaction_value_available"),
        ("gas_used_log1p", "gas_used_available"),
        ("event_count_log1p", None),
        ("transfer_count_log1p", None),
    )
    numeric: list[float] = []
    numeric_valid: list[bool] = []
    for source, mask_name in bindings:
        valid = source in nodes.columns and _finite(root.get(source))
        if mask_name is not None:
            valid = valid and _true(root.get(mask_name))
        numeric.append(float(root.get(source)) if valid else 0.0)
        numeric_valid.append(bool(valid))
    movements = nodes[nodes["node_type"].eq("event_transfer")].copy()
    for flag in ("mint_flag", "burn_flag", "self_flag"):
        observed = (
            pd.to_numeric(movements[flag], errors="coerce").to_numpy(dtype=np.float64)
            if flag in movements.columns
            else np.empty(0, dtype=np.float64)
        )
        observed = observed[np.isfinite(observed)]
        numeric.append(float(observed.mean()) if observed.size else 0.0)
        numeric_valid.append(bool(observed.size))
    asset_values = []
    if "asset_kind" in nodes.columns:
        for _, row in nodes.iterrows():
            if _true(row.get("asset_kind_known")) and not pd.isna(
                row.get("asset_kind")
            ):
                asset_values.append(row.get("asset_kind"))
    movement_values = movements.get(
        "movement_direction", pd.Series(dtype=object)
    ).tolist()
    asset, asset_valid = category_proportions(_counter(asset_values), D_ASSET_VOCAB)
    movement, movement_valid = category_proportions(
        _counter(movement_values), D_MOVEMENT_VOCAB
    )
    values = np.concatenate((np.asarray(numeric), asset, movement)).astype(np.float64)
    valid = np.concatenate(
        (np.asarray(numeric_valid), asset_valid, movement_valid)
    ).astype(bool)
    if values.shape != (14,):
        raise ContractError(f"D_dimension_mismatch:{values.shape}")
    return values, valid


def _graph_nodes(graph: Any) -> list[Mapping[str, Any]]:
    if not isinstance(graph, Mapping):
        return []
    return [node for node in graph.get("nodes", []) if isinstance(node, Mapping)]


def _categorical_block(
    nodes: Sequence[Mapping[str, Any]],
    field: str,
    vocabulary: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    return category_proportions(_counter(node.get(field) for node in nodes), vocabulary)


def extract_candidate_modules(
    feature_row: Mapping[str, Any],
    coverage_row: Mapping[str, Any],
    graph_pair: Sequence[Mapping[str, Any]] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    exec_complete = _true(coverage_row.get("execution_complete"))
    ctx_complete = _true(coverage_row.get("context_complete"))
    if graph_pair is None or len(graph_pair) != 2:
        exec_nodes: list[Mapping[str, Any]] = []
        ctx_nodes: list[Mapping[str, Any]] = []
    else:
        exec_nodes = _graph_nodes(graph_pair[0])
        ctx_nodes = _graph_nodes(graph_pair[1])

    exec_values = np.asarray(
        [
            float(feature_row.get(field)) if _finite(feature_row.get(field)) else 0.0
            for field in S_EXEC_NUMERIC_FIELDS
        ],
        dtype=np.float64,
    )
    exec_valid = np.asarray(
        [
            exec_complete and _finite(feature_row.get(field))
            for field in S_EXEC_NUMERIC_FIELDS
        ],
        dtype=bool,
    )
    exec_parts = [(exec_values, exec_valid)]
    for field, vocab in (
        ("node_type", S_EXEC_NODE_VOCAB),
        ("asset_kind", S_EXEC_ASSET_VOCAB),
        ("movement_direction", S_EXEC_MOVEMENT_VOCAB),
    ):
        values, valid = _categorical_block(exec_nodes, field, vocab)
        exec_parts.append((values, valid & exec_complete))

    allowed_ctx_nodes = [
        node
        for node in ctx_nodes
        if str(node.get("node_type")) in S_CTX_ALLOWED_NODE_TYPES
        and str(node.get("node_type")) not in S_CTX_EXCLUDED_NODE_TYPES
    ]
    ctx_values = [
        float(feature_row.get(field)) if _finite(feature_row.get(field)) else 0.0
        for field in S_CTX_NUMERIC_FIELDS
    ]
    ctx_valid = [
        ctx_complete and _finite(feature_row.get(field))
        for field in S_CTX_NUMERIC_FIELDS
    ]
    relative = [
        float(node.get("relative_time_log"))
        for node in allowed_ctx_nodes
        if _finite(node.get("relative_time_log"))
    ]
    ctx_values.append(float(np.mean(relative)) if relative else 0.0)
    ctx_valid.append(bool(ctx_complete and relative))
    ctx_parts: list[tuple[np.ndarray, np.ndarray]] = [
        (np.asarray(ctx_values, dtype=np.float64), np.asarray(ctx_valid, dtype=bool))
    ]
    for field, vocab in (
        ("node_type", S_CTX_NODE_VOCAB),
        ("asset_kind", S_CTX_ASSET_VOCAB),
        ("movement_direction", S_CTX_MOVEMENT_VOCAB),
    ):
        values, valid = _categorical_block(allowed_ctx_nodes, field, vocab)
        ctx_parts.append((values, valid & ctx_complete))

    s_exec = np.concatenate([part[0] for part in exec_parts]).astype(np.float64)
    s_exec_valid = np.concatenate([part[1] for part in exec_parts]).astype(bool)
    s_ctx = np.concatenate([part[0] for part in ctx_parts]).astype(np.float64)
    s_ctx_valid = np.concatenate([part[1] for part in ctx_parts]).astype(bool)
    if s_exec.shape != (21,) or s_ctx.shape != (32,):
        raise ContractError(
            f"candidate_dimension_mismatch:{s_exec.shape}:{s_ctx.shape}"
        )
    s_exec[~s_exec_valid] = 0.0
    s_ctx[~s_ctx_valid] = 0.0
    return {"S_exec": (s_exec, s_exec_valid), "S_ctx": (s_ctx, s_ctx_valid)}
