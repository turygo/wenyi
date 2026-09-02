"""Pure node-state transitions used by :mod:`runstore`."""

from __future__ import annotations

from trans_novel.pipeline.state.models import (
    NODE_FAILED_PERMANENT,
    NODE_FAILED_RETRYABLE,
    NODE_POLISH,
    NODE_RUNNING,
    NODE_SKIPPED,
    NODE_SUCCEEDED,
    ChapterProgress,
    NodeFailure,
    NodeState,
    now_iso,
)


def _node(state, key: str) -> NodeState:
    node = state.nodes.get(key)
    if node is None:
        node = NodeState(node_id=key)
        state.nodes[key] = node
    return node


def mark_node_running(state, key: str) -> None:
    node = _node(state, key)
    node.status = NODE_RUNNING
    node.attempts += 1
    node.failure = None
    node.started_at = now_iso()
    node.finished_at = None


def record_node_fingerprint(state, key: str, fingerprint: str) -> None:
    node = _node(state, key)
    node.status = NODE_SUCCEEDED
    node.input_fingerprint = fingerprint
    node.finished_at = now_iso()


def record_node_output(state, key: str, output: dict) -> None:
    _node(state, key).output = dict(output)


def mark_node_succeeded(state, key: str, fingerprint: str | None = None) -> None:
    node = _node(state, key)
    node.status = NODE_SUCCEEDED
    if fingerprint:
        node.input_fingerprint = fingerprint
    node.finished_at = now_iso()


def mark_node_skipped(state, key: str) -> None:
    _node(state, key).status = NODE_SKIPPED
    node = state.nodes[key]
    node.finished_at = now_iso()
    base, separator, suffix = key.partition(":")
    if separator and suffix.isdigit() and base == NODE_POLISH:
        state.progress.setdefault(int(suffix), ChapterProgress()).pending_polish = []


def fail_node(state, key: str, kind: str, message: str = "") -> None:
    node = _node(state, key)
    node.status = NODE_FAILED_PERMANENT if kind == "provider_permanent" else NODE_FAILED_RETRYABLE
    node.failure = NodeFailure(kind=kind, message=message, at=now_iso())


__all__ = [
    "fail_node",
    "mark_node_running",
    "mark_node_skipped",
    "mark_node_succeeded",
    "record_node_fingerprint",
    "record_node_output",
]
