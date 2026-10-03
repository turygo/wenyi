"""为显式格式迁移组合模型、独立状态副本和用量持久化。"""

from __future__ import annotations

import os

from trans_novel.pipeline.composition.agents import AgentBundle
from trans_novel.pipeline.composition.context import RunContext
from trans_novel.pipeline.nodes.reannotate import reannotate_store
from trans_novel.pipeline.nodes.reannotation_fingerprints import rebase_migrated_content
from trans_novel.pipeline.state import IdentityMismatchError, RunStore
from trans_novel.pipeline.state.clone import clone_closed_runstore
from trans_novel.pipeline.state.models import TRANSLATION_POLICY_VERSION, NodeState, stable_digest


def ensure_current_policy(store: RunStore) -> None:
    """旧策略必须先显式迁移，不能自动重新翻译或信任旧槽位。"""
    if (
        store.exists()
        and store.load_state().identity.translation_policy_version != TRANSLATION_POLICY_VERSION
    ):
        raise IdentityMismatchError(
            "翻译策略版本不一致；EPUB 可用 tools reannotate --state-copy 建立格式标注副本，"
            "原有译文保持不变。"
        )


def reannotate_existing(application, store, input_path, destination, *, progress=None) -> RunStore:
    """克隆封闭状态后标注；已有副本只能在来源证明一致时续跑。"""
    origin = {
        "version": 1,
        "run_dir": os.path.realpath(store.run_dir),
        "manifest_sha256": stable_digest(store.read_json(store.manifest_path)),
    }
    if os.path.realpath(destination) == origin["run_dir"]:
        raise ValueError("rich annotation requires an independent state copy")
    provenance = os.path.join(destination, "richtext_origin.json")
    if os.path.exists(destination):
        if not os.path.isfile(provenance) or store.read_json(provenance) != origin:
            raise ValueError("rich annotation copy does not match its original run")
    else:
        clone_closed_runstore(store.run_dir, destination)
        store.write_json(provenance, origin)
    migrated = RunStore(destination)
    with migrated.lock():
        application.bind_usage(migrated, scope="richtext")
        try:
            state = migrated.load_state()
            bundle = AgentBundle(
                client=application.client,
                config=application.config,
                src=state.identity.source_lang,
                tgt=state.identity.target_lang,
            )
            report = reannotate_store(
                migrated, input_path, annotator=bundle.annotator, progress=progress
            )
            migrated.write_json(migrated.path_for("richtext_report.json"), report)
            context = RunContext(
                store=migrated,
                config=application.config,
                doc=None,
                agent_builder=lambda src, tgt: bundle,
                output=application.config.output,
            )
            try:
                rebase_migrated_content(application.config, migrated, context)
            finally:
                context.close()
            state = migrated.load_state()
            state.nodes["assemble"] = NodeState(node_id="assemble")
            migrated.save_state(state)
            migrated.log_event("richtext_migration_completed", **report)
        finally:
            application.finish_usage("richtext")
    return migrated


__all__ = ["ensure_current_policy", "reannotate_existing"]
