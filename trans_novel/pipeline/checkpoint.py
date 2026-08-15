"""崩溃一致性检查点日志：翻译/润色批次的二阶段提交协议。

问题：V2 把译文（chapters_v2 章节文件）与 pending_polish 标记（manifest）分成
两次原子写。单独靠写序无法同时保证两条不变量：
  (a) 已落盘的译文必须有对应的润色标记 —— 否则崩溃后续跑把该批当“已译完”跳过，
      永不润色；
  (b) 已落盘的润色结果不能残留活标记 —— 否则同一段会被润色第二次。
协议（每批一次二阶段提交，批次间串行，故 journal 只保留单条在途记录）：
  翻译批（polish_on）：
    begin_translate(chapter, start, count) → save_chapter（译文） →
    save_progress（标记）→ clear
  润色批（PolishNode 排干）：
    begin_polish(chapter, start, count, polished) → save_chapter（润色后译文） →
    save_progress（清标记）→ clear
崩溃恢复在运行锁内执行（与 running→pending 恢复同位置，幂等）：
  translate 记录：标记在 → 已提交，清记录；标记缺 → 译文在则补标记（不变量 a），
    译文不在则清记录（未提交，批按未译重跑）；
  polish 记录：标记缺 → 已提交，清记录；标记在 → 译文等于记录值则清标记
    （不变量 b），否则清记录（未提交，保留标记由润色节点重跑该批）。
journal.json 由 RunStore 原子写入/删除；恢复完成后日志写入 events.jsonl。
"""

from __future__ import annotations

from trans_novel.pipeline.state import PolishBatch


def begin_translate(store, ci: int, start: int, count: int) -> None:
    """翻译批提交前记录在途批次（译文将写入章节文件、标记将写入 manifest）。"""
    store.save_journal({"kind": "translate", "chapter": ci, "start": start, "count": count})


def begin_polish(store, ci: int, start: int, count: int, polished: list[str]) -> None:
    """润色批提交前记录在途批次（润色结果将写入章节文件、标记将从 manifest 移除）。"""
    store.save_journal(
        {"kind": "polish", "chapter": ci, "start": start, "count": count, "targets": polished}
    )


def begin_naturalize(store, ci: int, applied: list[tuple[int, str]]) -> None:
    """去翻译腔提交前记录在途改写（改写将写入章节文件、naturalized 标记将写入 manifest）。"""
    store.save_journal(
        {
            "kind": "naturalize",
            "chapter": ci,
            "entries": [[idx, after] for idx, after in applied],
        }
    )


def clear(store) -> None:
    """两处写都完成后清除在途记录。"""
    store.clear_journal()


def recover(store) -> None:
    """在运行锁内重放在途记录：补齐缺失标记或清除已提交标记。幂等。"""
    record = store.load_journal()
    if record is None:
        return
    kind = record.get("kind")
    ci = record.get("chapter")
    start = record.get("start")
    count = record.get("count")
    if kind not in ("translate", "polish", "naturalize") or not isinstance(ci, int):
        store.clear_journal()
        return
    if kind == "naturalize":
        _recover_naturalize(store, ci, record)
        return
    if not isinstance(start, int) or not isinstance(count, int) or count <= 0:
        store.clear_journal()
        return
    progress = store.load_progress(ci)
    marker = next(
        (p for p in progress.pending_polish if p.start == start and p.count == count), None
    )
    if kind == "translate":
        if marker is not None:
            # 译文与标记都已落盘：只清记录。
            store.log_event(
                "checkpoint_recovered",
                kind="translate",
                chapter=ci,
                start=start,
                action="committed",
            )
            store.clear_journal()
            return
        chapter = store.load_chapter(ci)
        segs = chapter.text_segments[start : start + count]
        if len(segs) == count and all((s.target or "").strip() for s in segs):
            # 崩溃窗口在 save_chapter 之后、save_progress 之前：补回标记（不变量 a）。
            progress.pending_polish.append(PolishBatch(start=start, count=count))
            store.save_progress(ci, progress)
            store.log_event(
                "checkpoint_recovered",
                kind="translate",
                chapter=ci,
                start=start,
                action="repaired_marker",
            )
        else:
            # 译文未落盘：未提交，批保持未译状态，续跑重译。
            store.log_event(
                "checkpoint_recovered",
                kind="translate",
                chapter=ci,
                start=start,
                action="rolled_back",
            )
        store.clear_journal()
        return
    # kind == "polish"
    if marker is None:
        # 标记已清：润色结果与清标记都已完成。
        store.log_event(
            "checkpoint_recovered", kind="polish", chapter=ci, start=start, action="committed"
        )
        store.clear_journal()
        return
    chapter = store.load_chapter(ci)
    segs = chapter.text_segments[start : start + count]
    polished = record.get("targets")
    if (
        isinstance(polished, list)
        and len(segs) == count
        and len(polished) == count
        and all((s.target or "") == p for s, p in zip(segs, polished, strict=False))
    ):
        # 崩溃窗口在 save_chapter（润色结果）之后、save_progress（清标记）之前：
        # 清除标记（不变量 b），避免同一段被润色两次。
        progress.pending_polish = [p for p in progress.pending_polish if p.start != start]
        store.save_progress(ci, progress)
        store.log_event(
            "checkpoint_recovered",
            kind="polish",
            chapter=ci,
            start=start,
            action="repaired_marker",
        )
    else:
        # 润色结果未落盘：保留标记，润色节点下次重跑该批。
        store.log_event(
            "checkpoint_recovered",
            kind="polish",
            chapter=ci,
            start=start,
            action="rolled_back",
        )
    store.clear_journal()


def _recover_naturalize(store, ci: int, record: dict) -> None:
    """naturalize 记录：改写写章节文件、标记写 manifest，两次独立原子写。

    崩溃窗口在 save_chapter（改写）之后、set_naturalized（标记）之前：章节里的
    改写已落盘但标记缺失 → 恢复补标记（完成事务，不再重复去翻译腔）；改写未落盘
    （与记录不一致）→ 回滚，下次 run 重新自然化。
    """
    progress = store.load_progress(ci)
    if progress.naturalized:
        store.log_event("checkpoint_recovered", kind="naturalize", chapter=ci, action="committed")
        store.clear_journal()
        return
    raw_entries = record.get("entries")
    if isinstance(raw_entries, list) and raw_entries:
        chapter = store.load_chapter(ci)
        segs = chapter.text_segments
        if all(
            isinstance(e, list)
            and len(e) == 2
            and isinstance(e[0], int)
            and 0 <= e[0] < len(segs)
            and (segs[e[0]].target or "") == e[1]
            for e in raw_entries
        ):
            # 全部改写已落盘：补标记（完成事务），不重放改写。
            progress.naturalized = True
            store.save_progress(ci, progress)
            store.log_event(
                "checkpoint_recovered",
                kind="naturalize",
                chapter=ci,
                action="repaired_marker",
            )
            store.clear_journal()
            return
    # 改写未落盘：回滚，保持未自然化，下次 run 重跑。
    store.log_event("checkpoint_recovered", kind="naturalize", chapter=ci, action="rolled_back")
    store.clear_journal()


__all__ = ["begin_naturalize", "begin_polish", "begin_translate", "clear", "recover"]
