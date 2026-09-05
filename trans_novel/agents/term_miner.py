"""LLM-assisted terminology candidate mining."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent, retry_protocol
from trans_novel.glossary.miner import Candidate, merge_candidates, mine_candidates_en


class TermMiner(Agent):
    def mine(
        self,
        chapters: list[tuple[int, str]],
        *,
        concurrency: int = 1,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Candidate]:
        if not self.src.strip().lower().startswith("en"):
            return self._mine_llm(chapters, concurrency=concurrency, on_progress=on_progress)
        deterministic = mine_candidates_en(chapters)
        llm = self._mine_llm(chapters, concurrency=concurrency, on_progress=on_progress)
        return merge_candidates(deterministic, llm)

    def _mine_llm(
        self,
        chapters: list[tuple[int, str]],
        *,
        concurrency: int,
        on_progress: Callable[[int, int], None] | None,
    ) -> list[Candidate]:
        todo = [(ci, text) for ci, text in chapters if text.strip()]

        def mine_one(ci: int, text: str) -> list[str]:
            system = prompts.render("term_miner_system", src=self.src, tgt=self.tgt)
            user = prompts.render(
                "term_miner_user", src=self.src, tgt=self.tgt, chapter=ci, source=text[:8000]
            )
            raw = retry_protocol(
                lambda: self._ask_json(
                    system,
                    user,
                    key="candidates",
                    agent="preparer",
                    operation="prescan.term_mine",
                    strict=True,
                ),
                retries=self.config.pipeline.protocol_retry_limit,
            )
            return [s.strip() for s in raw or [] if isinstance(s, str) and s.strip()]

        results: dict[int, list[str]] = {}
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = {executor.submit(mine_one, ci, text): ci for ci, text in todo}
            for index, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                if on_progress:
                    on_progress(index, len(todo))

        candidates: dict[str, Candidate] = {}
        for ci, _ in todo:
            for surface in results.get(ci, []):
                candidate = candidates.setdefault(surface, Candidate(surface=surface))
                candidate.count += 1
                if ci not in candidate.chapters:
                    candidate.chapters.append(ci)
        out = list(candidates.values())
        out.sort(key=lambda candidate: candidate.count, reverse=True)
        return out
