from __future__ import annotations

import copy
import json
import re
import threading
from collections import Counter
from typing import Any, ClassVar

from tests.fixtures.migration_artifacts import canonical_json, sha256_bytes
from tests.fixtures.migration_epub import (
    FIXTURE_ID,
    MODEL,
    RESPONSE_DEFINITIONS,
    SOURCE_TARGETS,
    TITLE_TARGETS,
)
from tests.fixtures.migration_responses import _annotation_runs, _naming_candidates


class FixtureSDK:
    def __init__(self, boundary: RequestBoundary) -> None:
        self.boundary = boundary
        self._lock = threading.Lock()
        self._bound: dict[tuple[str, str], dict[str, Any]] = {}
        self._consumed: set[tuple[str, str]] = set()
        self.chat = type("Chat", (), {"completions": self})()

    def bind(self, model: str, messages: list[dict[str, str]], payload: str) -> str:
        digest = _messages_sha256(messages)
        key = (model, digest)
        with self._lock:
            if key in self._bound:
                raise AssertionError("duplicate model/messages request identity")
            self._bound[key] = {
                "content": payload,
                "usage": copy.deepcopy(RESPONSE_DEFINITIONS["token_usage"]),
            }
        return digest

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        messages = kwargs.get("messages")
        if not isinstance(model, str) or not isinstance(messages, list):
            raise AssertionError("SDK received an invalid model request")
        key = (model, _messages_sha256(messages))
        with self._lock:
            bound = self._bound.get(key)
            if bound is None or key in self._consumed:
                raise AssertionError("SDK request does not match one unique validated binding")
            self._consumed.add(key)
        usage = bound["usage"]
        response = type("SDKResponse", (), {})()
        response.id = f"synthetic-{key[1][:16]}"
        response.model = model
        response.usage = type("SDKUsage", (), usage)()
        response.choices = [
            type(
                "SDKChoice",
                (),
                {
                    "finish_reason": "stop",
                    "message": type("SDKMessage", (), {"content": bound["content"]})(),
                },
            )()
        ]
        return response

    def assert_all_consumed(self) -> None:
        if set(self._bound) != self._consumed:
            raise AssertionError("validated SDK responses were not consumed exactly once")


class RequestBoundary:
    _CONTRACTS: ClassVar = {
        "analyzer.analyze": {("analyst", "Analyzer", True, None)},
        "chapter.classify": {("analyst", "ChapterClassifier", True, 500)},
        "prescan.term_mine": {("preparer", "TermMiner", True, None)},
        "prescan.name_terms": {("analyst", "CastNamer", True, None)},
        "translate.batch": {("translator", "Translator", True, None)},
        "translate.heading": {("analyst", "Translator", False, None)},
        "richtext.annotate": {("analyst", "RichTextAnnotator", True, None)},
        "title.translate": {("analyst", "title_translate", True, None)},
        "glossary.extract": {("preparer", "GlossaryExtractor", True, None)},
    }

    def __init__(self, document: Any, analyzer_sample: str) -> None:
        self.document = document
        self.analyzer_sample = analyzer_sample
        self.sdk = FixtureSDK(self)
        self.transport = None
        self.lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.source_calls: Counter[str] = Counter()
        self.heading_calls: Counter[str] = Counter()
        self.annotation_calls: Counter[str] = Counter()
        self.title_calls: Counter[str] = Counter()
        self.operations: Counter[str] = Counter()
        self.expected_sources: Counter[str] = Counter()
        self.expected_headings: Counter[str] = Counter(
            chapter.title for chapter in document.chapters if chapter.title
        )
        self.expected_title_sources = Counter(TITLE_TARGETS.keys())
        self.title_sources: Counter[str] = Counter()
        self.expected_rich: dict[str, dict[str, Any]] = {}
        self.expected_rich_calls: Counter[str] = Counter()
        self.chapter_sources: dict[int, str] = {}
        self.classification_sources: dict[int, str] = {}
        for chapter in document.chapters:
            self.chapter_sources[chapter.index] = "\n".join(
                segment.source for segment in chapter.text_segments
            )
            self.classification_sources[chapter.index] = "\n\n".join(
                segment.source for segment in chapter.text_segments
            )
            for segment in chapter.text_segments:
                source = segment.translation_source
                if source not in SOURCE_TARGETS:
                    raise AssertionError(
                        f"fixture has no frozen translation for source identity {source!r}"
                    )
                if segment.kind != "heading" and SOURCE_TARGETS[source] != source:
                    self.expected_sources[source] += 1
                state = segment.epub_state
                rich = state.rich_source if state is not None else None
                if (
                    rich is not None
                    and (rich.marks or rich.atoms)
                    and SOURCE_TARGETS[source] != source
                ):
                    raw = rich.model_dump(mode="json")
                    identity = sha256_bytes(canonical_json(raw))
                    self.expected_rich[identity] = {
                        "source": raw,
                        "target": SOURCE_TARGETS[source],
                    }
                    self.expected_rich_calls[identity] += 1

    def attach(self, transport: Any) -> None:
        self.transport = transport

    def complete(
        self,
        messages,
        model_ref,
        *,
        json_mode=False,
        max_tokens=None,
        stage=None,
        agent,
        operation,
        logical_call_id=None,
        attempt_counter=None,
    ):
        contract = (agent, stage, json_mode, max_tokens)
        if contract not in self._CONTRACTS.get(operation, set()):
            raise AssertionError(f"invalid operation contract: {operation} {contract!r}")
        if (
            model_ref.provider != "deepseek"
            or model_ref.model != MODEL
            or model_ref.reasoning_enabled != (agent != "translator")
        ):
            raise AssertionError(f"invalid model selection for {operation}")
        payload, identities = self._response(operation, messages, agent=agent)
        message_digest = self.sdk.bind(model_ref.model, messages, payload)
        with self.lock:
            self.operations[operation] += 1
        call = {
            "agent": agent,
            "operation": operation,
            "stage": stage,
            "provider": model_ref.provider,
            "model": model_ref.model,
            "reasoning_enabled": model_ref.reasoning_enabled,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
            "message_sha256": message_digest,
            "response_sha256": sha256_bytes(payload.encode("utf-8")),
            "identities": identities,
            "response": payload,
            "response_id": f"synthetic-{message_digest[:16]}",
            "logical_call_id": logical_call_id,
            "token_usage": copy.deepcopy(RESPONSE_DEFINITIONS["token_usage"]),
        }
        with self.lock:
            self.calls.append(call)
        if self.transport is None:
            raise AssertionError("the production OpenAI-compatible transport was not attached")
        return self.transport.complete(
            messages,
            model_ref,
            json_mode=json_mode,
            max_tokens=max_tokens,
            stage=stage,
            agent=agent,
            operation=operation,
            logical_call_id=logical_call_id,
            attempt_counter=attempt_counter,
        )

    def _response(self, operation: str, messages: Any, *, agent: str) -> tuple[str, list[str]]:
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [item.get("role") for item in messages] != ["system", "user"]
            or any(not isinstance(item.get("content"), str) for item in messages)
        ):
            raise AssertionError(f"invalid source-bound messages for {operation}")
        user = messages[1]["content"]
        if operation == "analyzer.analyze":
            if self.analyzer_sample not in user:
                raise AssertionError("analysis request is not bound to the synthetic source sample")
            result = copy.deepcopy(RESPONSE_DEFINITIONS["analysis"])
            return json.dumps(result, ensure_ascii=False), [
                sha256_bytes(self.analyzer_sample.encode())
            ]
        if operation == "chapter.classify":
            data = json.loads(user)
            chapter = data.get("chapter_id") if isinstance(data, dict) else None
            source = data.get("source") if isinstance(data, dict) else None
            if (
                type(chapter) is not int
                or chapter not in self.chapter_sources
                or not isinstance(source, str)
            ):
                raise AssertionError("classification request lacks a known chapter identity")
            if source != self.classification_sources[chapter]:
                raise AssertionError("classification request source differs from the known chapter")
            result = {"chapter_id": chapter, **RESPONSE_DEFINITIONS["chapter_classification"]}
            return json.dumps(result, ensure_ascii=False), [
                f"chapter:{chapter}",
                sha256_bytes(source.encode()),
            ]
        if operation == "prescan.term_mine":
            matches = [
                index for index, source in self.chapter_sources.items() if source and source in user
            ]
            if len(matches) != 1:
                raise AssertionError(
                    "term-mining request is not bound to one complete source chapter"
                )
            return json.dumps(
                {"candidates": RESPONSE_DEFINITIONS["term_candidates"]},
                ensure_ascii=False,
            ), [f"chapter:{matches[0]}"]
        if operation == "prescan.name_terms":
            sources = _naming_candidates(user, self.chapter_sources)
            return json.dumps(
                {"terms": RESPONSE_DEFINITIONS["named_terms"]},
                ensure_ascii=False,
            ), [sha256_bytes(source.encode()) for source in sources]
        if operation == "translate.heading":
            source = _heading_source(user)
            if source not in self.expected_headings:
                raise AssertionError("heading request includes an unknown title source")
            with self.lock:
                self.heading_calls[source] += 1
            return TITLE_TARGETS[source], [sha256_bytes(source.encode())]
        if operation == "translate.batch":
            sources = _numbered_sources(user)
            if not sources or any(source not in SOURCE_TARGETS for source in sources):
                raise AssertionError("translation batch includes an unknown source segment")
            with self.lock:
                self.source_calls.update(sources)
            targets = [SOURCE_TARGETS[source] for source in sources]
            return json.dumps({"translations": targets}, ensure_ascii=False), [
                sha256_bytes(item.encode()) for item in sources
            ]
        if operation == "richtext.annotate":
            return self._annotation_response(user)
        if operation == "title.translate":
            return self._title_response(user)
        if operation == "glossary.extract":
            sources = [
                source
                for source, target in SOURCE_TARGETS.items()
                if source in user and target in user
            ]
            if not sources:
                raise AssertionError("glossary request is not bound to translated fixture content")
            return json.dumps(
                {"terms": RESPONSE_DEFINITIONS["glossary_terms"]},
                ensure_ascii=False,
            ), [sha256_bytes(item.encode()) for item in sources]
        raise AssertionError(f"unimplemented fixture operation: {operation}")

    def _annotation_response(self, user: str) -> tuple[str, list[str]]:
        request = json.loads(user)
        records = request.get("segments") if isinstance(request, dict) else None
        if not isinstance(records, list) or not records:
            raise AssertionError("rich annotation request has no source-bound segments")
        response_records = []
        identities = []
        for record in records:
            if not isinstance(record, dict) or type(record.get("id")) is not int:
                raise AssertionError("rich annotation record has an invalid identity")
            source = record.get("source")
            target = record.get("target")
            identity = sha256_bytes(canonical_json(source))
            expected = self.expected_rich.get(identity)
            if expected is None or source != expected["source"] or target != expected["target"]:
                raise AssertionError("rich annotation source/target identity is not admitted")
            with self.lock:
                self.annotation_calls[identity] += 1
            response_records.append({"id": record["id"], "runs": _annotation_runs(source, target)})
            identities.append(identity)
        return json.dumps({"annotated": response_records}, ensure_ascii=False), identities

    def _title_response(self, user: str) -> tuple[str, list[str]]:
        match = re.search(
            r"【全书有序标题体系（JSON）】\n(.*?)\n\n输出 JSON",
            user,
            re.S,
        )
        payload = json.loads(match.group(1)) if match else None
        records = payload.get("titles") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            raise AssertionError("title translation request has no stable catalog records")
        results = []
        identities = []
        seen_ids = set()
        sources = []
        for record in records:
            source = record.get("source") if isinstance(record, dict) else None
            identifier = record.get("id") if isinstance(record, dict) else None
            if (
                not isinstance(source, str)
                or source not in TITLE_TARGETS
                or not isinstance(identifier, str)
                or identifier in seen_ids
            ):
                raise AssertionError(
                    "title catalog contains an unknown or duplicate source identity"
                )
            seen_ids.add(identifier)
            sources.append(source)
            results.append({"id": identifier, "target": TITLE_TARGETS[source]})
            identities.append(sha256_bytes(canonical_json(record)))
        if Counter(sources) != self.expected_title_sources:
            raise AssertionError("title request does not contain the exact frozen title catalog")
        with self.lock:
            self.title_calls.update(seen_ids)
            self.title_sources.update(sources)
        return json.dumps({"titles": results}, ensure_ascii=False), identities

    def assert_complete(self) -> None:
        self.sdk.assert_all_consumed()
        required = {
            "analyzer.analyze",
            "chapter.classify",
            "prescan.term_mine",
            "translate.heading",
            "translate.batch",
            "richtext.annotate",
            "title.translate",
        }
        if not required <= self.operations.keys():
            raise AssertionError(
                f"full-flow model boundary missed operations: {sorted(required - self.operations.keys())}"
            )
        if self.source_calls != self.expected_sources:
            raise AssertionError(
                "the model boundary did not consume every synthetic source exactly once"
            )
        if self.heading_calls != self.expected_headings:
            raise AssertionError(
                "the model boundary did not translate every chapter title exactly once"
            )
        if self.annotation_calls != self.expected_rich_calls:
            raise AssertionError("rich source/target annotations were missing or duplicated")
        if self.title_sources != self.expected_title_sources:
            raise AssertionError(
                "the title catalog was incomplete, duplicated or source-mismatched"
            )
        if any(count != 1 for count in self.title_calls.values()):
            raise AssertionError("title catalog entries were duplicated")

    def evidence(self) -> dict[str, Any]:
        calls = sorted(self.calls, key=lambda item: (item["operation"], item["message_sha256"]))
        return {
            "schema_version": 1,
            "fixture_id": FIXTURE_ID,
            "model": MODEL,
            "response_definitions": copy.deepcopy(RESPONSE_DEFINITIONS),
            "response_definitions_sha256": sha256_bytes(canonical_json(RESPONSE_DEFINITIONS)),
            "calls": calls,
        }


class RecordingTransport:
    def __init__(self, transport: Any, boundary: RequestBoundary) -> None:
        self.transport = transport
        self.boundary = boundary
        self.provider = "deepseek"
        self.usage = transport.usage

    def capabilities_for(self, model: str) -> Any:
        return self.transport.capabilities_for(model)

    def complete(self, *args: Any, **kwargs: Any) -> str:
        return self.boundary.complete(*args, **kwargs)


class RejectingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        from trans_novel.llm.usage import UsageTracker

        self.usage = UsageTracker()

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        raise AssertionError("source-first assembly must not call a model")

    def complete_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        raise AssertionError("source-first assembly must not call a model")


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    return sha256_bytes(canonical_json(messages))


def _numbered_sources(user: str) -> list[str]:
    marker = "【待译英文段落】"
    if marker not in user:
        raise AssertionError("translation request lacks the source-segment boundary")
    body = user.split(marker, 1)[1].split("\n", 1)[1]
    body = body.split("\n\n请翻译以上每一段", 1)[0]
    sources: list[str] = []
    current: list[str] | None = None
    for line in body.splitlines():
        match = re.match(r"^\[\d+\] (.*)$", line)
        if match:
            if current is not None:
                sources.append("\n".join(current))
            current = [match.group(1)]
        elif current is not None:
            current.append(line)
    if current is not None:
        sources.append("\n".join(current))
    return sources


def _heading_source(user: str) -> str:
    match = re.search(
        r"【待译章节标题】\n(.*?)\n\n请只输出标题译文",
        user,
        re.S,
    )
    if match is None:
        raise AssertionError("heading request lacks the expected source-bound title")
    return match.group(1)
