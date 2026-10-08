from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from tests.fixtures.migration_admission import _portable_path
from tests.fixtures.migration_artifacts import sha256_file
from tests.fixtures.migration_epub_observer import observe_package as _package
from tests.fixtures.migration_errors import ObservationError
from tests.fixtures.migration_run_observer import _json_object, _state_observation, _workspace_file

_ROLES = {"mono", "bilingual_target_first", "bilingual_source_first"}


def _shape_text(tree: dict[str, Any]) -> str:
    if "tn-source" in tree.get("attributes", {}).get("class", "").split():
        return ""
    return (tree.get("text") or "") + "".join(
        _shape_text(child) + (child.get("tail") or "") for child in tree.get("children", [])
    )


def _validate_bilingual_order(
    package: dict[str, Any], segments: list[dict[str, Any]], order: str
) -> None:
    blocks = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "caption"}
    for resource in package["spine"]:
        for copied in package["source_copies"][resource]:
            matches = [
                segment
                for segment in segments
                if segment["resource_href"] == resource and segment["source"] in copied["text"]
            ]
            if not matches:
                raise ObservationError("bilingual source copy has no persisted source identity")
            segment = max(matches, key=lambda item: len(item["source"]))
            parent = package["html"][resource]["tree"]
            for index in copied["path"][:-1]:
                parent = parent["children"][index]
            position = copied["path"][-1]
            children = parent["children"]
            tag = copied["tree"]["tag"].rsplit("}", 1)[-1]
            if tag in blocks:
                neighbor = position + (1 if order == "source_first" else -1)
                if not 0 <= neighbor < len(children):
                    raise ObservationError("bilingual source copy has no adjacent target")
                text = _shape_text(children[neighbor])
            elif order == "source_first":
                text = (children[position].get("tail") or "") + "".join(
                    _shape_text(child) + (child.get("tail") or "")
                    for child in children[position + 1 :]
                )
            else:
                text = (parent.get("text") or "") + "".join(
                    _shape_text(child) + (child.get("tail") or "") for child in children[:position]
                )
            if segment["target"] not in text:
                raise ObservationError(
                    f"bilingual source/target pairing order differs from {order}"
                )
            copied["order"] = order
            copied["paired_target"] = segment["target"]


def _validate_rendered_richness(
    segments: list[dict[str, Any]],
    package: dict[str, Any],
) -> None:
    semantic_tags = {
        "bold": {"b", "strong"},
        "italic": {"i", "em"},
        "link": {"a"},
        "superscript": {"sup"},
        "subscript": {"sub"},
        "literal": {"code", "kbd", "samp", "pre"},
    }
    for segment in segments:
        source = segment.get("rich_source")
        target = segment.get("rich_target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            if isinstance(source, dict) and (source.get("marks") or source.get("atoms")):
                raise ObservationError("persisted rich source has no accepted rich target")
            continue
        resource = segment.get("resource_href")
        observed = package["html"].get(resource)
        if not isinstance(observed, dict):
            raise ObservationError(f"rich segment resource is absent from EPUB output: {resource}")
        elements = observed["target_elements"]
        marks = {
            mark.get("id"): mark
            for mark in source.get("marks", [])
            if isinstance(mark, dict) and isinstance(mark.get("id"), str)
        }
        atoms = {
            atom.get("id"): atom
            for atom in source.get("atoms", [])
            if isinstance(atom, dict) and isinstance(atom.get("id"), str)
        }
        runs = target.get("runs", [])
        if not isinstance(runs, list):
            raise ObservationError("persisted rich target has an invalid run list")
        target_atoms = Counter(
            run.get("atom")
            for run in runs
            if isinstance(run, dict) and isinstance(run.get("atom"), str)
        )
        if target_atoms != Counter(dict.fromkeys(atoms, 1)):
            raise ObservationError("rich target does not preserve the source atom inventory")
        breaks = sum(atom.get("kind") == "linebreak" for atom in atoms.values())
        images = sum(atom.get("kind") == "media" for atom in atoms.values())
        if sum(element.get("tag") == "br" for element in elements) < breaks:
            raise ObservationError("rich target omits a source linebreak object")
        if sum(element.get("tag") == "img" for element in elements) < images:
            raise ObservationError("rich target omits a source media object")
        for run in runs:
            if not isinstance(run, dict):
                raise ObservationError("persisted rich target contains an invalid run")
            text = run.get("text", "")
            if not isinstance(text, str):
                raise ObservationError("persisted rich target contains non-text run content")
            for mark_id in run.get("marks", []):
                mark = marks.get(mark_id)
                if mark is None:
                    raise ObservationError(
                        f"rich target refers to an unknown source mark: {mark_id}"
                    )
                tag = str(mark.get("tag", "")).rsplit("}", 1)[-1].lower()
                semantic = set(mark.get("semantics", []))
                allowed = set().union(*(semantic_tags.get(role, set()) for role in semantic))
                allowed = allowed or {tag}
                attributes = mark.get("attributes", {})
                matches = [
                    element
                    for element in elements
                    if element.get("tag") in allowed
                    and all(
                        element.get("attributes", {}).get(key) == value
                        for key, value in attributes.items()
                    )
                    and (not text or text in element.get("text", ""))
                ]
                if not matches:
                    raise ObservationError(f"rendered EPUB lost rich mark {mark_id} ({tag})")


def _semantic_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in resource.items() if key != "size_bytes"}
        for resource in resources
    ]


def _validate_preserved_images(source: dict[str, Any], output: dict[str, Any]) -> None:
    expected = {
        resource["path"]: resource.get("sha256")
        for resource in source["resources"]
        if resource.get("media_type", "").startswith("image/")
    }
    actual = {
        resource["path"]: resource.get("sha256")
        for resource in output["resources"]
        if resource.get("media_type", "").startswith("image/")
    }
    if any(actual.get(path) != digest for path, digest in expected.items()):
        raise ObservationError(
            "EPUB output did not preserve all source image resources byte-for-byte"
        )


def _verify_output_record(
    workspace: Path,
    output: dict[str, Any],
    expected_segments: list[dict[str, Any]],
    source_package: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    role = output.get("role")
    if role not in _ROLES:
        raise ObservationError(f"unknown output role: {role}")
    file_path = _portable_path(output.get("file"))
    proof_path = _portable_path(output.get("proof"))
    receipt_path = _portable_path(output.get("receipt"))
    epub_path = _workspace_file(workspace, file_path)
    proof = _json_object(_workspace_file(workspace, proof_path))
    receipt = _json_object(_workspace_file(workspace, receipt_path))
    package = _package(epub_path)
    label = epub_path.name
    labels = proof.get("published_outputs")
    if not isinstance(labels, dict) or label not in labels:
        raise ObservationError(f"published output receipt is absent for {label}")
    evidence = labels[label]
    archive_sha256 = sha256_file(epub_path)
    if (
        not isinstance(evidence, dict)
        or evidence.get("passed") is not True
        or evidence.get("published") is not True
        or evidence.get("output_sha256") != archive_sha256
    ):
        raise ObservationError(f"persisted EPUB proof or receipt is invalid for {label}")
    receipt_evidence = receipt.get("published_outputs", {}).get(label)
    if receipt_evidence != evidence:
        raise ObservationError(f"publication receipt disagrees with proof for {label}")
    bilingual = role != "mono"
    order = "source_first" if role == "bilingual_source_first" else "target_first"
    copies = [
        copy for resource in package["spine"] for copy in package["source_copies"].get(resource, [])
    ]
    if bilingual and not copies:
        raise ObservationError(f"bilingual output has no source-copy evidence: {role}")
    if not bilingual and copies:
        raise ObservationError("monolingual output unexpectedly contains source copies")
    if bilingual:
        _validate_bilingual_order(package, expected_segments, order)
    if bilingual and any(copy["order"] != order for copy in copies):
        raise ObservationError(f"bilingual source/target order does not match {role}")
    source_copy_text = "\n".join(copy["text"] for copy in copies)
    expected_sources = [
        segment["source"]
        for segment in expected_segments
        if segment.get("kind") != "heading"
        and isinstance(segment.get("source"), str)
        and segment["source"] != segment["target"]
    ]
    if bilingual:
        source_cursor = 0
        for source in expected_sources:
            position = source_copy_text.find(source, source_cursor)
            if position < 0:
                raise ObservationError(
                    f"bilingual EPUB does not preserve ordered source segment: {source!r}"
                )
            source_cursor = position + len(source)
    targets = [segment["target"] for segment in expected_segments]
    output_text = "\n".join(
        text for resource in package["spine"] for text in package["target_blocks"].get(resource, [])
    )
    target_cursor = 0
    for target in targets:
        position = output_text.find(target, target_cursor)
        if position < 0:
            raise ObservationError(
                "reopened EPUB target order or text differs from persisted state"
            )
        target_cursor = position + len(target)
    _validate_rendered_richness(expected_segments, package)
    _validate_preserved_images(source_package, package)
    stable_proof = {key: value for key, value in evidence.items() if key != "output_sha256"}
    observed = {
        "role": role,
        "bilingual": bilingual,
        "order": order if bilingual else None,
        "output_label": label,
        "proof": stable_proof,
        "members": sorted(package["members"]),
        "resources": _semantic_resources(package["resources"]),
        "nav": package["nav"],
        "ncx": package["ncx"],
        "spine": package["spine"],
        "link_graph": package["link_graph"],
        "target_blocks": package["target_blocks"],
        "source_copies": package["source_copies"],
        "elements": package["html"],
    }
    volatile = {
        "archive_sha256": archive_sha256,
        "member_sha256": package["member_sha256"],
        "proof_output_sha256": evidence["output_sha256"],
    }
    return observed, volatile


def observe_run(
    workspace: str | Path,
    run_relative: str,
    source_relative: str,
    outputs: list[dict[str, Any]],
    *,
    source_origin: str | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace).resolve(strict=True)
    source_path = _workspace_file(workspace_path, source_relative)
    source = _package(source_path)
    state = _state_observation(
        workspace_path, run_relative, source_relative, source_origin or str(source_path)
    )
    source_sha256 = sha256_file(source_path)
    if state["manifest"]["source_bytes_sha256"] != source_sha256:
        raise ObservationError("source archive and persisted identity hashes differ")
    segments = [
        segment for chapter in state["manifest"]["chapters"] for segment in chapter["segments"]
    ]
    roles = [item.get("role") for item in outputs if isinstance(item, dict)]
    if (
        not outputs
        or len(roles) != len(outputs)
        or len(roles) != len(set(roles))
        or set(roles) - _ROLES
    ):
        raise ObservationError("candidate output roles are empty, duplicated, or unknown")
    observed_outputs = []
    volatile_outputs = {}
    for item in outputs:
        observed, volatile = _verify_output_record(workspace_path, item, segments, source)
        observed_outputs.append(observed)
        volatile_outputs[observed["role"]] = volatile
    source_target_observations = [
        {
            "source": segment["source"],
            "translation_source": segment["translation_source"],
            "target": segment["target"],
            "kind": segment["kind"],
            "resource_href": segment["resource_href"],
            "source_fingerprint": segment["source_fingerprint"],
            "rich_source": segment["rich_source"],
            "rich_target": segment["rich_target"],
        }
        for segment in segments
    ]
    return {
        "format_version": 1,
        "source": {
            "members": sorted(source["members"]),
            "resources": _semantic_resources(source["resources"]),
            "opf_path": source["opf_path"],
            "metadata": source["metadata"],
            "spine": source["spine"],
            "nav": source["nav"],
            "ncx": source["ncx"],
            "html": source["html"],
            "link_graph": source["link_graph"],
        },
        "run": state,
        "outputs": sorted(observed_outputs, key=lambda item: item["role"]),
        "source_target_observations": source_target_observations,
        "volatile": {
            "source_archive_sha256": source_sha256,
            "source_member_sha256": source["member_sha256"],
            "outputs": volatile_outputs,
        },
    }


def assert_semantic_match(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    def stable(projection: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in projection.items() if key != "volatile"}
        run = result.get("run")
        if isinstance(run, dict):
            result["run"] = {key: value for key, value in run.items() if key != "volatile"}
            result["run"]["manifest"] = {
                key: value for key, value in run["manifest"].items() if key != "source_path"
            }
        return result

    if stable(actual) != stable(expected):
        raise ObservationError("candidate semantic projection differs from the frozen reference")
