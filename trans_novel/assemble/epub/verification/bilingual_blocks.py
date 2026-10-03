"""识别经过原书完整子树证明的整块双语副本。"""

from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

from trans_novel.assemble.epub.rendering.bilingual import (
    is_bilingual_container_tag,
    japanese_ruby_source_copy,
    sanitized_source_copy,
)
from trans_novel.assemble.epub.verification import archive_model, structure
from trans_novel.epub.markup import resource_parser


def _shape(node, *, root=True):
    return (
        node.tag.rsplit("}", 1)[-1].lower(),
        tuple(
            sorted(
                (key, value) for key, value in node.attrib.items() if not (root and key == "class")
            )
        ),
        node.text,
        tuple(
            (_shape(child, root=False), child.tail) for child in node if isinstance(child.tag, str)
        ),
    )


def whole_source_mode(source_path: Path, output_path: Path, resource: str) -> bool:
    """完整含 BR 副本启动整块证明；拒绝混入旧逐槽副本或伪造子树。"""
    with zipfile.ZipFile(output_path) as archive:
        output_data = archive_model.read_member(archive, archive.getinfo(resource))
    output_root = resource_parser(output_data)[0].getroot()
    nodes = [
        node
        for node in output_root.iter()
        if isinstance(node.tag, str) and "tn-source" in str(node.get("class", "")).split()
    ]
    if not any(
        archive_model.local_name(node.tag).lower() in {"p", "div"}
        and any(
            isinstance(child.tag, str) and archive_model.local_name(child.tag).lower() == "br"
            for child in node.iterdescendants()
        )
        for node in nodes
    ):
        return False
    with zipfile.ZipFile(source_path) as archive:
        data = archive_model.read_member(archive, archive.getinfo(resource))
    root = resource_parser(data)[0].getroot()
    candidates = []
    block_tags = structure.BLOCK_TAGS | {"div"}
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag not in block_tags or tag in structure.HEADING_TAGS:
            continue
        if any(
            isinstance(child.tag, str)
            and child.tag.rsplit("}", 1)[-1].lower() in block_tags
            and "".join(child.itertext()).strip()
            for child in node.iterdescendants()
        ):
            continue
        if "".join(node.itertext()).strip():
            candidates.append(node)
    used: Counter = Counter()
    for actual in nodes:
        # 精确证明直接读取原始输出树，避免 Soup 折叠纯空白后再序列化。
        signature = _shape(actual)
        actual_tag = archive_model.local_name(actual.tag).lower()
        matched = None
        for index, original in enumerate(candidates):
            if used[index]:
                continue
            original_tag = original.tag.rsplit("}", 1)[-1].lower()
            source_tag = (
                "div"
                if is_bilingual_container_tag(original.tag)
                else original_tag
                if original_tag in {"p", "div"}
                else "p"
            )
            if actual_tag != source_tag:
                continue
            safe = sanitized_source_copy(original, actual_tag)
            ruby = japanese_ruby_source_copy(original, "ja", actual_tag)
            if signature == _shape(safe) or (ruby is not None and signature == _shape(ruby)):
                matched = index
                break
        if matched is None:
            raise ValueError("whole bilingual source subtree mismatch")
        used[matched] += 1
    return True
