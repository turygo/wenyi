"""仅为已翻译中文生成可独立证明的首字装饰覆盖。"""

from __future__ import annotations

from lxml import etree

from trans_novel.assemble.epub.rendering.source_dom import (
    parse_source_markup,
    serialize_source_tree,
)
from trans_novel.assemble.epub.rendering.theme.source_css import (
    chinese_dropcap_overrides,
    collect_source_stylesheets,
)
from trans_novel.ingest.models import segment_preserves_source
from trans_novel.postprocess.language import normalize_lang_code

DECORATION_ATTRIBUTES = {"type": "text/css", "data-tn-richtext": "decorations"}


def source_decoration_css(archive, resource: str, root) -> str:
    """覆盖规则只来自原书样式，不能由模型或输出文件提供。"""
    sheets = collect_source_stylesheets(archive, root, resource)
    return chinese_dropcap_overrides(sheets, resource=resource, root=root)


def render_decoration_overrides(archive, resource, original, rendered, segments, target_lang):
    """独立于可选主题安装语言限定的排版覆盖，保留全部源样式。"""
    if normalize_lang_code(target_lang) != "zh" or not any(
        item.rich_target is not None and not segment_preserves_source(item) for item in segments
    ):
        return rendered
    source_tree, _ = parse_source_markup(original)
    root = source_tree.getroot()
    if any(node.get("data-tn-richtext") == "decorations" for node in root.iter()):
        raise ValueError("EPUB reserved decoration marker collision")
    css = source_decoration_css(archive, resource, root)
    if not css:
        return rendered
    tree, mode = parse_source_markup(rendered)
    output = tree.getroot()
    head = next((node for node in output if node.tag.rsplit("}", 1)[-1] == "head"), None)
    if head is None:
        raise ValueError("EPUB decoration override requires an original head")
    namespace = output.tag.split("}", 1)[0] + "}" if output.tag.startswith("{") else ""
    style = etree.SubElement(head, namespace + "style", attrib=DECORATION_ATTRIBUTES)
    style.text = css
    return serialize_source_tree(tree, rendered, mode)
