"""核对生成的首字覆盖内容，之后移除以继续源资源守恒证明。"""

from __future__ import annotations

from trans_novel.assemble.epub.rendering.decorations import (
    DECORATION_ATTRIBUTES,
    source_decoration_css,
)


def prove_and_remove_decoration_style(archive, resource, source_root, output_root, *, required):
    """只允许来源绑定的精确样式，不能忽略任意新增 style。"""
    nodes = [node for node in output_root.iter() if node.get("data-tn-richtext") == "decorations"]
    if not required and not nodes:
        return
    expected = source_decoration_css(archive, resource, source_root)
    if len(nodes) != bool(expected):
        raise ValueError("EPUB decoration override missing or duplicated")
    for node in nodes:
        parent = node.getparent()
        if (
            node.tag.rsplit("}", 1)[-1] != "style"
            or parent is None
            or parent.tag.rsplit("}", 1)[-1] != "head"
            or dict(node.attrib) != DECORATION_ATTRIBUTES
            or node.text != expected
            or len(node)
            or node.tail
        ):
            raise ValueError("EPUB decoration override does not match source")
        parent.remove(node)


def prove_soup_decoration_style(archive, resource, source_root, soup):
    """资源级验证只检查已声明覆盖；缺失检查由状态绑定的 DOM 门禁完成。"""
    nodes = soup.find_all(attrs={"data-tn-richtext": "decorations"})
    if not nodes:
        return
    expected = source_decoration_css(archive, resource, source_root)
    if len(nodes) != bool(expected):
        raise ValueError("EPUB decoration override duplicated or fabricated")
    for node in nodes:
        if (
            node.name != "style"
            or node.parent.name != "head"
            or dict(node.attrs) != DECORATION_ATTRIBUTES
            or str(node.string or "") != expected
        ):
            raise ValueError("EPUB decoration override does not match source")
        node.extract()
