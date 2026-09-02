"""Navigation DOM proof helpers."""

from __future__ import annotations

from lxml import etree

from trans_novel.assemble.epub.verification import archive_model, dom


def nav_label_locations(
    root: etree._Element, *, is_ncx: bool
) -> list[tuple[etree._Element, tuple[int, ...]]]:
    """Locate exactly the labels enumerated by the production TOC parser."""
    locations: list[tuple[etree._Element, tuple[int, ...]]] = []

    def direct(parent: etree._Element, name: str) -> etree._Element | None:
        return next(
            (
                child
                for child in dom.element_children_lxml(parent)
                if child.tag.rsplit("}", 1)[-1].lower() == name.lower()
            ),
            None,
        )

    if is_ncx:
        nav_map = next(
            (
                node
                for node in root.iter()
                if archive_model.local_name(node.tag).lower() == "navmap"
            ),
            None,
        )
        if nav_map is None:
            return locations

        def walk_ncx(parent: etree._Element) -> None:
            for point in dom.element_children_lxml(parent):
                if archive_model.local_name(point.tag).lower() != "navpoint":
                    continue
                label_parent = direct(point, "navLabel")
                label = (
                    next(
                        (
                            child
                            for child in label_parent.iter()
                            if isinstance(child.tag, str)
                            and archive_model.local_name(child.tag) == "text"
                        ),
                        None,
                    )
                    if label_parent is not None
                    else None
                )
                if label is not None:
                    path = dom.element_path_lxml(root, label)
                    if path is not None:
                        locations.append((label, path))
                walk_ncx(point)

        walk_ncx(nav_map)
        return locations

    navs = [
        node
        for node in root.iter()
        if isinstance(node.tag, str)
        and archive_model.local_name(node.tag).lower() == "nav"
        and "toc"
        in (
            str(node.get("epub:type", node.get("type", "")))
            + " "
            + str(node.get("{http://www.idpf.org/2007/ops}type", ""))
        ).split()
    ]
    if not navs:
        all_navs = [
            node
            for node in root.iter()
            if isinstance(node.tag, str) and archive_model.local_name(node.tag).lower() == "nav"
        ]
        navs = all_navs[:1]

    def walk_nav(ordered_list: etree._Element) -> None:
        for li in dom.element_children_lxml(ordered_list):
            if archive_model.local_name(li.tag).lower() != "li":
                continue
            label = next(
                (
                    child
                    for child in dom.element_children_lxml(li)
                    if archive_model.local_name(child.tag).lower() in {"a", "span"}
                ),
                None,
            )
            if label is not None:
                path = dom.element_path_lxml(root, label)
                if path is not None:
                    locations.append((label, path))
            nested = direct(li, "ol")
            if nested is not None:
                walk_nav(nested)

    for nav in navs:
        ordered = direct(nav, "ol")
        if ordered is None:
            ordered = next(
                (node for node in nav.iter() if archive_model.local_name(node.tag).lower() == "ol"),
                None,
            )
        if ordered is not None:
            walk_nav(ordered)
    return locations
