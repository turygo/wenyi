from __future__ import annotations

import zipfile
from pathlib import Path

FIXTURE_ID = "structured-epub-balanced-v1"
BASELINE_COMMIT = "61835587606b5cb81cdcd81eec8427fd4f638df4"
MODEL = "deepseek-chat"

SOURCE_TARGETS = {
    "Chapter One": "第一章",
    "Chapter Two": "第二章",
    "A quiet passage holds a story.": "一段安静的文字讲述着故事。",
    "A curved line returns.": "一条弯曲的线条回来了。",
    "Follow the harbor home.": "沿着这条通往港湾的路回家。",
    "A🙂 reader keeps cafe\u0301 exact.": "🙂读者原样保留 cafe\u0301。",
    "Day falls.\nNight returns.": "白昼落下。\n夜色归来。",
    "A small map rests here.": "这里留有一张小地图。",
    "Every reader needs a note.": "每位读者都需要注释。",
    "First list item.": "列表中的第一项。",
    "Second list item.": "列表中的第二项。",
    "A quiet harbor waits.": "安静的港湾仍在等待。",
    "Coordinates": "坐标",
    "Axis": "坐标轴",
    "Value": "数值",
    "east": "东",
    "3": "3",
    "One careful footnote.": "一条简短脚注。",
}

TITLE_TARGETS = {
    "Harbor Notes": "港湾札记",
    "Chapter One": "第一章",
    "Chapter Two": "第二章",
}

MARK_TARGETS = {
    "quiet": "安静",
    "curved": "弯曲",
    "the harbor": "港湾",
    "1": "1",
    "↩": "↩",
}

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

OPF = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">urn:wenyi:r0:structured-v1</dc:identifier>
    <dc:title>Harbor Notes</dc:title><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="style" href="style.css" media-type="text/css"/>
    <item id="map" href="images/map.png" media-type="image/png"/>
    <item id="chapter-one" href="text/chapter-1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter-two" href="text/chapter-2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx"><itemref idref="chapter-one"/><itemref idref="chapter-two"/></spine>
</package>"""

NAV = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title></head><body><nav epub:type="toc"><h1>Contents</h1><ol>
<li><a href="text/chapter-1.xhtml#chapter-one">Chapter One</a></li>
<li><a href="text/chapter-2.xhtml#chapter-two">Chapter Two</a></li>
</ol></nav></body></html>"""

NCX = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><head/>
<docTitle><text>Harbor Notes</text></docTitle><navMap>
<navPoint id="one"><navLabel><text>Chapter One</text></navLabel><content src="text/chapter-1.xhtml#chapter-one"/></navPoint>
<navPoint id="two"><navLabel><text>Chapter Two</text></navLabel><content src="text/chapter-2.xhtml#chapter-two"/></navPoint>
</navMap></ncx>"""

CHAPTER_ONE = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en">
<head><title>Chapter One</title><link rel="stylesheet" type="text/css" href="../style.css"/></head>
<body><h1 id="chapter-one">Chapter One</h1>
<p id="bold">A <strong>quiet</strong> passage holds a story.</p>
<p id="italic">A <em>curved</em> line returns.</p>
<p id="cross-link">Follow <a href="chapter-2.xhtml#chapter-two">the harbor</a> home.</p>
<p id="unicode">A🙂 reader keeps cafe\u0301 exact.</p>
<p id="linebreak">Day falls.<br/>Night returns.</p>
<p id="image">A small map rests here.<img src="../images/map.png" alt="map"/></p>
<p id="note-reference">Every reader needs a note.<a id="note-ref" epub:type="noteref" href="chapter-2.xhtml#footnote-one"><sup>1</sup></a></p>
<ul><li>First list item.</li><li>Second list item.</li></ul>
<blockquote><p>A quiet harbor waits.</p></blockquote>
<table><caption>Coordinates</caption><thead><tr><th>Axis</th><th>Value</th></tr></thead>
<tbody><tr><td>east</td><td>3</td></tr></tbody></table>
</body></html>"""

CHAPTER_TWO = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en">
<head><title>Chapter Two</title><link rel="stylesheet" type="text/css" href="../style.css"/></head>
<body><h1 id="chapter-two">Chapter Two</h1>
<aside id="footnote-one" epub:type="footnote"><p>One careful footnote.
<a epub:type="backlink" href="chapter-1.xhtml#note-ref">↩</a></p></aside>
</body></html>"""

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da6360f8cff01f00040101ffaeb555f50000000049454e44ae426082"
)

RESPONSE_DEFINITIONS = {
    "model": MODEL,
    "token_usage": {
        "prompt_tokens": 37,
        "completion_tokens": 13,
        "total_tokens": 50,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 37,
        "reasoning_tokens": 0,
    },
    "analysis": {
        "genre": "短篇叙事",
        "tone": "平静、克制",
        "style_guide": "保持简洁叙述和明确段落边界。",
        "narration": "第三人称叙述",
        "pacing": "短句为主",
        "register": "中性书面语",
        "dialogue_style": "未观察到对话",
        "rhetoric": "未观察到明显修辞",
        "characters": [],
        "terms": [],
        "conventions": "保留数字和源文中的字形。",
    },
    "chapter_classification": {
        "kind": "translatable",
        "reason": "合成叙事章节包含面向读者的正文。",
    },
    "term_candidates": [],
    "named_terms": [],
    "glossary_terms": [],
}


def _member(name: str, data: bytes, *, stored: bool = False) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info, data


def write_structured_epub(path: str | Path) -> None:
    members = [
        _member("mimetype", b"application/epub+zip", stored=True),
        _member("META-INF/container.xml", CONTAINER.encode("utf-8")),
        _member("OEBPS/content.opf", OPF.encode("utf-8")),
        _member("OEBPS/nav.xhtml", NAV.encode("utf-8")),
        _member("OEBPS/toc.ncx", NCX.encode("utf-8")),
        _member("OEBPS/style.css", b"body { font-family: serif; }\n"),
        _member("OEBPS/images/map.png", PNG),
        _member("OEBPS/text/chapter-1.xhtml", CHAPTER_ONE.encode("utf-8")),
        _member("OEBPS/text/chapter-2.xhtml", CHAPTER_TWO.encode("utf-8")),
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for info, data in members:
            archive.writestr(info, data)
