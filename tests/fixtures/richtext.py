"""结构测试的显式格式目标；语义范围测试应自行构造精确 runs。"""

from trans_novel.epub.richtext import InlineRun, RichTarget, validate_rich_target
from trans_novel.epub.slots import EpubSegmentState


def synthetic_rich_target(state: EpubSegmentState, text: str) -> RichTarget:
    """让结构测试显式声明全部来源对象，避免使用源长度猜测位置。"""
    source = state.rich_source
    if source is None:
        raise ValueError("test fixture requires extracted source evidence")
    target = RichTarget(runs=[InlineRun(text=text, marks=tuple(mark.id for mark in source.marks))])
    target.runs.extend(InlineRun(atom=atom.id) for atom in source.atoms)
    validate_rich_target(source, target, expected_text=text)
    return target
