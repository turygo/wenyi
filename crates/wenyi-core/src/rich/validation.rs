use super::{
    AtomKind, InlineMark, InlineRun, MarkKind, NoteReference, RichSource, RichTarget, SemanticRole,
    invalid, local_tag, note_label_key, semantic_roles,
};
use crate::CoreError;
use std::collections::{HashMap, HashSet};

fn shape(run: &InlineRun) -> Result<(), CoreError> {
    if run.atom.is_some() && (!run.text.is_empty() || run.slot_id.is_some()) {
        return Err(invalid(
            "atom_run_shape",
            "rich atom run cannot contain prose or a source slot",
        ));
    }
    if run.note.is_some()
        && (!run.text.is_empty()
            || run.slot_id.is_some()
            || run.atom.is_some()
            || run.omission.is_some())
    {
        return Err(invalid(
            "note_run_shape",
            "rich note object cannot contain prose, a slot or another object",
        ));
    }
    if run.marks.iter().collect::<HashSet<_>>().len() != run.marks.len() {
        return Err(invalid(
            "duplicate_run_marks",
            "rich run has duplicate marks",
        ));
    }
    if run.omission.is_some() {
        if !run.text.is_empty()
            || run.atom.is_some()
            || run.slot_id.is_some()
            || run.marks.is_empty()
        {
            return Err(invalid(
                "omission_run_shape",
                "rich omission requires an empty marked target range",
            ));
        }
        if run.explanation.trim().is_empty() {
            return Err(invalid(
                "omission_explanation",
                "rich omission requires an explanation",
            ));
        }
    } else if !run.explanation.is_empty() {
        return Err(invalid(
            "unexpected_explanation",
            "rich explanation requires an omission",
        ));
    }
    Ok(())
}

pub(super) fn validate_source(source: &RichSource) -> Result<(), CoreError> {
    if source.version != 2 {
        return Err(invalid("source_version", "rich source requires version 2"));
    }
    let mut ids = HashSet::new();
    for id in source
        .marks
        .iter()
        .map(|mark| &mark.id)
        .chain(source.atoms.iter().map(|atom| &atom.id))
    {
        if !ids.insert(id) {
            return Err(invalid(
                "duplicate_source_ids",
                "rich source IDs must be unique",
            ));
        }
    }
    let known: HashSet<_> = source.marks.iter().map(|mark| mark.id.as_str()).collect();
    for run in &source.runs {
        shape(run)?;
        if run.omission.is_some() || !run.explanation.is_empty() || run.note.is_some() {
            return Err(invalid(
                "source_target_declaration",
                "rich source cannot contain target omission declarations",
            ));
        }
        if run.marks.iter().any(|key| !known.contains(key.as_str())) {
            return Err(invalid(
                "unknown_source_mark",
                "rich source run has an unknown mark",
            ));
        }
    }
    atoms_once(source, &source.runs)?;
    note_inventory(source)?;
    Ok(())
}

pub(super) fn validate_target(target: &RichTarget) -> Result<(), CoreError> {
    if !matches!(target.version, 2 | 3) {
        return Err(invalid(
            "target_version",
            "rich target requires version 2 or 3",
        ));
    }
    for run in &target.runs {
        shape(run)?;
        if target.version != 3 && (run.omission.is_some() || run.note.is_some()) {
            return Err(invalid(
                "target_object_version",
                "rich omission or note object requires target version 3",
            ));
        }
    }
    Ok(())
}

fn atoms_once(source: &RichSource, runs: &[InlineRun]) -> Result<(), CoreError> {
    let known: HashSet<_> = source.atoms.iter().map(|atom| atom.id.as_str()).collect();
    let mut actual = HashSet::new();
    for atom in runs.iter().filter_map(|run| run.atom.as_deref()) {
        if !known.contains(atom) || !actual.insert(atom) {
            return Err(invalid(
                "atom_inventory",
                "rich runs must own each source atom exactly once",
            ));
        }
    }
    if actual != known {
        return Err(invalid(
            "atom_inventory",
            "rich runs must own each source atom exactly once",
        ));
    }
    Ok(())
}

fn note_inventory(source: &RichSource) -> Result<HashMap<&str, &NoteReference>, CoreError> {
    let marks: HashMap<_, _> = source
        .marks
        .iter()
        .map(|mark| (mark.id.as_str(), mark))
        .collect();
    let mut inventory = HashMap::new();
    for reference in &source.note_references {
        if inventory
            .insert(reference.mark_id.as_str(), reference)
            .is_some()
        {
            return Err(invalid(
                "duplicate_note_evidence",
                "rich note reference requires unique source evidence",
            ));
        }
        let mark = marks
            .get(reference.mark_id.as_str())
            .ok_or_else(|| invalid("note_source_identity", "missing note source mark"))?;
        if local_tag(&mark.tag) != "a"
            || mark.source_text != reference.label
            || reference.content_marks.first() != Some(&reference.mark_id)
        {
            return Err(invalid(
                "note_source_identity",
                "rich note reference source identity mismatch",
            ));
        }
        note_label_key(&reference.label)?;
        let mut parent = *mark;
        for key in &reference.content_marks[1..] {
            let child = marks
                .get(key.as_str())
                .ok_or_else(|| invalid("note_content_chain", "missing content chain mark"))?;
            if !matches!(local_tag(&child.tag).as_ref(), "span" | "sup")
                || !descends(&child.path, &parent.path)
            {
                return Err(invalid(
                    "note_content_chain",
                    "rich note reference has an invalid content chain",
                ));
            }
            parent = child;
        }
        let descendants: HashSet<_> = source
            .marks
            .iter()
            .filter(|item| descends(&item.path, &mark.path))
            .map(|item| item.id.as_str())
            .collect();
        let content: HashSet<_> = reference.content_marks[1..]
            .iter()
            .map(String::as_str)
            .collect();
        if content != descendants {
            return Err(invalid(
                "note_incomplete_chain",
                "rich note reference content chain is incomplete",
            ));
        }
    }
    Ok(inventory)
}

fn descends(child: &[i32], parent: &[i32]) -> bool {
    child.len() > parent.len() && child.starts_with(parent)
}

pub fn validate_rich_target(
    source: &RichSource,
    target: &RichTarget,
    expected_text: Option<&str>,
) -> Result<(), CoreError> {
    source.validate()?;
    target.validate()?;
    if expected_text.is_some_and(|text| {
        !target
            .runs
            .iter()
            .map(|run| run.text.as_str())
            .flat_map(str::chars)
            .eq(text.chars())
    }) {
        return Err(invalid(
            "target_text_changed",
            "rich annotation changed the target text",
        ));
    }
    let marks: HashMap<_, _> = source
        .marks
        .iter()
        .map(|mark| (mark.id.as_str(), mark))
        .collect();
    let mut used = HashSet::new();
    let mut represented = HashSet::new();
    for run in &target.runs {
        if run.slot_id.is_some() {
            return Err(invalid(
                "target_source_slot",
                "rich target cannot contain source slot coordinates",
            ));
        }
        for key in &run.marks {
            if !marks.contains_key(key.as_str()) {
                return Err(invalid(
                    "unknown_target_mark",
                    "rich target contains unknown marks",
                ));
            }
            represented.insert(key.as_str());
        }
        if !run.text.trim().is_empty() || run.note.is_some() {
            used.extend(run.marks.iter().map(String::as_str));
        } else if run.text.is_empty()
            && run.atom.is_none()
            && !run.marks.is_empty()
            && run.omission.is_none()
            && !run.marks.iter().any(|key| anchor(marks[key.as_str()]))
        {
            return Err(invalid(
                "empty_marked_run",
                "empty marked run requires an original anchor identity",
            ));
        }
    }
    used.extend(omitted_marks(target, &marks)?);
    validate_notes(source, target, &marks)?;
    atoms_once(source, &target.runs)?;
    validate_nesting(&marks, &target.runs)?;
    let note_ids: HashSet<_> = source
        .atoms
        .iter()
        .filter(|atom| atom.kind == AtomKind::Note)
        .map(|atom| atom.id.as_str())
        .collect();
    for run in &target.runs {
        if run.atom.as_deref().is_some_and(|id| note_ids.contains(id))
            && run
                .marks
                .iter()
                .any(|key| semantic_roles(marks[key.as_str()]).contains(&SemanticRole::Link))
        {
            return Err(invalid(
                "nested_note_links",
                "rich target contains nested note links",
            ));
        }
    }
    for mark in &source.marks {
        if !mark.source_text.trim().is_empty()
            && (mark.required || !semantic_roles(mark).is_empty())
            && !used.contains(mark.id.as_str())
        {
            return Err(invalid(
                "required_mark_lost",
                "rich target lost a required semantic mark",
            ));
        }
        if anchor(mark) && !represented.contains(mark.id.as_str()) {
            return Err(invalid(
                "anchor_identity_lost",
                "rich target lost an inline anchor identity",
            ));
        }
    }
    Ok(())
}

fn anchor(mark: &InlineMark) -> bool {
    ["id", "name"].iter().any(|key| {
        mark.attributes
            .get(key)
            .is_some_and(|value| !value.is_empty())
    })
}

fn omitted_marks<'a>(
    target: &'a RichTarget,
    marks: &HashMap<&str, &InlineMark>,
) -> Result<HashSet<&'a str>, CoreError> {
    let meaningful: HashSet<_> = target
        .runs
        .iter()
        .filter(|run| meaningful(run))
        .flat_map(|run| run.marks.iter().map(String::as_str))
        .collect();
    let mut omitted = HashSet::new();
    for run in target.runs.iter().filter(|run| run.omission.is_some()) {
        for key in &run.marks {
            let mark = marks[key.as_str()];
            let roles = semantic_roles(mark);
            if mark.source_text.trim().is_empty()
                || mark.kind == MarkKind::Decoration
                || roles
                    .iter()
                    .any(|role| !matches!(role, SemanticRole::Bold | SemanticRole::Italic))
                || (roles.is_empty() && !(mark.kind == MarkKind::Style && mark.required))
            {
                return Err(invalid(
                    "ineligible_omission",
                    "rich omission contains an ineligible source mark",
                ));
            }
            if !omitted.insert(key.as_str()) {
                return Err(invalid(
                    "duplicate_omission",
                    "rich source mark has duplicate omissions",
                ));
            }
            if meaningful.contains(key.as_str()) {
                return Err(invalid(
                    "meaningful_omission",
                    "rich source mark has both omission and meaningful target range",
                ));
            }
        }
    }
    Ok(omitted)
}

fn validate_notes(
    source: &RichSource,
    target: &RichTarget,
    marks: &HashMap<&str, &InlineMark>,
) -> Result<(), CoreError> {
    let references = note_inventory(source)?;
    let mut restored = HashSet::new();
    for run in &target.runs {
        let Some(key) = run.note.as_deref() else {
            continue;
        };
        let reference = references.get(key).ok_or_else(|| {
            invalid(
                "unproven_note",
                "rich note object requires proven source evidence",
            )
        })?;
        if !restored.insert(key) {
            return Err(invalid(
                "duplicate_note_object",
                "rich target has a duplicate note object",
            ));
        }
        if !run.marks.ends_with(&reference.content_marks) {
            return Err(invalid(
                "note_target_chain",
                "rich note object lost its source content chain",
            ));
        }
        let parent_path = &marks[key].path;
        if run.marks[..run.marks.len() - reference.content_marks.len()]
            .iter()
            .any(|key| !descends(parent_path, &marks[key.as_str()].path))
        {
            return Err(invalid(
                "note_parent_scope",
                "rich note object has an unrelated parent scope",
            ));
        }
    }
    for (key, reference) in references {
        let label: String = target
            .runs
            .iter()
            .filter(|run| run.marks.iter().any(|mark| mark == key))
            .map(|run| run.text.as_str())
            .collect();
        if restored.contains(key) {
            if !label.is_empty() {
                return Err(invalid(
                    "duplicate_note_text",
                    "rich note reference has both restored and existing text",
                ));
            }
        } else if note_label_key(&label)? != note_label_key(&reference.label)? {
            return Err(invalid(
                "note_label_lost",
                "rich note reference lost its existing target label",
            ));
        }
    }
    Ok(())
}

fn meaningful(run: &InlineRun) -> bool {
    !run.text.is_empty() || run.atom.is_some() || run.note.is_some()
}

fn validate_nesting(
    marks: &HashMap<&str, &InlineMark>,
    runs: &[InlineRun],
) -> Result<(), CoreError> {
    let meaningful_marks: HashSet<_> = runs
        .iter()
        .filter(|run| meaningful(run))
        .flat_map(|run| run.marks.iter().map(String::as_str))
        .collect();
    let mut active: &[String] = &[];
    let mut opened = HashSet::new();
    for run in runs {
        if !meaningful(run)
            && run
                .marks
                .iter()
                .all(|key| meaningful_marks.contains(key.as_str()))
        {
            continue;
        }
        if run
            .marks
            .iter()
            .filter(|key| semantic_roles(marks[key.as_str()]).contains(&SemanticRole::Link))
            .count()
            > 1
        {
            return Err(invalid("nested_links", "rich target contains nested links"));
        }
        let common = active
            .iter()
            .zip(&run.marks)
            .take_while(|(left, right)| left == right)
            .count();
        for key in &run.marks[common..] {
            if semantic_roles(marks[key.as_str()]).contains(&SemanticRole::Link)
                && !opened.insert(key.as_str())
            {
                return Err(invalid(
                    "reopened_link",
                    "rich target splits a source link entity",
                ));
            }
        }
        active = &run.marks;
    }
    Ok(())
}
