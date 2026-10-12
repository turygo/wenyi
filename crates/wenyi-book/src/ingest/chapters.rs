#![forbid(unsafe_code)]
use crate::document::{
    ContentResource, NavigationEntry, SegmentKind, SourceChapter, SourceSegment,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(super) fn assemble(
    reading_order: &[String],
    resources: &[ContentResource],
    segments: &[SourceSegment],
    navigation: &mut [NavigationEntry],
) -> (Vec<SourceChapter>, Option<String>) {
    let mut ordered = Vec::new();
    let mut starts = HashMap::new();
    for href in reading_order {
        if let Some(resource) = resources.iter().find(|resource| resource.href == *href) {
            starts.insert(href.as_str(), ordered.len());
            ordered.extend_from_slice(&resource.segment_ids);
        }
    }
    let positions: HashMap<_, _> = ordered
        .iter()
        .enumerate()
        .map(|(position, &id)| (id, position))
        .collect();
    let boundaries = bind_navigation(resources, navigation, &starts, &positions);
    let mut scalar_prefix = Vec::with_capacity(ordered.len() + 1);
    scalar_prefix.push(0);
    for &id in &ordered {
        scalar_prefix.push(scalar_prefix.last().unwrap() + segments[id].source.chars().count());
    }
    let paths: Vec<_> = navigation.iter().map(|entry| entry.toc_path.as_str()).fold(
        Vec::new(),
        |mut paths, path| {
            if !paths.contains(&path) {
                paths.push(path);
            }
            paths
        },
    );
    for path in paths {
        let entries: Vec<_> = navigation
            .iter()
            .filter(|entry| {
                entry.toc_path == path
                    && !entry.external
                    && boundaries.contains_key(&entry.entry_id)
            })
            .collect();
        let depths: BTreeSet<_> = entries.iter().map(|entry| entry.depth).collect();
        let Some(&shallow) = depths.first() else {
            continue;
        };
        let mut selected = at_depth(&entries, &boundaries, shallow);
        for depth in depths {
            let candidates = at_depth(&entries, &boundaries, depth);
            if median(&candidates, &scalar_prefix) >= 3000 {
                selected = candidates;
            }
        }
        let mut chapters = Vec::new();
        let mut cuts: Vec<_> = selected.into_iter().collect();
        if cuts.first().is_some_and(|(position, _)| *position > 0) {
            cuts.insert(0, (0, None));
        }
        for (index, &(start, boundary)) in cuts.iter().enumerate() {
            let end = cuts.get(index + 1).map_or(ordered.len(), |cut| cut.0);
            if end <= start || start >= ordered.len() {
                continue;
            }
            let ids = ordered[start..end].to_vec();
            let first = &segments[ids[0]];
            chapters.push(SourceChapter {
                index: chapters.len(),
                title: boundary.map_or_else(
                    || {
                        if first.kind == SegmentKind::Heading {
                            first.source.clone()
                        } else {
                            String::new()
                        }
                    },
                    |entry| entry.label.clone(),
                ),
                href: resources[first.resource_index].href.clone(),
                toc_entry_id: boundary.map(|entry| entry.entry_id.clone()),
                segment_ids: ids,
            });
        }
        if !chapters.is_empty() {
            return (chapters, Some(path.to_owned()));
        }
    }
    let chapters = reading_order
        .iter()
        .filter_map(|href| resources.iter().find(|resource| resource.href == *href))
        .filter(|resource| !resource.segment_ids.is_empty())
        .enumerate()
        .map(|(index, resource)| SourceChapter {
            index,
            title: resource.title.clone(),
            href: resource.href.clone(),
            toc_entry_id: None,
            segment_ids: resource.segment_ids.clone(),
        })
        .collect();
    (chapters, None)
}
fn bind_navigation(
    resources: &[ContentResource],
    navigation: &mut [NavigationEntry],
    starts: &HashMap<&str, usize>,
    positions: &HashMap<usize, usize>,
) -> HashMap<String, usize> {
    let mut boundaries = HashMap::new();
    for entry in navigation.iter_mut().filter(|entry| !entry.external) {
        let Some(&start) = starts.get(entry.resource_href.as_str()) else {
            continue;
        };
        let resource = resources
            .iter()
            .find(|resource| resource.href == entry.resource_href)
            .unwrap();
        let segment = if entry.fragment.is_empty() {
            resource.segment_ids.first().copied()
        } else {
            let Some(segment) = resource.fragment_anchors.get(&entry.fragment) else {
                continue;
            };
            *segment
        };
        entry.boundary_segment = segment;
        boundaries.insert(
            entry.entry_id.clone(),
            segment
                .and_then(|id| positions.get(&id).copied())
                .unwrap_or(start + resource.segment_ids.len()),
        );
    }
    for index in (0..navigation.len()).rev() {
        let entry = &navigation[index];
        if !entry.raw_href.is_empty() || boundaries.contains_key(&entry.entry_id) {
            continue;
        }
        if let Some(position) = navigation
            .iter()
            .filter(|child| child.parent_entry_id.as_deref() == Some(&entry.entry_id))
            .find_map(|child| boundaries.get(&child.entry_id).copied())
        {
            boundaries.insert(entry.entry_id.clone(), position);
        }
    }
    boundaries
}

fn at_depth<'a>(
    entries: &[&'a NavigationEntry],
    positions: &HashMap<String, usize>,
    depth: usize,
) -> BTreeMap<usize, Option<&'a NavigationEntry>> {
    let mut result: BTreeMap<usize, Option<&NavigationEntry>> = BTreeMap::new();
    for &entry in entries.iter().filter(|entry| entry.depth <= depth) {
        let position = positions[&entry.entry_id];
        let replace = result.get(&position).is_none_or(|previous| {
            previous.is_none_or(|previous| previous.boundary_segment.is_none())
        });
        if replace {
            result.insert(position, Some(entry));
        }
    }
    result
}
fn median(candidates: &BTreeMap<usize, Option<&NavigationEntry>>, prefix: &[usize]) -> usize {
    let mut bounds: Vec<_> = candidates.keys().copied().collect();
    if bounds.first().is_some_and(|&position| position > 0) {
        bounds.insert(0, 0);
    }
    bounds.push(prefix.len() - 1);
    let mut lengths: Vec<_> = bounds
        .windows(2)
        .filter(|bounds| bounds[1] > bounds[0])
        .map(|bounds| prefix[bounds[1]] - prefix[bounds[0]])
        .collect();
    lengths.sort_unstable();
    if lengths.is_empty() {
        0
    } else if lengths.len() % 2 == 1 {
        lengths[lengths.len() / 2]
    } else {
        (lengths[lengths.len() / 2 - 1] + lengths[lengths.len() / 2]) / 2
    }
}
