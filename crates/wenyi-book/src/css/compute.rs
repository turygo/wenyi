use super::values::{COUNT, Projection, Value, project};
use super::{ElementStyle, SourceCss, error};
use crate::error::BookError;
use crate::source::{NodeId, SourceTree};
use lightningcss::stylesheet::PrinterOptions;
use lightningcss::traits::ToCss;
use std::collections::HashSet;
use wenyi_core::{SemanticRole, tag_semantics};

#[derive(Clone)]
struct State {
    values: [f32; COUNT],
    evidence: Vec<String>,
    emphasis: Vec<EmphasisScope>,
    decoration: bool,
}
#[derive(Clone, Copy)]
struct EmphasisScope {
    origin: NodeId,
    role: SemanticRole,
    threshold: f32,
}
impl Default for State {
    fn default() -> Self {
        Self {
            values: [400.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            evidence: Vec::new(),
            emphasis: Vec::new(),
            decoration: false,
        }
    }
}
#[derive(Clone, Copy)]
struct Winner {
    value: Value,
    priority: (bool, bool, u32, usize),
    rule: Option<usize>,
}
pub(super) fn evidence(
    css: &SourceCss,
    tree: &SourceTree,
    requested: &[NodeId],
    resource: &str,
) -> Result<Vec<ElementStyle>, BookError> {
    let mut needed = HashSet::new();
    for &node in requested {
        if tree.local_name(node).is_none() {
            return Err(error(
                resource,
                "css-node",
                "requested style node is not an element",
            ));
        }
        let mut current = Some(node);
        let mut depth = 0;
        while let Some(node) = current {
            depth += 1;
            if depth > 64 {
                return Err(error(
                    resource,
                    "css-depth",
                    "requested source ancestry exceeds 64",
                ));
            }
            needed.insert(node);
            current = tree.node(node).and_then(|node| node.parent);
        }
    }
    let projections: Vec<_> = css
        .rules
        .iter()
        .map(|rule| project(&rule.declarations))
        .collect();
    let inline: Vec<_> = css
        .inline
        .iter()
        .map(|(node, declarations)| (*node, project(declarations)))
        .collect();
    let evaluation = Evaluation {
        css,
        tree,
        projections: &projections,
        inline: &inline,
        resource,
    };
    let mut baseline: Option<Vec<ElementStyle>> = None;
    for alternative in 0..(1_usize << css.conditions.len()) {
        let candidate = evaluation.evaluate(requested, &needed, alternative)?;
        if let Some(baseline) = baseline.as_mut() {
            for (baseline, candidate) in baseline.iter_mut().zip(candidate) {
                if baseline.roles != candidate.roles
                    || baseline.active_emphasis != candidate.active_emphasis
                    || baseline.required != candidate.required
                    || baseline.decoration != candidate.decoration
                {
                    return Err(error(
                        resource,
                        "css-semantic-ambiguity",
                        "conditional source styles disagree on semantic evidence",
                    ));
                }
                for text in candidate.evidence {
                    if !baseline.evidence.contains(&text) {
                        baseline.evidence.push(text);
                    }
                }
            }
        } else {
            baseline = Some(candidate);
        }
    }
    Ok(baseline.unwrap_or_default())
}
struct Evaluation<'a> {
    css: &'a SourceCss,
    tree: &'a SourceTree,
    projections: &'a [Projection],
    inline: &'a [(NodeId, Projection)],
    resource: &'a str,
}
impl Evaluation<'_> {
    fn evaluate(
        &self,
        requested: &[NodeId],
        needed: &HashSet<NodeId>,
        alternative: usize,
    ) -> Result<Vec<ElementStyle>, BookError> {
        let mut states: Vec<Option<State>> = vec![None; self.tree.nodes.len()];
        let mut styles: Vec<Option<ElementStyle>> = vec![None; self.tree.nodes.len()];
        let mut pending = vec![self.tree.root];
        while let Some(node) = pending.pop() {
            if !needed.contains(&node) {
                continue;
            }
            pending.extend(self.tree.element_children(node));
            let default_parent = State::default();
            let parent = self.tree.nodes[node.index()]
                .parent
                .and_then(|parent| states[parent.index()].as_ref())
                .unwrap_or(&default_parent);
            let native = native_values(self.tree.local_name(node).unwrap_or(""), parent);
            let winners = self.winners(node, alternative)?;
            let state = self.state(node, parent, native, &winners)?;
            let style = classify(self.tree, node, parent, &state);
            states[node.index()] = Some(state);
            styles[node.index()] = Some(style);
        }
        requested
            .iter()
            .map(|node| {
                styles[node.index()].clone().ok_or_else(|| {
                    error(
                        self.resource,
                        "css-node",
                        "requested element is disconnected from source root",
                    )
                })
            })
            .collect()
    }
    fn winners(
        &self,
        node: NodeId,
        alternative: usize,
    ) -> Result<[Option<Winner>; COUNT], BookError> {
        let mut winners = [None; COUNT];
        for (index, (rule, projection)) in self.css.rules.iter().zip(self.projections).enumerate() {
            if !projection.relevant()
                || rule
                    .conditions
                    .iter()
                    .any(|condition| alternative & (1 << condition) == 0)
            {
                continue;
            }
            let specificity = rule.selection.specificity(self.tree, node).map_err(|()| {
                error(
                    self.resource,
                    "css-selector-state",
                    "relevant selector has undetermined source selection",
                )
            })?;
            let Some(specificity) = specificity else {
                continue;
            };
            if rule.unsupported {
                return Err(error(
                    self.resource,
                    "css-rule",
                    "matched source rule has unsupported cascade or scope semantics",
                ));
            }
            if projection.animation {
                return Err(error(
                    self.resource,
                    "css-animation",
                    "animation affects requested source ancestry",
                ));
            }
            cascade(
                &mut winners,
                projection,
                false,
                specificity,
                index,
                Some(index),
            );
        }
        if let Some((_, projection)) = self.inline.iter().find(|(id, _)| *id == node) {
            if projection.animation {
                return Err(error(
                    self.resource,
                    "css-animation",
                    "inline animation affects requested source ancestry",
                ));
            }
            cascade(
                &mut winners,
                projection,
                true,
                0,
                self.css.rules.len(),
                None,
            );
        }
        Ok(winners)
    }
    fn state(
        &self,
        node: NodeId,
        parent: &State,
        native: [f32; COUNT],
        winners: &[Option<Winner>; COUNT],
    ) -> Result<State, BookError> {
        let mut state = State {
            values: native,
            evidence: Vec::new(),
            emphasis: Vec::new(),
            decoration: false,
        };
        let mut inline_evidence = false;
        for (index, winner) in winners.iter().enumerate() {
            if let Some(winner) = winner {
                state.values[index] =
                    resolve(winner.value, index, parent, native[index], self.resource)?;
                if let Some(rule) = winner.rule {
                    if !state.evidence.contains(&self.css.rules[rule].evidence) {
                        state.evidence.push(self.css.rules[rule].evidence.clone());
                    }
                } else {
                    inline_evidence = true;
                }
            }
        }
        if state.values[..2].iter().any(|value| !value.is_finite()) {
            return Err(error(
                self.resource,
                "css-value",
                "computed emphasis value is not finite",
            ));
        }
        if inline_evidence
            && let Some((_, declarations)) = self.css.inline.iter().find(|(id, _)| *id == node)
        {
            let text = declarations
                .to_css_string(PrinterOptions::default())
                .map_err(|cause| error(self.resource, "css-printer", cause.to_string()))?;
            state.evidence.push(text);
        }
        for text in &parent.evidence {
            if !state.evidence.contains(text) {
                state.evidence.push(text.clone());
            }
        }
        state.decoration = decoration(self.tree, node, parent, &state, self.resource)?;
        if !state.decoration {
            state
                .emphasis
                .extend(parent.emphasis.iter().copied().filter(|scope| {
                    let index = if scope.role == SemanticRole::Bold {
                        0
                    } else {
                        1
                    };
                    state.values[index] > scope.threshold
                }));
            for (index, role) in [(0, SemanticRole::Bold), (1, SemanticRole::Italic)] {
                if state.values[index] > parent.values[index] {
                    state.emphasis.push(EmphasisScope {
                        origin: node,
                        role,
                        threshold: parent.values[index],
                    });
                }
            }
        }
        Ok(state)
    }
}
fn cascade(
    winners: &mut [Option<Winner>; COUNT],
    projection: &Projection,
    inline: bool,
    specificity: u32,
    order: usize,
    rule: Option<usize>,
) {
    for (index, value) in projection.values.iter().enumerate() {
        if let Some((value, important)) = value {
            let candidate = Winner {
                value: *value,
                priority: (*important, inline, specificity, order),
                rule,
            };
            if winners[index].is_none_or(|winner| candidate.priority >= winner.priority) {
                winners[index] = Some(candidate);
            }
        }
    }
}
fn native_values(tag: &str, parent: &State) -> [f32; COUNT] {
    let mut values = [
        parent.values[0],
        parent.values[1],
        0.0,
        parent.values[3],
        0.0,
        0.0,
    ];
    match tag {
        "b" | "strong" => values[0] = bolder(parent.values[0]),
        "i" | "em" => values[1] = 1.0,
        "sup" => values[2] = 1.0,
        "sub" => values[2] = -1.0,
        _ => {}
    }
    values
}
fn bolder(weight: f32) -> f32 {
    if weight < 350.0 {
        400.0
    } else if weight < 550.0 {
        700.0
    } else {
        900.0
    }
}
fn resolve(
    value: Value,
    index: usize,
    parent: &State,
    native: f32,
    resource: &str,
) -> Result<f32, BookError> {
    let initial = State::default().values[index];
    match value {
        Value::Number(value) => Ok(value),
        Value::Relative(value) => Ok(value * parent.values[index]),
        Value::Bolder => Ok(bolder(parent.values[0])),
        Value::Lighter => Ok(if parent.values[0] < 550.0 {
            100.0
        } else if parent.values[0] < 750.0 {
            400.0
        } else {
            700.0
        }),
        Value::Initial => Ok(initial),
        Value::Inherit => Ok(parent.values[index]),
        Value::Unset => Ok(if matches!(index, 0 | 1 | 3) {
            parent.values[index]
        } else {
            initial
        }),
        Value::Revert => Ok(native),
        Value::Unknown if index == 3 => Ok(f32::NAN),
        Value::Unknown => Err(error(
            resource,
            "css-value",
            "winning semantic property has undetermined value",
        )),
    }
}
fn classify(tree: &SourceTree, node: NodeId, parent: &State, state: &State) -> ElementStyle {
    let mut roles = Vec::new();
    if state.values[0] > parent.values[0] {
        roles.push(SemanticRole::Bold);
    }
    if state.values[1] > parent.values[1] {
        roles.push(SemanticRole::Italic);
    }
    match state.values[2] {
        1.0 => roles.push(SemanticRole::Superscript),
        -1.0 => roles.push(SemanticRole::Subscript),
        _ => {}
    }
    roles.extend(
        tag_semantics(tree.local_name(node).unwrap_or(""))
            .iter()
            .copied()
            .filter(|role| matches!(role, SemanticRole::Link | SemanticRole::Literal)),
    );
    if state.decoration {
        roles.retain(|role| !matches!(role, SemanticRole::Bold | SemanticRole::Italic));
    }
    let required = !roles.is_empty();
    let active_emphasis = state
        .emphasis
        .iter()
        .map(|scope| (scope.origin, scope.role))
        .collect();
    ElementStyle {
        node,
        roles,
        active_emphasis,
        required,
        decoration: state.decoration,
        evidence: state.evidence.clone(),
    }
}
fn decoration(
    tree: &SourceTree,
    node: NodeId,
    parent: &State,
    state: &State,
    resource: &str,
) -> Result<bool, BookError> {
    let own_character = tree.element_children(node).next().is_none()
        && tree.nodes[node.index()]
            .text
            .as_deref()
            .is_some_and(|text| text.trim().chars().count() == 1);
    let geometry = state.values[4] != 0.0 || state.values[5] > 1.0;
    if own_character && geometry && (state.values[3].is_nan() || parent.values[3].is_nan()) {
        return Err(error(
            resource,
            "css-decoration",
            "initial-letter size is undetermined",
        ));
    }
    Ok(own_character
        && (state.values[5] > 1.0
            || (state.values[4] != 0.0 && state.values[3] >= parent.values[3] * 1.5)))
}
